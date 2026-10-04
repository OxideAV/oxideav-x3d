//! [`Scene3D`] → X3D 4.0 XML ([`Mesh3DEncoder`]).
//!
//! * Nodes become `Transform`s (TRS; matrices are decomposed) named by
//!   a sanitised, unique `DEF`.
//! * Each mesh primitive becomes a `Shape`: triangles (any triangle
//!   topology) as `IndexedTriangleSet` with `Coordinate`, `Normal`,
//!   `TextureCoordinate` (v flipped back to X3D's lower-left origin;
//!   several sets via `MultiTextureCoordinate` + `mapping`) and
//!   `Color` / `ColorRGBA`; lines as `IndexedLineSet`; points as
//!   `PointSet`. Shared meshes, materials and textures are shared with
//!   `DEF`/`USE`.
//! * Materials become `PhysicalMaterial` (or `UnlitMaterial`, or a
//!   Phong `Material` when the decoder recorded one in
//!   `extras["x3d:material"]`), with `alphaMode` / `alphaCutoff`.
//!   External textures keep their URI; embedded ones become `data:`
//!   URIs.
//! * Cameras → `Viewpoint` / `OrthoViewpoint`; lights →
//!   `DirectionalLight` / `PointLight` / `SpotLight` (global, with
//!   inverse-square attenuation like glTF punctual lights).
//! * Animations → one `TimeSensor` each plus `PositionInterpolator` /
//!   `OrientationInterpolator` (translation, scale, rotation) and
//!   `CoordinateInterpolator` (morph weights, evaluated to positions)
//!   wired with `ROUTE`s. Step interpolation is emulated with
//!   duplicated keys; cubic splines are sampled at their keyframes.
//! * Scene units, `x3d:environment` bindables (WorldInfo,
//!   NavigationInfo, ...), node metadata and header `meta` recorded by
//!   the decoder are written back.
//!
//! Not encoded: skins (meshes are written in bind pose), KHR texture
//! transforms, material extensions beyond unlit.

use std::collections::{HashMap, HashSet};

use oxideav_mesh3d::{
    AlphaMode, AnimationProperty, AnimationValues, Camera, ImageData, Indices, Interpolation,
    Light, Material, Mesh3DEncoder, MeshId, NodeId, Primitive, Scene3D, Texture, Topology,
    Transform, Unit, WrapMode,
};
use serde_json::Value;

use crate::convert::appearance::base64_encode;
use crate::convert::math::axis_angle_from_quat;
use crate::document::{ComponentDecl, NodeIdx, Route, UnitDecl, X3dDocument, X3dNode};
use crate::field::{parse_xml_value, FieldData, FieldType, FieldValue};
use crate::nodes;

/// X3D XML encoder.
#[derive(Clone, Debug, Default)]
pub struct X3dEncoder {
    gzip: bool,
    classic: bool,
}

impl X3dEncoder {
    /// Encoder producing plain `.x3d` XML.
    pub fn new() -> Self {
        Self::default()
    }

    /// Produce gzip-compressed output (`.x3dz`).
    pub fn with_gzip(mut self, gzip: bool) -> Self {
        self.gzip = gzip;
        self
    }

    /// Write the ClassicVRML encoding (`.x3dv`) instead of XML.
    pub fn with_classic(mut self, classic: bool) -> Self {
        self.classic = classic;
        self
    }

    /// Encode, reporting the crate-local error type.
    pub fn encode_scene(&self, scene: &Scene3D) -> crate::Result<Vec<u8>> {
        let doc = scene_to_document(scene);
        let text = if self.classic {
            crate::write_classic(&doc).into_bytes()
        } else {
            crate::write_xml(&doc).into_bytes()
        };
        if self.gzip {
            crate::gzip(&text)
        } else {
            Ok(text)
        }
    }
}

impl Mesh3DEncoder for X3dEncoder {
    fn encode(&mut self, scene: &Scene3D) -> oxideav_mesh3d::Result<Vec<u8>> {
        self.encode_scene(scene).map_err(Into::into)
    }
}

fn fv_floats(ty: FieldType, v: Vec<f32>) -> FieldValue {
    FieldValue {
        ty,
        data: FieldData::Float(v),
    }
}

fn fv3(ty: FieldType, v: [f32; 3]) -> FieldValue {
    fv_floats(ty, v.to_vec())
}

fn fv_f(v: f32) -> FieldValue {
    fv_floats(FieldType::SFFloat, vec![v])
}

fn fv_bool(b: bool) -> FieldValue {
    FieldValue {
        ty: FieldType::SFBool,
        data: FieldData::Bool(vec![b]),
    }
}

fn fv_ints(v: Vec<i32>) -> FieldValue {
    FieldValue {
        ty: FieldType::MFInt32,
        data: FieldData::Int32(v),
    }
}

fn fv_strs(ty: FieldType, v: Vec<String>) -> FieldValue {
    FieldValue {
        ty,
        data: FieldData::String(v),
    }
}

fn flat<const N: usize>(v: &[[f32; N]]) -> Vec<f32> {
    v.iter().flat_map(|a| a.iter().copied()).collect()
}

/// Interchange-profile component levels (ISO/IEC 19775-1 Table B.2).
const INTERCHANGE: &[(&str, u32)] = &[
    ("Core", 1),
    ("Time", 1),
    ("Networking", 1),
    ("Grouping", 1),
    ("Rendering", 3),
    ("Shape", 1),
    ("Geometry3D", 2),
    ("Lighting", 1),
    ("Texturing", 2),
    ("Interpolation", 2),
    ("Navigation", 1),
    ("EnvironmentalEffects", 1),
];

/// Build an [`X3dDocument`] from a scene.
pub fn scene_to_document(scene: &Scene3D) -> X3dDocument {
    let mut e = Enc {
        scene,
        doc: X3dDocument::new(),
        names: HashSet::new(),
        node_map: HashMap::new(),
        shape_map: HashMap::new(),
        app_map: HashMap::new(),
        tex_map: HashMap::new(),
        coord_of_mesh: HashMap::new(),
        visiting: HashSet::new(),
    };
    e.header();
    e.environment();
    for &r in &scene.roots {
        if let Some(i) = e.node(r, 0) {
            e.doc.scene.roots.push(i);
        }
    }
    e.animations();
    e.components();
    e.doc
}

struct Enc<'s> {
    scene: &'s Scene3D,
    doc: X3dDocument,
    names: HashSet<String>,
    node_map: HashMap<NodeId, NodeIdx>,
    shape_map: HashMap<(MeshId, usize), NodeIdx>,
    app_map: HashMap<(Option<u32>, bool), NodeIdx>,
    tex_map: HashMap<u32, NodeIdx>,
    coord_of_mesh: HashMap<MeshId, NodeIdx>,
    visiting: HashSet<NodeId>,
}

/// Make `name` a valid, unique DEF (XML NCName-ish).
fn sanitise(name: &str) -> String {
    let mut s: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() || s.starts_with(|c: char| c.is_ascii_digit() || c == '-' || c == '.') {
        s.insert(0, '_');
    }
    s
}

impl Enc<'_> {
    fn unique(&mut self, base: &str) -> String {
        let b = sanitise(base);
        let mut cand = b.clone();
        let mut k = 1;
        while self.names.contains(&cand) {
            k += 1;
            cand = format!("{b}_{k}");
        }
        self.names.insert(cand.clone());
        cand
    }

    fn add(&mut self, n: X3dNode) -> NodeIdx {
        self.doc.add_node(n)
    }

    fn header(&mut self) {
        let x = self.scene.extras.get("x3d");
        if let Some(meta) = x.and_then(|x| x.get("meta")).and_then(Value::as_array) {
            for m in meta {
                if let (Some(n), Some(c)) = (
                    m.get("name").and_then(Value::as_str),
                    m.get("content").and_then(Value::as_str),
                ) {
                    self.doc.meta.push((n.to_string(), c.to_string()));
                }
            }
        }
        if !self.doc.meta.iter().any(|(n, _)| n == "generator") {
            self.doc
                .meta
                .push(("generator".into(), "oxideav-x3d".into()));
        }
        let unit = match self.scene.unit {
            Unit::Metres => None,
            Unit::Centimetres => Some(("centimetres", 0.01)),
            Unit::Millimetres => Some(("millimetres", 0.001)),
            Unit::Inches => Some(("inches", 0.0254)),
            Unit::Feet => Some(("feet", 0.3048)),
            Unit::Yards => Some(("yards", 0.9144)),
        };
        if let Some((name, f)) = unit {
            self.doc.units.push(UnitDecl {
                category: "length".into(),
                name: name.into(),
                conversion_factor: f,
            });
        }
    }

    /// Rebuild a node from a decoder JSON record (`{"type": ..,
    /// field: value}`), typing values through the node table.
    fn node_from_json(&mut self, v: &Value, depth: usize) -> Option<NodeIdx> {
        let obj = v.as_object()?;
        let ty = obj.get("type")?.as_str()?;
        let table = nodes::lookup(ty)?;
        let mut n = X3dNode::new(ty);
        for (k, val) in obj {
            if k == "type" || k == "DEF" {
                continue;
            }
            let Some(fd) = table.field(k) else { continue };
            if matches!(fd.ty, FieldType::SFNode | FieldType::MFNode) {
                if ty == "MetadataSet" && k == "value" && depth < 32 {
                    let kids: Vec<NodeIdx> = val
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(|c| self.node_from_json(c, depth + 1))
                        .collect();
                    n.set("value", FieldValue::nodes(FieldType::MFNode, kids));
                }
                continue;
            }
            if let Some(fv) = json_to_field(fd.ty, val) {
                n.set(k.clone(), fv);
            }
        }
        Some(self.add(n))
    }

    fn environment(&mut self) {
        let Some(list) = self
            .scene
            .extras
            .get("x3d:environment")
            .and_then(Value::as_array)
        else {
            return;
        };
        for v in list {
            if let Some(i) = self.node_from_json(v, 0) {
                self.doc.scene.roots.push(i);
            }
        }
    }

    fn node(&mut self, id: NodeId, depth: usize) -> Option<NodeIdx> {
        if depth > 512 || !self.visiting.insert(id) {
            return None;
        }
        let r = self.node_inner(id, depth);
        self.visiting.remove(&id);
        r
    }

    fn node_inner(&mut self, id: NodeId, depth: usize) -> Option<NodeIdx> {
        let sn = self.scene.node(id)?;
        // A bare mesh holder (identity transform, single primitive, no
        // children) is written as the Shape itself — the inverse of the
        // decoder's Shape → node mapping, so round trips are stable.
        let identity = match sn.transform {
            Transform::Trs {
                translation,
                rotation,
                scale,
            } => translation == [0.0; 3] && rotation == [0.0, 0.0, 0.0, 1.0] && scale == [1.0; 3],
            Transform::Matrix(m) => m == crate::convert::math::IDENTITY,
        };
        if identity
            && sn.children.is_empty()
            && sn.camera.is_none()
            && sn.light.is_none()
            && sn.skin.is_none()
            && !sn.extras.contains_key("x3d:metadata")
        {
            if let Some(m) = sn.mesh {
                let existing_def = self
                    .shape_map
                    .get(&(m, 0))
                    .and_then(|i| self.doc.nodes[i.index()].def.clone());
                let fresh = !self.shape_map.contains_key(&(m, 0));
                let same_name = match (&sn.name, &existing_def) {
                    (None, _) => true,
                    (Some(n), Some(d)) => sanitise(n) == *d,
                    _ => false,
                };
                if self.scene.mesh(m).map(|x| x.primitives.len()) == Some(1) && (fresh || same_name)
                {
                    let i = self.shape(m, 0, sn.name.as_deref().filter(|_| fresh))?;
                    self.node_map.insert(id, i);
                    return Some(i);
                }
            }
        }
        let mut t = X3dNode::new("Transform");
        if let Some(name) = &sn.name {
            t.def = Some(self.unique(name));
        }
        let (tr, rot, sc) = match sn.transform {
            Transform::Trs {
                translation,
                rotation,
                scale,
            } => (translation, rotation, scale),
            Transform::Matrix(m) => match Transform::from_matrix(m) {
                Transform::Trs {
                    translation,
                    rotation,
                    scale,
                } => (translation, rotation, scale),
                Transform::Matrix(_) => ([0.0; 3], [0.0, 0.0, 0.0, 1.0], [1.0; 3]),
            },
        };
        if tr != [0.0; 3] {
            t.set("translation", fv3(FieldType::SFVec3f, tr));
        }
        let aa = axis_angle_from_quat(rot);
        if aa[3] != 0.0 {
            t.set("rotation", fv_floats(FieldType::SFRotation, aa.to_vec()));
        }
        if sc != [1.0; 3] {
            t.set("scale", fv3(FieldType::SFVec3f, sc));
        }
        if let Some(md) = sn.extras.get("x3d:metadata") {
            if let Some(m) = self.node_from_json(md, 0) {
                t.set("metadata", FieldValue::nodes(FieldType::SFNode, vec![m]));
            }
        }
        let mut kids = Vec::new();
        if let Some(c) = sn.camera {
            if let Some(i) = self.camera(c.0 as usize, sn) {
                kids.push(i);
            }
        }
        if let Some(l) = sn.light {
            if let Some(i) = self.light(l.0 as usize) {
                kids.push(i);
            }
        }
        if let Some(m) = sn.mesh {
            if let Some(mesh) = self.scene.mesh(m) {
                for p in 0..mesh.primitives.len() {
                    if let Some(i) = self.shape(m, p, None) {
                        kids.push(i);
                    }
                }
            }
        }
        let children = sn.children.clone();
        let idx = self.add(t);
        self.node_map.insert(id, idx);
        for c in children {
            if let Some(i) = self.node(c, depth + 1) {
                kids.push(i);
            }
        }
        if !kids.is_empty() {
            self.doc.nodes[idx.index()].set("children", FieldValue::nodes(FieldType::MFNode, kids));
        }
        Some(idx)
    }

    fn camera(&mut self, c: usize, owner: &oxideav_mesh3d::Node) -> Option<NodeIdx> {
        let cam = *self.scene.cameras.get(c)?;
        let mut v = match cam {
            Camera::Perspective {
                yfov, znear, zfar, ..
            } => {
                let mut v = X3dNode::new("Viewpoint");
                v.set("fieldOfView", fv_f(yfov));
                v.set("nearDistance", fv_f(znear));
                if let Some(f) = zfar {
                    v.set("farDistance", fv_f(f));
                }
                v
            }
            Camera::Orthographic {
                xmag,
                ymag,
                znear,
                zfar,
            } => {
                let mut v = X3dNode::new("OrthoViewpoint");
                v.set(
                    "fieldOfView",
                    fv_floats(FieldType::MFFloat, vec![-xmag, -ymag, xmag, ymag]),
                );
                v.set("nearDistance", fv_f(znear));
                v.set("farDistance", fv_f(zfar));
                v
            }
        };
        v.set("position", fv3(FieldType::SFVec3f, [0.0; 3]));
        if let Some(d) = owner
            .extras
            .get("x3d:viewpoint")
            .and_then(|x| x.get("description"))
            .and_then(Value::as_str)
        {
            v.set("description", FieldValue::sf_string(d));
        } else if let Some(n) = &owner.name {
            v.set("description", FieldValue::sf_string(n.clone()));
        }
        Some(self.add(v))
    }

    fn light(&mut self, l: usize) -> Option<NodeIdx> {
        let light = *self.scene.lights.get(l)?;
        let n = match light {
            Light::Directional { color, intensity } => {
                let mut n = X3dNode::new("DirectionalLight");
                n.set("color", fv3(FieldType::SFColor, color));
                n.set("intensity", fv_f(intensity));
                n.set("direction", fv3(FieldType::SFVec3f, [0.0, 0.0, -1.0]));
                n.set("global", fv_bool(true));
                n
            }
            Light::Point {
                color,
                intensity,
                range,
            } => {
                let mut n = X3dNode::new("PointLight");
                n.set("color", fv3(FieldType::SFColor, color));
                n.set("intensity", fv_f(intensity));
                n.set("attenuation", fv3(FieldType::SFVec3f, [0.0, 0.0, 1.0]));
                n.set("radius", fv_f(range.unwrap_or(1.0e6)));
                n
            }
            Light::Spot {
                color,
                intensity,
                range,
                inner_cone_angle,
                outer_cone_angle,
            } => {
                let mut n = X3dNode::new("SpotLight");
                n.set("color", fv3(FieldType::SFColor, color));
                n.set("intensity", fv_f(intensity));
                n.set("attenuation", fv3(FieldType::SFVec3f, [0.0, 0.0, 1.0]));
                n.set("radius", fv_f(range.unwrap_or(1.0e6)));
                n.set("direction", fv3(FieldType::SFVec3f, [0.0, 0.0, -1.0]));
                n.set("beamWidth", fv_f(inner_cone_angle));
                n.set("cutOffAngle", fv_f(outer_cone_angle));
                n
            }
        };
        Some(self.add(n))
    }

    fn shape(&mut self, m: MeshId, p: usize, def: Option<&str>) -> Option<NodeIdx> {
        if let Some(&i) = self.shape_map.get(&(m, p)) {
            return Some(i);
        }
        let mesh = self.scene.mesh(m)?;
        let prim = mesh.primitives.get(p)?;
        let material = prim
            .material
            .and_then(|id| self.scene.materials.get(id.0 as usize));
        let geom = self.geometry(m, prim, material)?;
        let app = self.appearance(prim, material);
        let mut s = X3dNode::new("Shape");
        if let Some(d) = def {
            s.def = Some(self.unique(d));
        } else if let Some(name) = &mesh.name {
            let base = if mesh.primitives.len() > 1 {
                format!("{name}_{p}")
            } else {
                name.clone()
            };
            s.def = Some(self.unique(&base));
        }
        if let Some(a) = app {
            s.set("appearance", FieldValue::nodes(FieldType::SFNode, vec![a]));
        }
        s.set("geometry", FieldValue::nodes(FieldType::SFNode, vec![geom]));
        let i = self.add(s);
        self.shape_map.insert((m, p), i);
        Some(i)
    }

    fn geometry(
        &mut self,
        mesh_id: MeshId,
        prim: &Primitive,
        material: Option<&Material>,
    ) -> Option<NodeIdx> {
        if prim.positions.is_empty() {
            return None;
        }
        let mut coord = X3dNode::new("Coordinate");
        coord.set(
            "point",
            fv_floats(FieldType::MFVec3f, flat(&prim.positions)),
        );
        let coord_i = self.add(coord);
        self.coord_of_mesh.entry(mesh_id).or_insert(coord_i);
        let colors = prim
            .colors
            .first()
            .filter(|c| c.len() == prim.positions.len());
        let color_node = colors.map(|c| {
            if c.iter().all(|x| x[3] == 1.0) {
                let mut n = X3dNode::new("Color");
                n.set(
                    "color",
                    fv_floats(
                        FieldType::MFColor,
                        c.iter().flat_map(|x| [x[0], x[1], x[2]]).collect(),
                    ),
                );
                n
            } else {
                let mut n = X3dNode::new("ColorRGBA");
                n.set("color", fv_floats(FieldType::MFColorRGBA, flat(c)));
                n
            }
        });
        let set_common = |e: &mut Self, g: &mut X3dNode, color_node: Option<X3dNode>| {
            if let Some(c) = color_node {
                let ci = e.add(c);
                g.set("color", FieldValue::nodes(FieldType::SFNode, vec![ci]));
            }
            g.set("coord", FieldValue::nodes(FieldType::SFNode, vec![coord_i]));
        };
        let n_verts = prim.positions.len() as u32;
        let index_list: Vec<u32> = match &prim.indices {
            Some(Indices::U16(v)) => v
                .iter()
                .map(|&i| i as u32)
                .filter(|&i| i < n_verts)
                .collect(),
            Some(Indices::U32(v)) => v.iter().copied().filter(|&i| i < n_verts).collect(),
            None => (0..n_verts).collect(),
        };
        let g = match prim.topology {
            Topology::Triangles | Topology::TriangleStrip | Topology::TriangleFan => {
                let tris: Vec<i32> = prim
                    .triangle_indices()
                    .into_iter()
                    .filter(|t| t.iter().all(|&i| i < n_verts))
                    .flat_map(|t| t.map(|i| i as i32))
                    .collect();
                if tris.is_empty() {
                    return None;
                }
                let mut g = X3dNode::new("IndexedTriangleSet");
                g.set("index", fv_ints(tris));
                let double = material.map(|m| m.double_sided).unwrap_or(false);
                if double {
                    g.set("solid", fv_bool(false));
                }
                set_common(self, &mut g, color_node);
                if let Some(nrm) = prim
                    .normals
                    .as_ref()
                    .filter(|n| n.len() == prim.positions.len())
                {
                    let mut nn = X3dNode::new("Normal");
                    nn.set("vector", fv_floats(FieldType::MFVec3f, flat(nrm)));
                    let ni = self.add(nn);
                    g.set("normal", FieldValue::nodes(FieldType::SFNode, vec![ni]));
                }
                let sets: Vec<&Vec<[f32; 2]>> = prim
                    .uvs
                    .iter()
                    .filter(|s| s.len() == prim.positions.len())
                    .collect();
                let tc_node = |e: &mut Self, set: &Vec<[f32; 2]>, mapping: Option<String>| {
                    let mut t = X3dNode::new("TextureCoordinate");
                    t.set(
                        "point",
                        fv_floats(
                            FieldType::MFVec2f,
                            set.iter().flat_map(|uv| [uv[0], 1.0 - uv[1]]).collect(),
                        ),
                    );
                    if let Some(m) = mapping {
                        t.set("mapping", FieldValue::sf_string(m));
                    }
                    e.add(t)
                };
                if sets.len() == 1 {
                    let t = tc_node(self, sets[0], None);
                    g.set("texCoord", FieldValue::nodes(FieldType::SFNode, vec![t]));
                } else if sets.len() > 1 {
                    let kids: Vec<NodeIdx> = sets
                        .iter()
                        .enumerate()
                        .map(|(k, s)| tc_node(self, s, Some(format!("TEXCOORD_{k}"))))
                        .collect();
                    let mut mt = X3dNode::new("MultiTextureCoordinate");
                    mt.set("texCoord", FieldValue::nodes(FieldType::MFNode, kids));
                    let mi = self.add(mt);
                    g.set("texCoord", FieldValue::nodes(FieldType::SFNode, vec![mi]));
                }
                g
            }
            Topology::Lines | Topology::LineStrip | Topology::LineLoop => {
                let mut ci = Vec::new();
                match prim.topology {
                    Topology::Lines => {
                        for p in index_list.chunks_exact(2) {
                            ci.extend_from_slice(&[p[0] as i32, p[1] as i32, -1]);
                        }
                    }
                    _ => {
                        ci.extend(index_list.iter().map(|&i| i as i32));
                        if prim.topology == Topology::LineLoop {
                            if let Some(&f) = index_list.first() {
                                ci.push(f as i32);
                            }
                        }
                    }
                }
                if ci.is_empty() {
                    return None;
                }
                let mut g = X3dNode::new("IndexedLineSet");
                g.set("coordIndex", fv_ints(ci));
                set_common(self, &mut g, color_node);
                g
            }
            Topology::Points => {
                let mut g = X3dNode::new("PointSet");
                if prim.indices.is_some() {
                    // PointSet is not indexed: expand.
                    let pts: Vec<[f32; 3]> = index_list
                        .iter()
                        .map(|&i| prim.positions[i as usize])
                        .collect();
                    self.doc.nodes[coord_i.index()]
                        .set("point", fv_floats(FieldType::MFVec3f, flat(&pts)));
                    if let Some(c) = colors {
                        let cs: Vec<[f32; 4]> = index_list.iter().map(|&i| c[i as usize]).collect();
                        let mut n = X3dNode::new("ColorRGBA");
                        n.set("color", fv_floats(FieldType::MFColorRGBA, flat(&cs)));
                        set_common(self, &mut g, Some(n));
                    } else {
                        set_common(self, &mut g, None);
                    }
                } else {
                    set_common(self, &mut g, color_node);
                }
                g
            }
        };
        Some(self.add(g))
    }

    fn texture(&mut self, id: u32) -> Option<NodeIdx> {
        if let Some(&i) = self.tex_map.get(&id) {
            return Some(i);
        }
        let tex: &Texture = self.scene.textures.get(id as usize)?;
        let url = match &tex.image {
            ImageData::External { uri, .. } => uri.clone(),
            ImageData::Source(src) => {
                let mut r = src.open().ok()?;
                let mut b = Vec::new();
                std::io::Read::read_to_end(&mut r, &mut b).ok()?;
                let mime = src.mime().unwrap_or("application/octet-stream");
                format!("data:{mime};base64,{}", base64_encode(&b))
            }
            #[allow(unreachable_patterns)]
            _ => return None,
        };
        let mut n = X3dNode::new("ImageTexture");
        if let Some(name) = &tex.name {
            n.def = Some(self.unique(name));
        }
        n.set("url", fv_strs(FieldType::MFString, vec![url]));
        let s = tex.sampler;
        if s.wrap_s == WrapMode::ClampToEdge {
            n.set("repeatS", fv_bool(false));
        }
        if s.wrap_t == WrapMode::ClampToEdge {
            n.set("repeatT", fv_bool(false));
        }
        let mirrored = s.wrap_s == WrapMode::MirroredRepeat || s.wrap_t == WrapMode::MirroredRepeat;
        if mirrored || s.mag_filter.is_some() || s.min_filter.is_some() {
            let mut tp = X3dNode::new("TextureProperties");
            let mode = |w: WrapMode| match w {
                WrapMode::Repeat => "REPEAT",
                WrapMode::MirroredRepeat => "MIRRORED_REPEAT",
                WrapMode::ClampToEdge => "CLAMP_TO_EDGE",
            };
            tp.set("boundaryModeS", FieldValue::sf_string(mode(s.wrap_s)));
            tp.set("boundaryModeT", FieldValue::sf_string(mode(s.wrap_t)));
            if let Some(m) = s.mag_filter {
                tp.set(
                    "magnificationFilter",
                    FieldValue::sf_string(match m {
                        oxideav_mesh3d::MagFilter::Nearest => "NEAREST_PIXEL",
                        oxideav_mesh3d::MagFilter::Linear => "AVG_PIXEL",
                    }),
                );
            }
            if let Some(m) = s.min_filter {
                use oxideav_mesh3d::MinFilter as M;
                tp.set(
                    "minificationFilter",
                    FieldValue::sf_string(match m {
                        M::Nearest => "NEAREST_PIXEL",
                        M::Linear => "AVG_PIXEL",
                        M::NearestMipNearest => "NEAREST_PIXEL_NEAREST_MIPMAP",
                        M::LinearMipNearest => "AVG_PIXEL_NEAREST_MIPMAP",
                        M::NearestMipLinear => "NEAREST_PIXEL_AVG_MIPMAP",
                        M::LinearMipLinear => "AVG_PIXEL_AVG_MIPMAP",
                    }),
                );
                tp.set("generateMipMaps", fv_bool(m.uses_mipmaps()));
            }
            let ti = self.add(tp);
            n.set(
                "textureProperties",
                FieldValue::nodes(FieldType::SFNode, vec![ti]),
            );
        }
        let i = self.add(n);
        self.tex_map.insert(id, i);
        Some(i)
    }

    fn appearance(&mut self, prim: &Primitive, material: Option<&Material>) -> Option<NodeIdx> {
        let mid = prim.material.map(|m| m.0);
        let lines = !matches!(
            prim.topology,
            Topology::Triangles | Topology::TriangleStrip | Topology::TriangleFan
        );
        if let Some(&i) = self.app_map.get(&(mid, lines)) {
            return Some(i);
        }
        let m = material?;
        let x3d_type = m
            .extras
            .get("x3d:material")
            .and_then(|x| x.get("type"))
            .cloned();
        if x3d_type == Some(Value::Null) && m.base_color_texture.is_none() {
            // Decoder saw no material: X3D unlit white default.
            return None;
        }
        let tex = |e: &mut Self, r: &Option<oxideav_mesh3d::TextureRef>| {
            r.as_ref().and_then(|r| e.texture(r.texture.0))
        };
        let transparency = (1.0 - m.base_color[3]).clamp(0.0, 1.0);
        let rgb = [m.base_color[0], m.base_color[1], m.base_color[2]];
        let mut mat_kids: Vec<(String, NodeIdx)> = Vec::new();
        let mat = if m.ext.unlit || x3d_type == Some(Value::Null) {
            let mut n = X3dNode::new("UnlitMaterial");
            n.set("emissiveColor", fv3(FieldType::SFColor, rgb));
            if let Some(t) = tex(self, &m.base_color_texture) {
                mat_kids.push(("emissiveTexture".into(), t));
            }
            n
        } else if x3d_type.as_ref().and_then(Value::as_str) == Some("Material")
            || x3d_type.as_ref().and_then(Value::as_str) == Some("TwoSidedMaterial")
        {
            let x = &m.extras["x3d:material"];
            let mut n = X3dNode::new("Material");
            let dc = x
                .get("diffuseColor")
                .and_then(|v| json_to_field(FieldType::SFColor, v))
                .unwrap_or_else(|| fv3(FieldType::SFColor, rgb));
            n.set("diffuseColor", dc);
            for (k, ty) in [
                ("specularColor", FieldType::SFColor),
                ("shininess", FieldType::SFFloat),
                ("ambientIntensity", FieldType::SFFloat),
            ] {
                if let Some(v) = x.get(k).and_then(|v| json_to_field(ty, v)) {
                    n.set(k, v);
                }
            }
            n.set("emissiveColor", fv3(FieldType::SFColor, m.emissive_factor));
            if let Some(t) = tex(self, &m.base_color_texture) {
                mat_kids.push(("diffuseTexture".into(), t));
            }
            if let Some(t) = tex(self, &m.emissive_texture) {
                mat_kids.push(("emissiveTexture".into(), t));
            }
            if let Some(t) = tex(self, &m.normal_texture) {
                mat_kids.push(("normalTexture".into(), t));
            }
            if let Some(t) = tex(self, &m.occlusion_texture) {
                mat_kids.push(("occlusionTexture".into(), t));
            }
            n
        } else {
            let mut n = X3dNode::new("PhysicalMaterial");
            n.set("baseColor", fv3(FieldType::SFColor, rgb));
            n.set("metallic", fv_f(m.metallic));
            n.set("roughness", fv_f(m.roughness));
            if m.emissive_factor != [0.0; 3] {
                n.set("emissiveColor", fv3(FieldType::SFColor, m.emissive_factor));
            }
            if let Some(t) = tex(self, &m.base_color_texture) {
                mat_kids.push(("baseTexture".into(), t));
            }
            if let Some(t) = tex(self, &m.metallic_roughness_texture) {
                mat_kids.push(("metallicRoughnessTexture".into(), t));
            }
            if let Some(t) = tex(self, &m.emissive_texture) {
                mat_kids.push(("emissiveTexture".into(), t));
            }
            if let Some(t) = tex(self, &m.normal_texture) {
                mat_kids.push(("normalTexture".into(), t));
            }
            if let Some(t) = tex(self, &m.occlusion_texture) {
                mat_kids.push(("occlusionTexture".into(), t));
            }
            n
        };
        let mut mat = mat;
        if transparency > 0.0 {
            mat.set("transparency", fv_f(transparency));
        }
        if mat.type_name != "UnlitMaterial" {
            if m.normal_texture.is_some() && m.normal_scale != 1.0 {
                mat.set("normalScale", fv_f(m.normal_scale));
            }
            if m.occlusion_texture.is_some() && m.occlusion_strength != 1.0 {
                mat.set("occlusionStrength", fv_f(m.occlusion_strength));
            }
        }
        for (f, t) in mat_kids {
            mat.set(f, FieldValue::nodes(FieldType::SFNode, vec![t]));
        }
        if let Some(name) = &m.name {
            mat.def = Some(self.unique(&format!("{name}_mat")));
        }
        let mi = self.add(mat);
        let mut app = X3dNode::new("Appearance");
        if let Some(name) = &m.name {
            app.def = Some(self.unique(name));
        }
        match m.alpha_mode {
            AlphaMode::Opaque => {
                if transparency > 0.0 {
                    app.set("alphaMode", FieldValue::sf_string("OPAQUE"));
                }
            }
            AlphaMode::Blend => {
                if transparency == 0.0 {
                    app.set("alphaMode", FieldValue::sf_string("BLEND"));
                }
            }
            AlphaMode::Mask { cutoff } => {
                app.set("alphaMode", FieldValue::sf_string("MASK"));
                app.set("alphaCutoff", fv_f(cutoff));
            }
        }
        app.set("material", FieldValue::nodes(FieldType::SFNode, vec![mi]));
        let ai = self.add(app);
        self.app_map.insert((mid, lines), ai);
        Some(ai)
    }

    fn animations(&mut self) {
        let scene = self.scene;
        for (ai, anim) in scene.animations.iter().enumerate() {
            let end = anim
                .channels
                .iter()
                .filter_map(|c| c.sampler.keyframes.last().copied())
                .fold(0.0f32, f32::max);
            let cycle = if end > 0.0 { end } else { 1.0 };
            let mut ts = X3dNode::new("TimeSensor");
            let base = anim
                .name
                .clone()
                .unwrap_or_else(|| format!("Animation{ai}"));
            let ts_name = self.unique(&base);
            ts.def = Some(ts_name.clone());
            ts.set(
                "cycleInterval",
                FieldValue {
                    ty: FieldType::SFTime,
                    data: FieldData::Double(vec![cycle as f64]),
                },
            );
            ts.set("loop", fv_bool(true));
            let ts_i = self.add(ts);
            self.doc.scene.roots.push(ts_i);
            for (ci, ch) in anim.channels.iter().enumerate() {
                let Some(&target) = self.node_map.get(&ch.target.node) else {
                    continue;
                };
                let (keys, interp, to_node, to_field) = match self.channel(ch, cycle, target) {
                    Some(x) => x,
                    None => continue,
                };
                let mut n = interp;
                n.set("key", fv_floats(FieldType::MFFloat, keys));
                let iname = self.unique(&format!("{ts_name}_ch{ci}"));
                n.def = Some(iname.clone());
                let ii = self.add(n);
                self.doc.scene.roots.push(ii);
                let to_def = self.ensure_def(to_node);
                self.doc.scene.routes.push(Route {
                    from_node: ts_i,
                    from_def: ts_name.clone(),
                    from_field: "fraction_changed".into(),
                    to_node: ii,
                    to_def: iname.clone(),
                    to_field: "set_fraction".into(),
                });
                self.doc.scene.routes.push(Route {
                    from_node: ii,
                    from_def: iname,
                    from_field: "value_changed".into(),
                    to_node,
                    to_def,
                    to_field: to_field.into(),
                });
            }
        }
    }

    fn ensure_def(&mut self, idx: NodeIdx) -> String {
        if let Some(d) = self.doc.nodes[idx.index()].def.clone() {
            return d;
        }
        let name = self.unique(&format!(
            "{}_{}",
            self.doc.nodes[idx.index()].type_name,
            idx.0
        ));
        self.doc.nodes[idx.index()].def = Some(name.clone());
        name
    }

    /// Interpolator for one channel: `(keys, node, target, field)`.
    fn channel(
        &mut self,
        ch: &oxideav_mesh3d::AnimationChannel,
        cycle: f32,
        target: NodeIdx,
    ) -> Option<(Vec<f32>, X3dNode, NodeIdx, &'static str)> {
        let s = &ch.sampler;
        let n = s.keyframes.len();
        if n == 0 {
            return None;
        }
        // Per-keyframe value rows (cubic: middle of each triple).
        let pick = |i: usize| -> usize {
            if s.interpolation == Interpolation::CubicSpline {
                i * 3 + 1
            } else {
                i
            }
        };
        // (time, row index) sequence with step emulation.
        let mut seq: Vec<(f32, usize)> = Vec::new();
        for i in 0..n {
            let t = s.keyframes[i] / cycle;
            if s.interpolation == Interpolation::Step && i > 0 {
                seq.push((t, pick(i - 1)));
            }
            seq.push((t, pick(i)));
        }
        let keys: Vec<f32> = seq.iter().map(|x| x.0).collect();
        match (ch.target.property, &s.values) {
            (
                AnimationProperty::Translation | AnimationProperty::Scale,
                AnimationValues::Vec3(v),
            ) => {
                let vals: Vec<f32> = seq
                    .iter()
                    .filter_map(|&(_, r)| v.get(r))
                    .flat_map(|x| x.iter().copied())
                    .collect();
                if vals.len() != keys.len() * 3 {
                    return None;
                }
                let mut p = X3dNode::new("PositionInterpolator");
                p.set("keyValue", fv_floats(FieldType::MFVec3f, vals));
                let field = if ch.target.property == AnimationProperty::Translation {
                    "set_translation"
                } else {
                    "set_scale"
                };
                Some((keys, p, target, field))
            }
            (AnimationProperty::Rotation, AnimationValues::Quat(v)) => {
                let vals: Vec<f32> = seq
                    .iter()
                    .filter_map(|&(_, r)| v.get(r))
                    .flat_map(|q| axis_angle_from_quat(*q))
                    .collect();
                if vals.len() != keys.len() * 4 {
                    return None;
                }
                let mut p = X3dNode::new("OrientationInterpolator");
                p.set("keyValue", fv_floats(FieldType::MFRotation, vals));
                Some((keys, p, target, "set_rotation"))
            }
            (AnimationProperty::MorphWeights, AnimationValues::Scalar(w)) => {
                let node = self.scene.node(ch.target.node)?;
                let mesh_id = node.mesh?;
                let mesh = self.scene.mesh(mesh_id)?;
                let prim = mesh.primitives.first()?;
                let nt = prim.targets.len();
                if nt == 0 || w.len() < pick(n - 1) * nt + nt {
                    return None;
                }
                let coord = *self.coord_of_mesh.get(&mesh_id)?;
                let mut vals = Vec::with_capacity(seq.len() * prim.positions.len() * 3);
                for &(_, r) in &seq {
                    let row = &w[r * nt..r * nt + nt];
                    for (vi, base) in prim.positions.iter().enumerate() {
                        let mut p = *base;
                        for (ti, t) in prim.targets.iter().enumerate() {
                            if let Some(d) = t.position.as_ref().and_then(|d| d.get(vi)) {
                                p[0] += row[ti] * d[0];
                                p[1] += row[ti] * d[1];
                                p[2] += row[ti] * d[2];
                            }
                        }
                        vals.extend_from_slice(&p);
                    }
                }
                let mut p = X3dNode::new("CoordinateInterpolator");
                p.set("keyValue", fv_floats(FieldType::MFVec3f, vals));
                Some((keys, p, coord, "set_point"))
            }
            _ => None,
        }
    }

    fn components(&mut self) {
        let mut need: HashMap<&'static str, u32> = HashMap::new();
        for n in &self.doc.nodes {
            if let Some(d) = nodes::lookup(&n.type_name) {
                let base = INTERCHANGE
                    .iter()
                    .find(|(c, _)| *c == d.component)
                    .map(|c| c.1)
                    .unwrap_or(0);
                if d.level > base {
                    let e = need.entry(d.component).or_insert(0);
                    *e = (*e).max(d.level);
                }
            }
        }
        let mut list: Vec<_> = need.into_iter().collect();
        list.sort();
        for (name, level) in list {
            self.doc.components.push(ComponentDecl {
                name: name.into(),
                level,
            });
        }
    }
}

/// JSON (as recorded by the decoder) → typed field value.
fn json_to_field(ty: FieldType, v: &Value) -> Option<FieldValue> {
    let text = match v {
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => {
            if ty.is_mf() && ty == FieldType::MFString {
                crate::field::quote_string(s)
            } else {
                s.clone()
            }
        }
        Value::Array(a) => a
            .iter()
            .map(|x| match x {
                Value::String(s) if ty == FieldType::MFString => crate::field::quote_string(s),
                Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .collect::<Vec<_>>()
            .join(" "),
        _ => return None,
    };
    parse_xml_value(ty, &text).ok()
}
