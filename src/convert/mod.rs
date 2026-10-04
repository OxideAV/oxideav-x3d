//! [`X3dDocument`] → [`Scene3D`] conversion.
//!
//! The X3D transformation hierarchy maps node-for-node onto mesh3d
//! nodes (X3D and the mesh3d model share the right-handed, Y-up,
//! metre convention). Each converted X3D node keeps its `DEF` name as
//! the mesh3d node name; `USE` of a grouping node instantiates the
//! subtree again (mesh3d graphs are trees), while `USE` of a Shape
//! reuses the converted [`Mesh`].
//!
//! What maps where:
//!
//! | X3D | mesh3d |
//! |---|---|
//! | Transform, HAnimJoint, HAnimSite, CADPart | node with TRS (or matrix when `center` / `scaleOrientation` are used) |
//! | Group, StaticGroup, Collision, Anchor, Billboard, CAD*, Layer* | node (extras keep Anchor / Billboard / Collision parameters) |
//! | Switch / LOD | node with only the active / highest-detail child; extras keep `whichChoice` / `range` |
//! | Inline | node with `extras["x3d:inline"]`; contents spliced in when an inline resolver is configured |
//! | Shape + geometry | mesh with one primitive; see [`geometry`] |
//! | Appearance / *Material / textures | material + textures; see [`appearance`] |
//! | Viewpoint / OrthoViewpoint | camera node |
//! | Directional / Point / SpotLight | light node (location / direction become the node transform) |
//! | TimeSensor → *Interpolator → Transform ROUTEs | animations (one per TimeSensor) |
//! | CoordinateInterpolator → Coordinate ROUTE | morph targets + weight animation |
//! | HAnimHumanoid skin binding | skeleton + skin, joints / weights per vertex |
//! | metadata fields | `extras["x3d:metadata"]` |
//! | WorldInfo, NavigationInfo, Background, ... | scene extras |

mod animation;
pub mod appearance;
pub mod geometry;
mod hanim;
pub(crate) mod math;
pub mod proto;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use oxideav_mesh3d::{Camera, Light, Mesh, MeshId, Node, NodeId, Scene3D, Transform, Unit};
use serde_json::{json, Map, Value};

use crate::document::{NodeIdx, NodeKind, X3dDocument, X3dNode};
use crate::error::{Error, Result};
use crate::field::FieldData;
use crate::Limits;
use appearance::{MatKey, MaterialBuilder};
use geometry::{BuildNotes, GeomOptions};
use math::{quat_between, quat_from_axis_angle};

/// Callback resolving an `Inline` / texture URL to bytes.
pub type UrlResolver = Arc<dyn Fn(&str) -> Option<Vec<u8>> + Send + Sync>;

/// Options for [`document_to_scene`].
#[derive(Clone)]
pub struct ConvertOptions {
    /// Circumference segments for the analytic primitives.
    pub segments: u32,
    /// Hostile-input bounds.
    pub limits: Limits,
    /// Resolver used to splice `Inline` scenes (none by default).
    pub inline_resolver: Option<UrlResolver>,
    /// Remaining `Inline` nesting depth.
    pub inline_depth: u32,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            segments: 32,
            limits: Limits::default(),
            inline_resolver: None,
            inline_depth: 8,
        }
    }
}

impl std::fmt::Debug for ConvertOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvertOptions")
            .field("segments", &self.segments)
            .field("limits", &self.limits)
            .field("inline_resolver", &self.inline_resolver.is_some())
            .field("inline_depth", &self.inline_depth)
            .finish()
    }
}

/// Convert a document to a [`Scene3D`].
pub fn document_to_scene(doc: &X3dDocument, opts: &ConvertOptions) -> Result<Scene3D> {
    let mut doc = doc.clone();
    let mut routes = proto::expand_all(&mut doc, &opts.limits)?;
    routes.extend(doc.scene.routes.iter().cloned());
    let angle = doc.unit_factor("angle") as f32;
    let mut c = Conv {
        doc: &doc,
        opts,
        scene: Scene3D::new(),
        geom: GeomOptions {
            segments: opts.segments.max(3),
            angle_factor: angle,
            vertex_budget: opts.limits.max_vertices,
        },
        angle,
        on_stack: HashSet::new(),
        instances: HashMap::new(),
        shape_meshes: HashMap::new(),
        prim_sources: HashMap::new(),
        materials: MaterialBuilder::default(),
        skipped: BTreeSet::new(),
        warnings: doc.warnings.clone(),
        bindables: Vec::new(),
        humanoids: Vec::new(),
        animated: animation::animated_targets(&doc, &routes),
        pivots: HashMap::new(),
        pending: None,
        out_nodes: 0,
        depth: 0,
    };
    let roots = doc.scene.roots.clone();
    for r in roots {
        let ids = c.visit(r)?;
        for id in ids {
            c.scene.add_root(id);
        }
    }
    hanim::bind_skins(&mut c);
    animation::build_animations(&mut c, &routes);
    c.finish_units();
    c.finish_extras();
    Ok(c.scene)
}

pub(crate) struct Conv<'a> {
    pub(crate) doc: &'a X3dDocument,
    pub(crate) opts: &'a ConvertOptions,
    pub(crate) scene: Scene3D,
    pub(crate) geom: GeomOptions,
    pub(crate) angle: f32,
    on_stack: HashSet<NodeIdx>,
    /// mesh3d nodes created for each X3D node.
    pub(crate) instances: HashMap<NodeIdx, Vec<NodeId>>,
    shape_meshes: HashMap<NodeIdx, Option<MeshId>>,
    /// (mesh, primitive) → (coordinate node, per-vertex coord index).
    pub(crate) prim_sources: HashMap<(MeshId, usize), (NodeIdx, Vec<u32>)>,
    materials: MaterialBuilder,
    skipped: BTreeSet<String>,
    pub(crate) warnings: Vec<String>,
    bindables: Vec<Value>,
    pub(crate) humanoids: Vec<(NodeIdx, NodeId)>,
    /// X3D nodes targeted by interpolator routes.
    animated: HashSet<NodeIdx>,
    /// Outer node → (inner pivot node, centre) for split transforms.
    pub(crate) pivots: HashMap<NodeId, (NodeId, [f32; 3])>,
    pending: Option<Box<Plan>>,
    out_nodes: usize,
    depth: usize,
}

/// Deferred children of a grouping node (see `Conv::prepare`).
struct Plan {
    node: Box<Node>,
    pivot: Option<(Box<Node>, [f32; 3])>,
    kids: Vec<NodeIdx>,
    humanoid: bool,
}

fn kids_of(n: &X3dNode, fields: &[&str]) -> Vec<NodeIdx> {
    fields
        .iter()
        .flat_map(|f| n.children_of(f).iter().copied())
        .collect()
}

fn tuple3(n: &X3dNode, name: &str, d: [f32; 3]) -> [f32; 3] {
    n.value(name).and_then(|v| v.as_tuple::<3>()).unwrap_or(d)
}

fn tuple4(n: &X3dNode, name: &str, d: [f32; 4]) -> [f32; 4] {
    n.value(name).and_then(|v| v.as_tuple::<4>()).unwrap_or(d)
}

fn f32v(n: &X3dNode, name: &str, d: f32) -> f32 {
    n.value(name).and_then(|v| v.as_f32()).unwrap_or(d)
}

fn boolv(n: &X3dNode, name: &str, d: bool) -> bool {
    n.value(name).and_then(|v| v.as_bool()).unwrap_or(d)
}

/// JSON rendering of a field value (node references become type names).
pub(crate) fn value_json(doc: &X3dDocument, v: &crate::field::FieldValue) -> Value {
    let mf = v.ty.is_mf();
    let arr = |vals: Vec<Value>| -> Value {
        if !mf && vals.len() == 1 {
            vals.into_iter().next().unwrap_or(Value::Null)
        } else {
            Value::Array(vals)
        }
    };
    match &v.data {
        FieldData::Bool(b) => arr(b.iter().map(|&x| json!(x)).collect()),
        FieldData::Int32(b) => arr(b.iter().map(|&x| json!(x)).collect()),
        FieldData::Float(b) => arr(b.iter().map(|&x| json!(x)).collect()),
        FieldData::Double(b) => arr(b.iter().map(|&x| json!(x)).collect()),
        FieldData::String(b) => arr(b.iter().map(|x| json!(x)).collect()),
        FieldData::Image(b) => arr(b
            .iter()
            .map(|i| json!({"width": i.width, "height": i.height, "components": i.components}))
            .collect()),
        FieldData::Node(b) => arr(b
            .iter()
            .filter_map(|&i| doc.node(i))
            .map(|n| json!(n.type_name))
            .collect()),
    }
}

/// Explicit non-node fields of a node as a JSON object.
pub(crate) fn fields_json(doc: &X3dDocument, n: &X3dNode) -> Value {
    let mut m = Map::new();
    m.insert("type".into(), json!(n.type_name));
    if let Some(d) = &n.def {
        m.insert("DEF".into(), json!(d));
    }
    for (k, v) in &n.fields {
        if matches!(v.data, FieldData::Node(_)) {
            continue;
        }
        m.insert(k.clone(), value_json(doc, v));
    }
    Value::Object(m)
}

/// Metadata node (MetadataSet / Metadata*) → JSON, depth-bounded.
pub(crate) fn metadata_json(doc: &X3dDocument, idx: NodeIdx, depth: usize) -> Value {
    let Some(n) = doc.node(idx) else {
        return Value::Null;
    };
    let mut m = Map::new();
    m.insert("type".into(), json!(n.type_name));
    if let Some(name) = n.value("name").and_then(|v| v.as_str().map(str::to_string)) {
        m.insert("name".into(), json!(name));
    }
    if let Some(r) = n
        .value("reference")
        .and_then(|v| v.as_str().map(str::to_string))
        .filter(|r| !r.is_empty())
    {
        m.insert("reference".into(), json!(r));
    }
    if n.type_name == "MetadataSet" {
        let vals: Vec<Value> = if depth < 32 {
            n.children_of("value")
                .iter()
                .map(|&c| metadata_json(doc, c, depth + 1))
                .collect()
        } else {
            Vec::new()
        };
        m.insert("value".into(), Value::Array(vals));
    } else if let Some(v) = n.get("value") {
        let mut j = value_json(doc, v);
        if !j.is_array() {
            j = Value::Array(vec![j]);
        }
        m.insert("value".into(), j);
    }
    Value::Object(m)
}

impl Conv<'_> {
    fn new_node(&mut self, idx: NodeIdx, n: &X3dNode) -> Result<Node> {
        self.out_nodes += 1;
        if self.out_nodes > self.opts.limits.max_nodes {
            return Err(Error::limit("scene expands to more than max_nodes nodes"));
        }
        let mut node = Node::new();
        node.name = n.def.clone();
        if let Some(&m) = n.children_of("metadata").first() {
            node.extras
                .insert("x3d:metadata".into(), metadata_json(self.doc, m, 0));
        }
        if !matches!(n.type_name.as_str(), "Transform" | "Group" | "Shape") {
            node.extras.insert("x3d:type".into(), json!(n.type_name));
        }
        let _ = idx;
        Ok(node)
    }

    fn push(&mut self, idx: NodeIdx, node: Node) -> NodeId {
        let id = self.scene.add_node(node);
        self.instances.entry(idx).or_default().push(id);
        id
    }

    /// Transform of an X3D Transform-like node:
    /// `T × C × R × SR × S × −SR × −C`. Without a non-uniform
    /// `scaleOrientation` this is exactly the TRS
    /// `(T + C − R·(S∘C), R, S)`; otherwise a matrix.
    fn x3d_transform(&self, n: &X3dNode) -> Transform {
        let t = tuple3(n, "translation", [0.0; 3]);
        let mut r = tuple4(n, "rotation", [0.0, 0.0, 1.0, 0.0]);
        r[3] *= self.angle;
        let s = tuple3(n, "scale", [1.0; 3]);
        let mut so = tuple4(n, "scaleOrientation", [0.0, 0.0, 1.0, 0.0]);
        so[3] *= self.angle;
        let c = tuple3(n, "center", [0.0; 3]);
        let uniform = s[0] == s[1] && s[1] == s[2];
        if so[3] == 0.0 || uniform {
            let q = quat_from_axis_angle(r);
            let rsc = math::quat_rotate(q, [s[0] * c[0], s[1] * c[1], s[2] * c[2]]);
            Transform::Trs {
                translation: [
                    t[0] + c[0] - rsc[0],
                    t[1] + c[1] - rsc[1],
                    t[2] + c[2] - rsc[2],
                ],
                rotation: q,
                scale: s,
            }
        } else {
            Transform::Matrix(math::x3d_transform_matrix(t, r, s, so, c))
        }
    }

    /// Build a Transform-like node. Animated nodes with a non-zero
    /// `center` are split into an outer TRS node `(T + C, R, S)` (the
    /// animation target) and an inner pivot node translating by `−C`
    /// that holds the children, so translation / rotation / scale
    /// channels stay exact.
    fn transform_plan(
        &mut self,
        idx: NodeIdx,
        n: &X3dNode,
        mut node: Node,
        child_fields: &[&str],
        humanoid: bool,
    ) {
        let c = tuple3(n, "center", [0.0; 3]);
        let mut so = tuple4(n, "scaleOrientation", [0.0, 0.0, 1.0, 0.0]);
        so[3] *= self.angle;
        let s = tuple3(n, "scale", [1.0; 3]);
        let simple_so = so[3] == 0.0 || (s[0] == s[1] && s[1] == s[2]);
        let kids = kids_of(n, child_fields);
        let pivot = if self.animated.contains(&idx) && c != [0.0; 3] && simple_so {
            let t = tuple3(n, "translation", [0.0; 3]);
            let mut r = tuple4(n, "rotation", [0.0, 0.0, 1.0, 0.0]);
            r[3] *= self.angle;
            node.transform = Transform::Trs {
                translation: [t[0] + c[0], t[1] + c[1], t[2] + c[2]],
                rotation: quat_from_axis_angle(r),
                scale: s,
            };
            let mut pivot = Node::new();
            pivot.name = n.def.as_ref().map(|d| format!("{d}:pivot"));
            pivot.transform = Transform::Trs {
                translation: [-c[0], -c[1], -c[2]],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0; 3],
            };
            pivot.extras.insert("x3d:pivot".into(), json!(c));
            Some((Box::new(pivot), c))
        } else {
            node.transform = self.x3d_transform(n);
            None
        };
        if c != [0.0; 3] {
            node.extras.insert("x3d:center".into(), json!(c));
        }
        self.pending = Some(Box::new(Plan {
            node: Box::new(node),
            pivot,
            kids,
            humanoid,
        }));
    }

    fn plan(&mut self, node: Node, kids: Vec<NodeIdx>) {
        self.pending = Some(Box::new(Plan {
            node: Box::new(node),
            pivot: None,
            kids,
            humanoid: false,
        }));
    }

    /// Convert one X3D node; returns the mesh3d nodes to attach to the
    /// parent (empty for non-rendering nodes).
    pub(crate) fn visit(&mut self, idx: NodeIdx) -> Result<Vec<NodeId>> {
        let doc = self.doc;
        let Some(n) = doc.node(idx) else {
            return Ok(Vec::new());
        };
        if self.on_stack.contains(&idx) {
            self.warnings
                .push(format!("cyclic USE of {} ignored", n.type_name));
            return Ok(Vec::new());
        }
        if self.depth > self.opts.limits.max_depth {
            return Err(Error::limit("scene graph deeper than max_depth"));
        }
        if let NodeKind::ProtoInstance { .. } = n.kind {
            self.skipped
                .insert(format!("ProtoInstance:{}", n.type_name));
            return Ok(Vec::new());
        }
        self.on_stack.insert(idx);
        self.depth += 1;
        let r = self.visit_inner(idx, n);
        self.depth -= 1;
        self.on_stack.remove(&idx);
        r
    }

    fn visit_inner(&mut self, idx: NodeIdx, n: &X3dNode) -> Result<Vec<NodeId>> {
        let out = self.prepare(idx, n)?;
        match self.pending.take() {
            None => Ok(out),
            Some(plan) => self.finish_plan(idx, *plan),
        }
    }

    /// Second half of a grouping node: convert its children (the only
    /// recursive step — kept in a small stack frame) and attach them.
    fn finish_plan(&mut self, idx: NodeIdx, plan: Plan) -> Result<Vec<NodeId>> {
        let mut children = Vec::new();
        for k in plan.kids {
            children.extend(self.visit(k)?);
        }
        let mut node = *plan.node;
        let id = match plan.pivot {
            Some((mut pivot, c)) => {
                pivot.children = children;
                let pid = self.scene.add_node(*pivot);
                node.children = vec![pid];
                let id = self.push(idx, node);
                self.pivots.insert(id, (pid, c));
                id
            }
            None => {
                node.children = children;
                self.push(idx, node)
            }
        };
        if plan.humanoid {
            self.humanoids.push((idx, id));
        }
        Ok(vec![id])
    }

    /// Non-recursive part of a node conversion. Grouping nodes leave a
    /// [`Plan`] in `self.pending` instead of recursing here, so the
    /// (large) frame of this function is never on the recursion path.
    #[inline(never)]
    fn prepare(&mut self, idx: NodeIdx, n: &X3dNode) -> Result<Vec<NodeId>> {
        let t = n.type_name.as_str();
        match t {
            "Transform" | "HAnimJoint" | "HAnimSite" | "CADPart" | "EspduTransform" => {
                let mut node = self.new_node(idx, n)?;
                if t == "HAnimJoint" || t == "HAnimSite" {
                    if let Some(name) = n.value("name").and_then(|v| v.as_str().map(str::to_string))
                    {
                        if !name.is_empty() {
                            node.extras.insert("x3d:hanimName".into(), json!(name));
                        }
                    }
                }
                self.transform_plan(idx, n, node, &["children"], false);
                Ok(Vec::new())
            }
            "HAnimHumanoid" => {
                let mut node = self.new_node(idx, n)?;
                if let Some(name) = n.value("name").and_then(|v| v.as_str().map(str::to_string)) {
                    node.extras.insert("x3d:hanimName".into(), json!(name));
                }
                if let Some(v) = n
                    .value("version")
                    .and_then(|v| v.as_str().map(str::to_string))
                {
                    node.extras.insert("x3d:hanimVersion".into(), json!(v));
                }
                self.transform_plan(idx, n, node, &["skeleton", "skin", "viewpoints"], true);
                Ok(Vec::new())
            }
            "Group"
            | "StaticGroup"
            | "Collision"
            | "Anchor"
            | "Billboard"
            | "CADAssembly"
            | "CADLayer"
            | "CADFace"
            | "LayoutGroup"
            | "ScreenGroup"
            | "PickableGroup"
            | "HAnimSegment"
            | "LayerSet"
            | "Layer"
            | "GeoLocation"
            | "GeoTransform"
            | "Viewport"
            | "DISEntityTypeMapping"
            | "TransformSensor" => {
                let mut node = self.new_node(idx, n)?;
                match t {
                    "Anchor" => {
                        node.extras
                            .insert("x3d:anchor".into(), fields_json(self.doc, n));
                    }
                    "Billboard" => {
                        node.extras.insert(
                            "x3d:billboard".into(),
                            json!({"axisOfRotation": tuple3(n, "axisOfRotation", [0.0, 1.0, 0.0])}),
                        );
                    }
                    "Collision" => {
                        node.extras.insert(
                            "x3d:collision".into(),
                            json!({"enabled": boolv(n, "enabled", true)}),
                        );
                    }
                    "GeoLocation" | "GeoTransform" => {
                        node.extras
                            .insert("x3d:geo".into(), fields_json(self.doc, n));
                    }
                    _ => {}
                }
                let fields: &[&str] = match t {
                    "CADFace" => &["shape"],
                    "LayerSet" => &["layers"],
                    _ => &["children"],
                };
                self.plan(node, kids_of(n, fields));
                Ok(Vec::new())
            }
            "Switch" => {
                let mut node = self.new_node(idx, n)?;
                let which = n
                    .value("whichChoice")
                    .and_then(|v| v.as_i32())
                    .unwrap_or(-1);
                let kids: Vec<NodeIdx> = if !n.children_of("children").is_empty() {
                    n.children_of("children").to_vec()
                } else {
                    n.children_of("choice").to_vec()
                };
                node.extras.insert(
                    "x3d:switch".into(),
                    json!({"whichChoice": which, "choices": kids.len()}),
                );
                let chosen = if which >= 0 {
                    kids.get(which as usize).copied().into_iter().collect()
                } else {
                    Vec::new()
                };
                self.plan(node, chosen);
                Ok(Vec::new())
            }
            "LOD" => {
                let mut node = self.new_node(idx, n)?;
                let kids: Vec<NodeIdx> = if !n.children_of("children").is_empty() {
                    n.children_of("children").to_vec()
                } else {
                    n.children_of("level").to_vec()
                };
                let range = n.value("range").map(|v| v.as_f32s()).unwrap_or_default();
                node.extras.insert(
                    "x3d:lod".into(),
                    json!({"center": tuple3(n, "center", [0.0; 3]), "range": range, "levels": kids.len()}),
                );
                self.plan(node, kids.first().copied().into_iter().collect());
                Ok(Vec::new())
            }
            "Inline" => {
                let mut node = self.new_node(idx, n)?;
                let urls = n
                    .value("url")
                    .map(|v| v.as_strings().to_vec())
                    .unwrap_or_default();
                node.extras.insert(
                    "x3d:inline".into(),
                    json!({"url": urls, "load": boolv(n, "load", true)}),
                );
                let id = self.push(idx, node);
                self.splice_inline(id, &urls);
                Ok(vec![id])
            }
            "Shape" => {
                let mut node = self.new_node(idx, n)?;
                node.mesh = self.shape_mesh(idx, n);
                Ok(vec![self.push(idx, node)])
            }
            "Viewpoint" | "OrthoViewpoint" | "GeoViewpoint" => {
                let mut node = self.new_node(idx, n)?;
                let pos = tuple3(n, "position", [0.0, 0.0, 10.0]);
                let mut ori = tuple4(n, "orientation", [0.0, 0.0, 1.0, 0.0]);
                ori[3] *= self.angle;
                node.transform = Transform::Trs {
                    translation: pos,
                    rotation: quat_from_axis_angle(ori),
                    scale: [1.0; 3],
                };
                let near = f32v(n, "nearDistance", -1.0);
                let far = f32v(n, "farDistance", -1.0);
                let cam = if t == "OrthoViewpoint" {
                    let fov = n
                        .value("fieldOfView")
                        .map(|v| v.as_f32s())
                        .filter(|f| f.len() >= 4)
                        .unwrap_or(vec![-1.0, -1.0, 1.0, 1.0]);
                    Camera::Orthographic {
                        xmag: ((fov[2] - fov[0]) / 2.0).abs().max(1e-6),
                        ymag: ((fov[3] - fov[1]) / 2.0).abs().max(1e-6),
                        znear: if near > 0.0 { near } else { 0.1 },
                        zfar: if far > 0.0 { far } else { 1000.0 },
                    }
                } else {
                    let fov = f32v(n, "fieldOfView", std::f32::consts::FRAC_PI_4) * self.angle;
                    Camera::Perspective {
                        aspect_ratio: None,
                        yfov: fov.clamp(1e-4, std::f32::consts::PI - 1e-4),
                        znear: if near > 0.0 { near } else { 0.1 },
                        zfar: (far > 0.0).then_some(far),
                    }
                };
                node.camera = Some(self.scene.add_camera(cam));
                let mut ex = Map::new();
                if let Some(d) = n
                    .value("description")
                    .and_then(|v| v.as_str().map(str::to_string))
                {
                    if !d.is_empty() {
                        ex.insert("description".into(), json!(d));
                    }
                }
                ex.insert(
                    "centerOfRotation".into(),
                    json!(tuple3(n, "centerOfRotation", [0.0; 3])),
                );
                node.extras
                    .insert("x3d:viewpoint".into(), Value::Object(ex));
                Ok(vec![self.push(idx, node)])
            }
            "DirectionalLight" | "PointLight" | "SpotLight" => {
                let mut node = self.new_node(idx, n)?;
                let color = tuple3(n, "color", [1.0; 3]);
                let intensity = f32v(n, "intensity", 1.0);
                let dir = tuple3(n, "direction", [0.0, 0.0, -1.0]);
                let loc = tuple3(n, "location", [0.0; 3]);
                let radius = f32v(n, "radius", 100.0);
                let light = match t {
                    "DirectionalLight" => Light::Directional { color, intensity },
                    "PointLight" => Light::Point {
                        color,
                        intensity,
                        range: (radius > 0.0).then_some(radius),
                    },
                    _ => {
                        let cut = (f32v(n, "cutOffAngle", std::f32::consts::FRAC_PI_2)
                            * self.angle)
                            .clamp(1e-4, std::f32::consts::FRAC_PI_2);
                        let beam = (f32v(n, "beamWidth", std::f32::consts::FRAC_PI_4) * self.angle)
                            .clamp(0.0, cut);
                        let inner = if beam >= cut { cut * 0.999 } else { beam };
                        Light::Spot {
                            color,
                            intensity,
                            range: (radius > 0.0).then_some(radius),
                            inner_cone_angle: inner,
                            outer_cone_angle: cut,
                        }
                    }
                };
                let rotation = if t == "PointLight" {
                    [0.0, 0.0, 0.0, 1.0]
                } else {
                    quat_between([0.0, 0.0, -1.0], dir)
                };
                node.transform = Transform::Trs {
                    translation: if t == "DirectionalLight" {
                        [0.0; 3]
                    } else {
                        loc
                    },
                    rotation,
                    scale: [1.0; 3],
                };
                node.light = Some(self.scene.add_light(light));
                let mut ex = Map::new();
                ex.insert("on".into(), json!(boolv(n, "on", true)));
                ex.insert(
                    "global".into(),
                    json!(boolv(n, "global", t != "DirectionalLight")),
                );
                ex.insert(
                    "ambientIntensity".into(),
                    json!(f32v(n, "ambientIntensity", 0.0)),
                );
                if t != "DirectionalLight" {
                    ex.insert(
                        "attenuation".into(),
                        json!(tuple3(n, "attenuation", [1.0, 0.0, 0.0])),
                    );
                }
                if t == "SpotLight" {
                    ex.insert(
                        "beamWidth".into(),
                        json!(f32v(n, "beamWidth", std::f32::consts::FRAC_PI_4)),
                    );
                }
                node.extras.insert("x3d:light".into(), Value::Object(ex));
                Ok(vec![self.push(idx, node)])
            }
            "WorldInfo" | "NavigationInfo" | "Background" | "TextureBackground" | "Fog"
            | "FogCoordinate" | "LocalFog" | "EnvironmentLight" => {
                self.bindables.push(fields_json(self.doc, n));
                Ok(Vec::new())
            }
            "MetadataSet" | "MetadataString" | "MetadataInteger" | "MetadataFloat"
            | "MetadataDouble" | "MetadataBoolean" => {
                self.bindables.push(metadata_json(self.doc, idx, 0));
                Ok(Vec::new())
            }
            "TimeSensor"
            | "PositionInterpolator"
            | "OrientationInterpolator"
            | "ScalarInterpolator"
            | "ColorInterpolator"
            | "CoordinateInterpolator"
            | "NormalInterpolator"
            | "PositionInterpolator2D"
            | "CoordinateInterpolator2D"
            | "SplinePositionInterpolator"
            | "SplineScalarInterpolator"
            | "SquadOrientationInterpolator"
            | "EaseInEaseOut"
            | "ROUTE" => Ok(Vec::new()),
            other => {
                self.skipped.insert(other.to_string());
                Ok(Vec::new())
            }
        }
    }

    fn shape_mesh(&mut self, idx: NodeIdx, n: &X3dNode) -> Option<MeshId> {
        if let Some(m) = self.shape_meshes.get(&idx) {
            return *m;
        }
        let doc = self.doc;
        let geom_idx = n.children_of("geometry").first().copied();
        let app_idx = n.children_of("appearance").first().copied();
        let geom = geom_idx.and_then(|g| doc.node(g));
        let app = app_idx.and_then(|a| doc.node(a));
        let tex = appearance::tex_transform(doc, app, self.angle);
        let mut notes = BuildNotes::default();
        let built = geom.and_then(|g| {
            geometry::build_geometry(doc, g, tex.as_ref(), &mut self.geom, &mut notes)
        });
        self.warnings.extend(notes.warnings);
        let Some(mut built) = built else {
            if let Some(g) = geom {
                self.skipped.insert(g.type_name.clone());
            }
            self.shape_meshes.insert(idx, None);
            return None;
        };
        let g = geom.expect("built implies geometry");
        let key = MatKey {
            appearance: app_idx,
            solid: built.solid,
            has_colors: built.has_colors,
            unlit_geometry: appearance::is_unlit_geometry(&g.type_name)
                && built.prim.normals.is_none(),
        };
        let mat = self
            .materials
            .material(doc, &mut self.scene, key, &mut self.warnings);
        built.prim.material = Some(mat);
        built
            .prim
            .extras
            .insert("x3d:geometry".into(), json!(g.type_name));
        if let Some(mode) = appearance::texgen_mode(doc, g) {
            built
                .prim
                .extras
                .insert("x3d:textureCoordinateGenerator".into(), json!(mode));
        }
        if let Some(d) = &g.def {
            built.prim.extras.insert("x3d:geometryDEF".into(), json!(d));
        }
        let coord_idx = g.children_of("coord").first().copied();
        let source = built.source_index.take();
        let mut mesh = Mesh::new(n.def.clone().or_else(|| g.def.clone()));
        mesh.primitives.push(built.prim);
        let id = self.scene.add_mesh(mesh);
        if let (Some(c), Some(s)) = (coord_idx, source) {
            self.prim_sources.insert((id, 0), (c, s));
        }
        self.shape_meshes.insert(idx, Some(id));
        Some(id)
    }

    fn splice_inline(&mut self, parent: NodeId, urls: &[String]) {
        let Some(resolver) = self.opts.inline_resolver.clone() else {
            return;
        };
        if self.opts.inline_depth == 0 {
            self.warnings.push("Inline nesting too deep".into());
            return;
        }
        for u in urls {
            let Some(bytes) = resolver(u) else { continue };
            let sub_opts = ConvertOptions {
                inline_depth: self.opts.inline_depth - 1,
                ..self.opts.clone()
            };
            let sub = crate::parse_document_with_limits(&bytes, &self.opts.limits)
                .and_then(|d| document_to_scene(&d, &sub_opts));
            match sub {
                Ok(sub) => {
                    let off = self.scene.append(&sub);
                    let new_roots: Vec<NodeId> = self
                        .scene
                        .roots
                        .iter()
                        .copied()
                        .filter(|r| r.0 >= off.nodes)
                        .collect();
                    self.scene.roots.retain(|r| r.0 < off.nodes);
                    if let Some(p) = self.scene.node_mut(parent) {
                        p.children.extend(new_roots);
                    }
                    return;
                }
                Err(e) => self.warnings.push(format!("Inline '{u}': {e}")),
            }
        }
    }

    fn finish_units(&mut self) {
        let f = self.doc.unit_factor("length") as f32;
        if (f - 1.0).abs() < 1e-9 {
            return;
        }
        let known = [
            (Unit::Millimetres, 0.001f32),
            (Unit::Centimetres, 0.01),
            (Unit::Inches, 0.0254),
            (Unit::Feet, 0.3048),
            (Unit::Yards, 0.9144),
        ];
        if let Some((u, _)) = known.iter().find(|(_, k)| ((f - k) / k).abs() < 1e-5) {
            self.scene.unit = *u;
            return;
        }
        // Arbitrary factor: wrap the roots in a scaling node.
        let mut root = Node::new().with_name("x3d:units");
        root.transform = Transform::Trs {
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [f, f, f],
        };
        root.children = std::mem::take(&mut self.scene.roots);
        root.extras
            .insert("x3d:lengthConversionFactor".into(), json!(f));
        let id = self.scene.add_node(root);
        self.scene.roots = vec![id];
    }

    fn finish_extras(&mut self) {
        let d = self.doc;
        let mut x = Map::new();
        x.insert("version".into(), json!(d.version));
        x.insert("profile".into(), json!(d.profile));
        if !d.components.is_empty() {
            x.insert(
                "components".into(),
                Value::Array(
                    d.components
                        .iter()
                        .map(|c| json!({"name": c.name, "level": c.level}))
                        .collect(),
                ),
            );
        }
        if !d.units.is_empty() {
            x.insert(
                "units".into(),
                Value::Array(
                    d.units
                        .iter()
                        .map(|u| json!({"category": u.category, "name": u.name, "conversionFactor": u.conversion_factor}))
                        .collect(),
                ),
            );
        }
        if !d.meta.is_empty() {
            x.insert(
                "meta".into(),
                Value::Array(
                    d.meta
                        .iter()
                        .map(|(n, c)| json!({"name": n, "content": c}))
                        .collect(),
                ),
            );
        }
        self.scene.extras.insert("x3d".into(), Value::Object(x));
        if !self.bindables.is_empty() {
            self.scene.extras.insert(
                "x3d:environment".into(),
                Value::Array(std::mem::take(&mut self.bindables)),
            );
        }
        if !self.skipped.is_empty() {
            self.scene.extras.insert(
                "x3d:unconverted".into(),
                json!(self.skipped.iter().collect::<Vec<_>>()),
            );
        }
        if !self.warnings.is_empty() {
            self.warnings.truncate(crate::document::MAX_WARNINGS);
            self.scene
                .extras
                .insert("x3d:warnings".into(), json!(self.warnings));
        }
    }
}
