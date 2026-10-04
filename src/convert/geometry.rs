//! Geometry nodes → [`Primitive`].
//!
//! Polygonal geometry (IndexedFaceSet, the triangle / quad set family,
//! ElevationGrid, Extrusion, the Geometry2D surfaces) is first gathered
//! into a *polygon soup* — faces of corners, each corner carrying a
//! coordinate index plus optional normal / colour / texture
//! coordinates — and then triangulated (fan for convex faces, ear
//! clipping otherwise), given generated normals where the file has
//! none (honouring `creaseAngle`), and welded into an indexed
//! triangle list. Lines and points take a simpler path; the analytic
//! primitives (Box, Sphere, Cone, Cylinder) are built directly with
//! the texture layouts of ISO/IEC 19775-1 13.3.
//!
//! Texture coordinates are produced in X3D (s, t) space (origin lower
//! left), the appearance's `TextureTransform` is baked in, and the
//! result is flipped to the glTF convention `v = 1 − t`.

use std::collections::HashMap;
use std::f32::consts::PI;

use oxideav_mesh3d::{Indices, Primitive, Topology};

use super::math::{add, cross, dot, len, normalize, quat_from_axis_angle, quat_rotate, scale, sub};
use crate::document::{X3dDocument, X3dNode};
use crate::field::FieldValue;

/// Options steering geometry generation.
#[derive(Clone, Copy, Debug)]
pub struct GeomOptions {
    /// Segments around the circumference of Sphere / Cone / Cylinder /
    /// Disk2D / Circle2D / Arc2D.
    pub segments: u32,
    /// Multiplier applied to angle-valued fields (`unit angle`).
    pub angle_factor: f32,
    /// Remaining vertex budget.
    pub vertex_budget: usize,
}

/// 2D texture-coordinate transform in X3D (s, t) space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TexTransform2 {
    /// `center`.
    pub center: [f32; 2],
    /// `rotation` (radians).
    pub rotation: f32,
    /// `scale`.
    pub scale: [f32; 2],
    /// `translation`.
    pub translation: [f32; 2],
}

impl TexTransform2 {
    /// Apply to an (s, t) coordinate: scale and rotate about `center`,
    /// then translate (ISO/IEC 19775-1 18.4.8,
    /// `Tc' = −C × S × R × C × T × Tc` read in application order).
    pub fn apply(&self, p: [f32; 2]) -> [f32; 2] {
        let x = p[0] - self.center[0];
        let y = p[1] - self.center[1];
        let x = x * self.scale[0];
        let y = y * self.scale[1];
        let (s, c) = self.rotation.sin_cos();
        let (x, y) = (x * c - y * s, x * s + y * c);
        [
            x + self.center[0] + self.translation[0],
            y + self.center[1] + self.translation[1],
        ]
    }
}

/// Result of converting one geometry node.
#[derive(Debug)]
pub struct Built {
    /// The primitive (material unset).
    pub prim: Primitive,
    /// For coordinate-based geometry: the coordinate index each output
    /// vertex came from (used for H-Anim skin binding).
    pub source_index: Option<Vec<u32>>,
    /// `solid` field value.
    pub solid: bool,
    /// Geometry carries per-vertex colours.
    pub has_colors: bool,
}

/// Information gathered while building, reported back for extras.
#[derive(Debug, Default)]
pub struct BuildNotes {
    /// Non-fatal problems.
    pub warnings: Vec<String>,
}

fn child<'a>(doc: &'a X3dDocument, node: &X3dNode, field: &str) -> Option<&'a X3dNode> {
    node.children_of(field).first().and_then(|&i| doc.node(i))
}

fn f32_field(node: &X3dNode, name: &str, default: f32) -> f32 {
    node.value(name).and_then(|v| v.as_f32()).unwrap_or(default)
}

fn bool_field(node: &X3dNode, name: &str, default: bool) -> bool {
    node.value(name)
        .and_then(|v| v.as_bool())
        .unwrap_or(default)
}

fn i32s(node: &X3dNode, name: &str) -> Vec<i32> {
    node.value(name)
        .map(|v| v.as_i32s().to_vec())
        .unwrap_or_default()
}

fn tuples<const N: usize>(node: &X3dNode, name: &str) -> Vec<[f32; N]> {
    match node.value(name) {
        Some(v) => tuples_of(&v, node, name),
        None => Vec::new(),
    }
}

/// Tuples of a value; unknown nodes keep raw strings, so re-parse
/// them leniently.
fn tuples_of<const N: usize>(v: &FieldValue, _node: &X3dNode, _name: &str) -> Vec<[f32; N]> {
    if let Some(s) = v.as_str() {
        let f: Vec<f32> = s
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter_map(|t| t.parse::<f32>().ok())
            .collect();
        return f
            .chunks_exact(N)
            .map(|c| {
                let mut a = [0.0; N];
                a.copy_from_slice(c);
                a
            })
            .collect();
    }
    v.as_tuples::<N>()
}

/// Coordinates of a `coord` field (Coordinate / CoordinateDouble).
fn coords(doc: &X3dDocument, node: &X3dNode) -> Vec<[f32; 3]> {
    child(doc, node, "coord")
        .map(|c| tuples::<3>(c, "point"))
        .unwrap_or_default()
}

/// Colours of a `color` field as RGBA.
fn colors(doc: &X3dDocument, node: &X3dNode) -> Option<Vec<[f32; 4]>> {
    let c = child(doc, node, "color")?;
    match c.type_name.as_str() {
        "ColorRGBA" => Some(tuples::<4>(c, "color")),
        _ => Some(
            tuples::<3>(c, "color")
                .into_iter()
                .map(|[r, g, b]| [r, g, b, 1.0])
                .collect(),
        ),
    }
}

fn normals(doc: &X3dDocument, node: &X3dNode) -> Option<Vec<[f32; 3]>> {
    child(doc, node, "normal").map(|n| tuples::<3>(n, "vector"))
}

/// Texture-coordinate sets of a `texCoord` field (MultiTextureCoordinate
/// flattened). Generators yield `None` entries.
fn texcoord_sets(doc: &X3dDocument, node: &X3dNode) -> Vec<Option<Vec<[f32; 2]>>> {
    let Some(tc) = child(doc, node, "texCoord") else {
        return Vec::new();
    };
    let one = |n: &X3dNode| -> Option<Vec<[f32; 2]>> {
        match n.type_name.as_str() {
            "TextureCoordinate" | "TextureCoordinateDouble" => Some(tuples::<2>(n, "point")),
            "TextureCoordinate3D" => Some(
                tuples::<3>(n, "point")
                    .into_iter()
                    .map(|p| [p[0], p[1]])
                    .collect(),
            ),
            "TextureCoordinate4D" => Some(
                tuples::<4>(n, "point")
                    .into_iter()
                    .map(|p| {
                        let w = if p[3] != 0.0 { p[3] } else { 1.0 };
                        [p[0] / w, p[1] / w]
                    })
                    .collect(),
            ),
            _ => None,
        }
    };
    if tc.type_name == "MultiTextureCoordinate" {
        tc.children_of("texCoord")
            .iter()
            .filter_map(|&i| doc.node(i))
            .map(one)
            .collect()
    } else {
        vec![one(tc)]
    }
}

/// Polygon soup.
#[derive(Default)]
struct Soup {
    positions: Vec<[f32; 3]>,
    face_start: Vec<usize>,
    corner_pos: Vec<u32>,
    corner_normal: Option<Vec<[f32; 3]>>,
    corner_color: Option<Vec<[f32; 4]>>,
    corner_uv: Vec<Vec<[f32; 2]>>,
}

impl Soup {
    fn faces(&self) -> impl Iterator<Item = std::ops::Range<usize>> + '_ {
        (0..self.face_start.len()).map(move |f| {
            let s = self.face_start[f];
            let e = self
                .face_start
                .get(f + 1)
                .copied()
                .unwrap_or(self.corner_pos.len());
            s..e
        })
    }
}

/// Newell normal of a polygon.
fn newell(pts: &[[f32; 3]]) -> [f32; 3] {
    let mut n = [0.0f32; 3];
    for i in 0..pts.len() {
        let a = pts[i];
        let b = pts[(i + 1) % pts.len()];
        n[0] += (a[1] - b[1]) * (a[2] + b[2]);
        n[1] += (a[2] - b[2]) * (a[0] + b[0]);
        n[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    n
}

/// Triangulate one polygon given its corner positions; returns local
/// corner triples. Fan when `convex`, ear clipping otherwise.
fn triangulate(pts: &[[f32; 3]], convex: bool) -> Vec<[usize; 3]> {
    let n = pts.len();
    if n < 3 {
        return Vec::new();
    }
    if n == 3 {
        return vec![[0, 1, 2]];
    }
    if convex || n > 4096 {
        return (1..n - 1).map(|i| [0, i, i + 1]).collect();
    }
    // Project onto the dominant plane of the Newell normal.
    let nn = newell(pts);
    let (ax, ay) = if nn[0].abs() >= nn[1].abs() && nn[0].abs() >= nn[2].abs() {
        (1, 2)
    } else if nn[1].abs() >= nn[2].abs() {
        (2, 0)
    } else {
        (0, 1)
    };
    let sign = match (ax, ay) {
        (1, 2) => nn[0].signum(),
        (2, 0) => nn[1].signum(),
        _ => nn[2].signum(),
    };
    let p2: Vec<[f32; 2]> = pts.iter().map(|p| [p[ax], p[ay]]).collect();
    let area2 = |a: [f32; 2], b: [f32; 2], c: [f32; 2]| -> f32 {
        ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) * sign
    };
    let inside = |p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]| -> bool {
        area2(a, b, p) >= 0.0 && area2(b, c, p) >= 0.0 && area2(c, a, p) >= 0.0
    };
    let mut idx: Vec<usize> = (0..n).collect();
    let mut out = Vec::with_capacity(n - 2);
    let mut guard = 0usize;
    while idx.len() > 3 && guard < n * n {
        guard += 1;
        let m = idx.len();
        let mut clipped = false;
        for i in 0..m {
            let (a, b, c) = (idx[(i + m - 1) % m], idx[i], idx[(i + 1) % m]);
            if area2(p2[a], p2[b], p2[c]) <= 0.0 {
                continue;
            }
            let ear = idx
                .iter()
                .all(|&o| o == a || o == b || o == c || !inside(p2[o], p2[a], p2[b], p2[c]));
            if ear {
                out.push([a, b, c]);
                idx.remove(i);
                clipped = true;
                break;
            }
        }
        if !clipped {
            break;
        }
    }
    if idx.len() >= 3 {
        // Degenerate remainder: fan it.
        for i in 1..idx.len() - 1 {
            out.push([idx[0], idx[i], idx[i + 1]]);
        }
    }
    out
}

/// Supplied normal: kept verbatim when already unit length (so
/// round trips are bit-stable), normalised otherwise.
fn unit(n: [f32; 3]) -> [f32; 3] {
    let l = len(n);
    if (l - 1.0).abs() < 1e-4 {
        n
    } else {
        normalize(n).unwrap_or([0.0, 1.0, 0.0])
    }
}

fn bits3(v: [f32; 3]) -> [u32; 3] {
    [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()]
}

/// Triangulate, generate normals, weld.
fn finish_soup(
    soup: Soup,
    ccw: bool,
    convex: bool,
    crease: Option<f32>,
    tex: Option<&TexTransform2>,
    opts: &mut GeomOptions,
) -> Option<(Primitive, Vec<u32>, bool)> {
    let corner_count = soup.corner_pos.len();
    // Triangles over corner indices.
    let mut tris: Vec<[usize; 3]> = Vec::new();
    let mut tri_face: Vec<usize> = Vec::new();
    let mut face_normals: Vec<[f32; 3]> = Vec::new();
    for (fi, range) in soup.faces().enumerate() {
        let pts: Vec<[f32; 3]> = soup.corner_pos[range.clone()]
            .iter()
            .map(|&p| soup.positions[p as usize])
            .collect();
        let mut fnorm = normalize(newell(&pts)).unwrap_or([0.0, 0.0, 0.0]);
        if !ccw {
            fnorm = scale(fnorm, -1.0);
        }
        face_normals.push(fnorm);
        for t in triangulate(&pts, convex) {
            let mut tri = [range.start + t[0], range.start + t[1], range.start + t[2]];
            if !ccw {
                tri.swap(1, 2);
            }
            tris.push(tri);
            tri_face.push(fi);
        }
    }
    if tris.is_empty() {
        return None;
    }
    // Corner → face.
    let mut corner_face = vec![0usize; corner_count];
    for (fi, range) in soup.faces().enumerate() {
        for c in range {
            corner_face[c] = fi;
        }
    }
    let corner_normal: Vec<[f32; 3]> = match (&soup.corner_normal, crease) {
        (Some(n), _) => n.clone(),
        (None, crease) => {
            let crease = crease.unwrap_or(0.0);
            if crease <= 0.0 {
                corner_face.iter().map(|&f| face_normals[f]).collect()
            } else {
                // Faces incident to each position.
                let mut incident: HashMap<u32, Vec<usize>> = HashMap::new();
                for (c, &p) in soup.corner_pos.iter().enumerate() {
                    incident.entry(p).or_default().push(corner_face[c]);
                }
                let cos_crease = if crease >= PI { -2.0 } else { crease.cos() };
                (0..corner_count)
                    .map(|c| {
                        let f = corner_face[c];
                        let fnn = face_normals[f];
                        let inc = &incident[&soup.corner_pos[c]];
                        let mut acc = [0.0f32; 3];
                        if inc.len() > 512 && cos_crease > -1.5 {
                            // Pathological fan-in: flat to stay linear.
                            return fnn;
                        }
                        for &g in inc {
                            let gn = face_normals[g];
                            if g == f || dot(fnn, gn) > cos_crease {
                                acc = add(acc, gn);
                            }
                        }
                        normalize(acc).unwrap_or(fnn)
                    })
                    .collect()
            }
        }
    };
    // Texture coordinates: bake transform + flip.
    let uv_sets: Vec<Vec<[f32; 2]>> = soup
        .corner_uv
        .iter()
        .map(|set| {
            set.iter()
                .map(|&st| {
                    let st = tex.map(|t| t.apply(st)).unwrap_or(st);
                    [st[0], 1.0 - st[1]]
                })
                .collect()
        })
        .collect();
    // Weld.
    let mut map: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut corner_vertex = vec![u32::MAX; corner_count];
    let mut positions = Vec::new();
    let mut normals_out = Vec::new();
    let mut colors_out = Vec::new();
    let mut uvs_out: Vec<Vec<[f32; 2]>> = vec![Vec::new(); uv_sets.len()];
    let mut source = Vec::new();
    let mut used = vec![false; corner_count];
    for t in &tris {
        for &c in t {
            used[c] = true;
        }
    }
    for c in 0..corner_count {
        if !used[c] {
            continue;
        }
        let mut key: Vec<u32> = Vec::with_capacity(12);
        key.push(soup.corner_pos[c]);
        key.extend_from_slice(&bits3(corner_normal[c]));
        if let Some(col) = &soup.corner_color {
            let v = col[c];
            key.extend_from_slice(&[
                v[0].to_bits(),
                v[1].to_bits(),
                v[2].to_bits(),
                v[3].to_bits(),
            ]);
        }
        for set in &uv_sets {
            key.push(set[c][0].to_bits());
            key.push(set[c][1].to_bits());
        }
        let next = positions.len() as u32;
        let v = *map.entry(key).or_insert(next);
        if v == next {
            if positions.len() >= opts.vertex_budget {
                return None;
            }
            positions.push(soup.positions[soup.corner_pos[c] as usize]);
            normals_out.push(corner_normal[c]);
            if let Some(col) = &soup.corner_color {
                colors_out.push(col[c]);
            }
            for (k, set) in uv_sets.iter().enumerate() {
                uvs_out[k].push(set[c]);
            }
            source.push(soup.corner_pos[c]);
        }
        corner_vertex[c] = v;
    }
    opts.vertex_budget -= positions.len();
    let indices: Vec<u32> = tris
        .iter()
        .flat_map(|t| t.iter().map(|&c| corner_vertex[c]))
        .collect();
    let has_colors = soup.corner_color.is_some();
    let mut prim = Primitive::new(Topology::Triangles);
    prim.positions = positions;
    prim.normals = Some(normals_out);
    if has_colors {
        prim.colors = vec![colors_out];
    }
    prim.uvs = uvs_out;
    prim.indices = Some(Indices::U32(indices));
    Some((prim, source, has_colors))
}

/// Default IndexedFaceSet texture coordinates from the bounding box
/// (ISO/IEC 19775-1 13.3.6).
fn bbox_uv(pts: &[[f32; 3]]) -> impl Fn([f32; 3]) -> [f32; 2] {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for p in pts {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let size: Vec<f32> = (0..3)
        .map(|k| if hi[k] > lo[k] { hi[k] - lo[k] } else { 0.0 })
        .collect();
    let mut order = [0usize, 1, 2];
    // Stable sort by descending size keeps X, Y, Z tie preference.
    order.sort_by(|&a, &b| {
        size[b]
            .partial_cmp(&size[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (s_ax, t_ax) = (order[0], order[1]);
    let big = if size[s_ax] > 0.0 { size[s_ax] } else { 1.0 };
    let lo_s = if lo[s_ax].is_finite() { lo[s_ax] } else { 0.0 };
    let lo_t = if lo[t_ax].is_finite() { lo[t_ax] } else { 0.0 };
    move |p| [(p[s_ax] - lo_s) / big, (p[t_ax] - lo_t) / big]
}

/// Convert a geometry node.
pub fn build_geometry(
    doc: &X3dDocument,
    node: &X3dNode,
    tex: Option<&TexTransform2>,
    opts: &mut GeomOptions,
    notes: &mut BuildNotes,
) -> Option<Built> {
    let solid = bool_field(node, "solid", true);
    let t = node.type_name.as_str();
    let res: Option<(Primitive, Option<Vec<u32>>, bool)> = match t {
        "IndexedFaceSet" => {
            indexed_face_set(doc, node, tex, opts, notes).map(|(p, s, c)| (p, Some(s), c))
        }
        "IndexedTriangleSet"
        | "TriangleSet"
        | "IndexedTriangleStripSet"
        | "TriangleStripSet"
        | "IndexedTriangleFanSet"
        | "TriangleFanSet"
        | "IndexedQuadSet"
        | "QuadSet" => triangle_family(doc, node, tex, opts).map(|(p, s, c)| (p, Some(s), c)),
        "ElevationGrid" => elevation_grid(doc, node, tex, opts).map(|(p, _, c)| (p, None, c)),
        "Extrusion" => extrusion(node, tex, opts).map(|(p, _, c)| (p, None, c)),
        "IndexedLineSet" | "LineSet" => line_set(doc, node, opts).map(|(p, s, c)| (p, Some(s), c)),
        "PointSet" => point_set(doc, node, opts).map(|(p, s, c)| (p, Some(s), c)),
        "Box" => Some((box_prim(node, tex), None, false)),
        "Sphere" => Some((sphere_prim(node, tex, opts.segments), None, false)),
        "Cylinder" => Some((cylinder_prim(node, tex, opts.segments, false), None, false)),
        "Cone" => Some((cylinder_prim(node, tex, opts.segments, true), None, false)),
        "Rectangle2D" | "Disk2D" | "TriangleSet2D" | "ArcClose2D" => {
            surface_2d(node, tex, opts).map(|p| (p, None, false))
        }
        "Circle2D" | "Arc2D" | "Polyline2D" | "Polypoint2D" => {
            lines_2d(node, opts).map(|p| (p, None, false))
        }
        _ => None,
    };
    let (prim, source_index, has_colors) = res?;
    if prim.positions.is_empty() {
        return None;
    }
    Some(Built {
        prim,
        source_index,
        solid,
        has_colors,
    })
}

/// Split an index list at `-1` markers.
fn split_polys(idx: &[i32]) -> Vec<&[i32]> {
    idx.split(|&i| i < 0).filter(|p| !p.is_empty()).collect()
}

fn indexed_face_set(
    doc: &X3dDocument,
    node: &X3dNode,
    tex: Option<&TexTransform2>,
    opts: &mut GeomOptions,
    notes: &mut BuildNotes,
) -> Option<(Primitive, Vec<u32>, bool)> {
    let pts = coords(doc, node);
    let coord_index = i32s(node, "coordIndex");
    if pts.is_empty() || coord_index.is_empty() {
        return None;
    }
    let ccw = bool_field(node, "ccw", true);
    let convex = bool_field(node, "convex", true);
    let crease = f32_field(node, "creaseAngle", 0.0) * opts.angle_factor;
    let cols = colors(doc, node);
    let color_per_vertex = bool_field(node, "colorPerVertex", true);
    let color_index = i32s(node, "colorIndex");
    let nors = normals(doc, node);
    let normal_per_vertex = bool_field(node, "normalPerVertex", true);
    let normal_index = i32s(node, "normalIndex");
    let tsets = texcoord_sets(doc, node);
    let tex_index = i32s(node, "texCoordIndex");

    let mut soup = Soup {
        positions: pts.clone(),
        ..Soup::default()
    };
    let mut cnorm = nors.as_ref().map(|_| Vec::new());
    let mut ccol = cols.as_ref().map(|_| Vec::new());
    let usable_sets: Vec<Vec<[f32; 2]>> = tsets.into_iter().flatten().collect();
    let default_uv = usable_sets.is_empty();
    let n_sets = if default_uv { 1 } else { usable_sets.len() };
    soup.corner_uv = vec![Vec::new(); n_sets];
    let bb = bbox_uv(&pts);

    let mut face_no = 0usize;
    let mut pos = 0usize; // position in coordIndex
    let mut skipped = 0usize;
    while pos < coord_index.len() {
        let start = pos;
        while pos < coord_index.len() && coord_index[pos] >= 0 {
            pos += 1;
        }
        let end = pos;
        pos += 1; // skip -1
        if end - start < 3 {
            if end > start {
                skipped += 1;
                face_no += 1;
            }
            continue;
        }
        let poly = &coord_index[start..end];
        if poly.iter().any(|&i| i as usize >= pts.len()) {
            skipped += 1;
            face_no += 1;
            continue;
        }
        soup.face_start.push(soup.corner_pos.len());
        for (k, &ci) in poly.iter().enumerate() {
            let flat = start + k;
            soup.corner_pos.push(ci as u32);
            if let (Some(out), Some(src)) = (cnorm.as_mut(), nors.as_ref()) {
                let i = if normal_per_vertex {
                    if normal_index.is_empty() {
                        ci
                    } else {
                        normal_index.get(flat).copied().unwrap_or(ci)
                    }
                } else if normal_index.is_empty() {
                    face_no as i32
                } else {
                    normal_index.get(face_no).copied().unwrap_or(face_no as i32)
                };
                let n = src
                    .get(i.max(0) as usize)
                    .copied()
                    .unwrap_or([0.0, 1.0, 0.0]);
                out.push(unit(n));
            }
            if let (Some(out), Some(src)) = (ccol.as_mut(), cols.as_ref()) {
                let i = if color_per_vertex {
                    if color_index.is_empty() {
                        ci
                    } else {
                        color_index.get(flat).copied().unwrap_or(ci)
                    }
                } else if color_index.is_empty() {
                    face_no as i32
                } else {
                    color_index.get(face_no).copied().unwrap_or(face_no as i32)
                };
                out.push(src.get(i.max(0) as usize).copied().unwrap_or([1.0; 4]));
            }
            if default_uv {
                soup.corner_uv[0].push(bb(pts[ci as usize]));
            } else {
                for (s, set) in usable_sets.iter().enumerate() {
                    let i = if tex_index.is_empty() {
                        ci
                    } else {
                        tex_index.get(flat).copied().unwrap_or(ci)
                    };
                    soup.corner_uv[s].push(set.get(i.max(0) as usize).copied().unwrap_or([0.0; 2]));
                }
            }
        }
        face_no += 1;
    }
    if skipped > 0 {
        notes
            .warnings
            .push(format!("IndexedFaceSet: {skipped} invalid face(s) skipped"));
    }
    soup.corner_normal = cnorm;
    soup.corner_color = ccol;
    finish_soup(soup, ccw, convex, Some(crease), tex, opts)
}

fn triangle_family(
    doc: &X3dDocument,
    node: &X3dNode,
    tex: Option<&TexTransform2>,
    opts: &mut GeomOptions,
) -> Option<(Primitive, Vec<u32>, bool)> {
    let pts = coords(doc, node);
    if pts.is_empty() {
        return None;
    }
    let t = node.type_name.as_str();
    let ccw = bool_field(node, "ccw", true);
    let normal_per_vertex = bool_field(node, "normalPerVertex", true);
    let cols = colors(doc, node);
    let nors = normals(doc, node);
    let usable_sets: Vec<Vec<[f32; 2]>> = texcoord_sets(doc, node).into_iter().flatten().collect();

    // Faces as lists of coordinate indices.
    let mut faces: Vec<Vec<u32>> = Vec::new();
    let n = pts.len() as u32;
    let index: Vec<u32> = if t.starts_with("Indexed") {
        Vec::new()
    } else {
        (0..n).collect()
    };
    let idx_field = i32s(node, "index");
    let valid = |i: i32| i >= 0 && (i as u32) < n;
    let strips: Vec<Vec<u32>> = match t {
        "IndexedTriangleStripSet" | "IndexedTriangleFanSet" => split_polys(&idx_field)
            .into_iter()
            .map(|p| p.iter().filter(|&&i| valid(i)).map(|&i| i as u32).collect())
            .collect(),
        "TriangleStripSet" | "TriangleFanSet" => {
            let counts = i32s(
                node,
                if t == "TriangleStripSet" {
                    "stripCount"
                } else {
                    "fanCount"
                },
            );
            let mut out = Vec::new();
            let mut at = 0usize;
            for c in counts {
                let c = c.max(0) as usize;
                let end = (at + c).min(index.len());
                out.push(index[at..end].to_vec());
                at = end;
            }
            out
        }
        _ => Vec::new(),
    };
    match t {
        "IndexedTriangleSet" | "IndexedQuadSet" => {
            let k = if t == "IndexedQuadSet" { 4 } else { 3 };
            for c in idx_field.chunks_exact(k) {
                if c.iter().all(|&i| valid(i)) {
                    faces.push(c.iter().map(|&i| i as u32).collect());
                }
            }
        }
        "TriangleSet" | "QuadSet" => {
            let k = if t == "QuadSet" { 4 } else { 3 };
            for c in index.chunks_exact(k) {
                faces.push(c.to_vec());
            }
        }
        "IndexedTriangleStripSet" | "TriangleStripSet" => {
            for s in &strips {
                for k in 0..s.len().saturating_sub(2) {
                    if k % 2 == 0 {
                        faces.push(vec![s[k], s[k + 1], s[k + 2]]);
                    } else {
                        faces.push(vec![s[k + 1], s[k], s[k + 2]]);
                    }
                }
            }
        }
        _ => {
            for s in &strips {
                for k in 1..s.len().saturating_sub(1) {
                    faces.push(vec![s[0], s[k], s[k + 1]]);
                }
            }
        }
    }
    if faces.is_empty() {
        return None;
    }
    let mut soup = Soup {
        positions: pts.clone(),
        ..Soup::default()
    };
    let mut cnorm = nors.as_ref().map(|_| Vec::new());
    let mut ccol = cols.as_ref().map(|_| Vec::new());
    let default_uv = usable_sets.is_empty();
    let bb = bbox_uv(&pts);
    soup.corner_uv = vec![Vec::new(); if default_uv { 1 } else { usable_sets.len() }];
    for (fi, f) in faces.iter().enumerate() {
        soup.face_start.push(soup.corner_pos.len());
        for &ci in f {
            soup.corner_pos.push(ci);
            if let (Some(out), Some(src)) = (cnorm.as_mut(), nors.as_ref()) {
                let i = if normal_per_vertex { ci as usize } else { fi };
                out.push(unit(src.get(i).copied().unwrap_or([0.0, 1.0, 0.0])));
            }
            if let (Some(out), Some(src)) = (ccol.as_mut(), cols.as_ref()) {
                out.push(src.get(ci as usize).copied().unwrap_or([1.0; 4]));
            }
            if default_uv {
                soup.corner_uv[0].push(bb(pts[ci as usize]));
            } else {
                for (s, set) in usable_sets.iter().enumerate() {
                    soup.corner_uv[s].push(set.get(ci as usize).copied().unwrap_or([0.0; 2]));
                }
            }
        }
    }
    soup.corner_normal = cnorm;
    soup.corner_color = ccol;
    let crease = if normal_per_vertex { PI } else { 0.0 };
    finish_soup(soup, ccw, true, Some(crease), tex, opts)
}

fn elevation_grid(
    doc: &X3dDocument,
    node: &X3dNode,
    tex: Option<&TexTransform2>,
    opts: &mut GeomOptions,
) -> Option<(Primitive, Vec<u32>, bool)> {
    let xd = node.value("xDimension")?.as_i32()?.max(0) as usize;
    let zd = node.value("zDimension")?.as_i32()?.max(0) as usize;
    if xd < 2 || zd < 2 || xd.saturating_mul(zd) > opts.vertex_budget {
        return None;
    }
    let xs = f32_field(node, "xSpacing", 1.0);
    let zs = f32_field(node, "zSpacing", 1.0);
    let heights = node
        .value("height")
        .map(|v| v.as_f32s())
        .unwrap_or_default();
    let mut positions = Vec::with_capacity(xd * zd);
    for j in 0..zd {
        for i in 0..xd {
            let h = heights.get(i + j * xd).copied().unwrap_or(0.0);
            positions.push([xs * i as f32, h, zs * j as f32]);
        }
    }
    let ccw = bool_field(node, "ccw", true);
    let crease = f32_field(node, "creaseAngle", 0.0) * opts.angle_factor;
    let cols = colors(doc, node);
    let cpv = bool_field(node, "colorPerVertex", true);
    let nors = normals(doc, node);
    let npv = bool_field(node, "normalPerVertex", true);
    let sets: Vec<Vec<[f32; 2]>> = texcoord_sets(doc, node).into_iter().flatten().collect();
    let mut soup = Soup {
        positions,
        ..Soup::default()
    };
    let mut cnorm = nors.as_ref().map(|_| Vec::new());
    let mut ccol = cols.as_ref().map(|_| Vec::new());
    soup.corner_uv = vec![Vec::new(); sets.len().max(1)];
    for j in 0..zd - 1 {
        for i in 0..xd - 1 {
            let quad = j * (xd - 1) + i;
            let tri_a = [(i, j), (i, j + 1), (i + 1, j + 1)];
            let tri_b = [(i, j), (i + 1, j + 1), (i + 1, j)];
            for tri in [tri_a, tri_b] {
                soup.face_start.push(soup.corner_pos.len());
                for (ci, cj) in tri {
                    let v = ci + cj * xd;
                    soup.corner_pos.push(v as u32);
                    if let (Some(out), Some(src)) = (cnorm.as_mut(), nors.as_ref()) {
                        let k = if npv { v } else { quad };
                        out.push(unit(src.get(k).copied().unwrap_or([0.0, 1.0, 0.0])));
                    }
                    if let (Some(out), Some(src)) = (ccol.as_mut(), cols.as_ref()) {
                        let k = if cpv { v } else { quad };
                        out.push(src.get(k).copied().unwrap_or([1.0; 4]));
                    }
                    if sets.is_empty() {
                        soup.corner_uv[0]
                            .push([ci as f32 / (xd - 1) as f32, cj as f32 / (zd - 1) as f32]);
                    } else {
                        for (s, set) in sets.iter().enumerate() {
                            soup.corner_uv[s].push(set.get(v).copied().unwrap_or([0.0; 2]));
                        }
                    }
                }
            }
        }
    }
    soup.corner_normal = cnorm;
    soup.corner_color = ccol;
    finish_soup(soup, ccw, true, Some(crease), tex, opts)
}

/// Spine-aligned cross-section planes (ISO/IEC 19775-1 13.3.5.3).
fn extrusion_frames(spine: &[[f32; 3]]) -> Vec<[[f32; 3]; 3]> {
    let n = spine.len();
    let eq = |a: [f32; 3], b: [f32; 3]| len(sub(a, b)) < 1e-7;
    let closed = n > 2 && eq(spine[0], spine[n - 1]);
    // Next / previous distinct points.
    let next = |i: usize| -> Option<usize> { (i + 1..n).find(|&k| !eq(spine[k], spine[i])) };
    let prev = |i: usize| -> Option<usize> { (0..i).rev().find(|&k| !eq(spine[k], spine[i])) };
    let mut ys: Vec<Option<[f32; 3]>> = vec![None; n];
    let mut zs: Vec<Option<[f32; 3]>> = vec![None; n];
    for i in 0..n {
        let (p, q) = if closed && (i == 0 || i == n - 1) {
            (prev(n - 1).map(|k| spine[k]), next(0).map(|k| spine[k]))
        } else {
            (prev(i).map(|k| spine[k]), next(i).map(|k| spine[k]))
        };
        ys[i] = match (p, q) {
            (Some(p), Some(q)) => normalize(sub(q, p)),
            (None, Some(q)) => normalize(sub(q, spine[i])),
            (Some(p), None) => normalize(sub(spine[i], p)),
            _ => None,
        };
        let here = if closed && i == n - 1 {
            spine[0]
        } else {
            spine[i]
        };
        if let (Some(p), Some(q)) = (p, q) {
            zs[i] = normalize(cross(sub(q, here), sub(p, here)));
        }
    }
    // Open spine: first / last take the neighbours' Z.
    if !closed && n >= 2 {
        if zs[0].is_none() {
            zs[0] = zs.iter().flatten().next().copied();
        }
        if zs[n - 1].is_none() {
            zs[n - 1] = zs.iter().rev().flatten().next().copied();
        }
    }
    let all_collinear = zs.iter().all(|z| z.is_none());
    let mut frames = Vec::with_capacity(n);
    if all_collinear {
        // Rotate +Y onto the spine direction.
        let dir = (1..n)
            .find(|&k| !eq(spine[k], spine[0]))
            .and_then(|k| normalize(sub(spine[k], spine[0])))
            .unwrap_or([0.0, 1.0, 0.0]);
        let q = if dot(dir, [0.0, -1.0, 0.0]) > 1.0 - 1e-6 {
            [0.0, 0.0, 1.0, 0.0]
        } else {
            super::math::quat_between([0.0, 1.0, 0.0], dir)
        };
        let x = quat_rotate(q, [1.0, 0.0, 0.0]);
        let y = quat_rotate(q, [0.0, 1.0, 0.0]);
        let z = quat_rotate(q, [0.0, 0.0, 1.0]);
        return vec![[x, y, z]; n];
    }
    let mut last_z: Option<[f32; 3]> = None;
    for i in 0..n {
        let mut z = zs[i].or(last_z).unwrap_or([0.0, 0.0, 1.0]);
        if let Some(lz) = last_z {
            if dot(z, lz) < 0.0 {
                z = scale(z, -1.0);
            }
        }
        last_z = Some(z);
        let y = ys[i].unwrap_or([0.0, 1.0, 0.0]);
        let x = normalize(cross(y, z)).unwrap_or([1.0, 0.0, 0.0]);
        // Re-orthogonalise Z.
        let z = normalize(cross(x, y)).unwrap_or(z);
        frames.push([x, y, z]);
    }
    frames
}

fn extrusion(
    node: &X3dNode,
    tex: Option<&TexTransform2>,
    opts: &mut GeomOptions,
) -> Option<(Primitive, Vec<u32>, bool)> {
    let spine = tuples::<3>(node, "spine");
    let cs = tuples::<2>(node, "crossSection");
    let scales = tuples::<2>(node, "scale");
    let orients = tuples::<4>(node, "orientation");
    let ns = spine.len();
    let nc = cs.len();
    if ns < 2 || nc < 2 || ns.saturating_mul(nc) > opts.vertex_budget {
        return None;
    }
    let begin_cap = bool_field(node, "beginCap", true);
    let end_cap = bool_field(node, "endCap", true);
    let ccw = bool_field(node, "ccw", true);
    let convex = bool_field(node, "convex", true);
    let crease = f32_field(node, "creaseAngle", 0.0) * opts.angle_factor;
    let frames = extrusion_frames(&spine);
    let cs_closed = nc > 2 && cs[0] == cs[nc - 1];
    let mut positions = Vec::with_capacity(ns * nc);
    for i in 0..ns {
        let sc = scales
            .get(i)
            .or(scales.first())
            .copied()
            .unwrap_or([1.0, 1.0]);
        let o = orients
            .get(i)
            .or(orients.first())
            .copied()
            .unwrap_or([0.0, 0.0, 1.0, 0.0]);
        let q = quat_from_axis_angle([o[0], o[1], o[2], o[3] * opts.angle_factor]);
        let [fx, fy, fz] = frames[i];
        for c in &cs {
            let local = quat_rotate(q, [c[0] * sc[0], 0.0, c[1] * sc[1]]);
            let world = add(
                spine[i],
                add(
                    scale(fx, local[0]),
                    add(scale(fy, local[1]), scale(fz, local[2])),
                ),
            );
            positions.push(world);
        }
    }
    // Cumulative parameters for default UVs.
    let cum = |pts: &[f32]| -> Vec<f32> {
        let total: f32 = pts.iter().sum();
        let mut acc = 0.0;
        let mut out = vec![0.0];
        for d in pts {
            acc += d;
            out.push(if total > 0.0 { acc / total } else { 0.0 });
        }
        out
    };
    let cs_d: Vec<f32> = cs
        .windows(2)
        .map(|w| ((w[1][0] - w[0][0]).powi(2) + (w[1][1] - w[0][1]).powi(2)).sqrt())
        .collect();
    let sp_d: Vec<f32> = spine.windows(2).map(|w| len(sub(w[1], w[0]))).collect();
    let u = cum(&cs_d);
    let v = cum(&sp_d);
    let pos_of = |i: usize, j: usize| -> u32 {
        let j = if cs_closed && j == nc - 1 { 0 } else { j };
        (i * nc + j) as u32
    };
    let mut soup = Soup {
        positions,
        ..Soup::default()
    };
    soup.corner_uv = vec![Vec::new()];
    for i in 0..ns - 1 {
        for j in 0..nc - 1 {
            soup.face_start.push(soup.corner_pos.len());
            for (ii, jj) in [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)] {
                soup.corner_pos.push(pos_of(ii, jj));
                soup.corner_uv[0].push([u[jj], v[ii]]);
            }
        }
    }
    // Caps.
    let (mut lo, mut hi) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]);
    for c in &cs {
        for k in 0..2 {
            lo[k] = lo[k].min(c[k]);
            hi[k] = hi[k].max(c[k]);
        }
    }
    let big = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1e-12);
    let cap_n = if cs_closed { nc - 1 } else { nc };
    if cap_n >= 3 {
        if begin_cap {
            soup.face_start.push(soup.corner_pos.len());
            for j in (0..cap_n).rev() {
                soup.corner_pos.push(pos_of(0, j));
                soup.corner_uv[0].push([(cs[j][0] - lo[0]) / big, (cs[j][1] - lo[1]) / big]);
            }
        }
        if end_cap {
            soup.face_start.push(soup.corner_pos.len());
            for (j, c) in cs.iter().enumerate().take(cap_n) {
                soup.corner_pos.push(pos_of(ns - 1, j));
                soup.corner_uv[0].push([(c[0] - lo[0]) / big, (c[1] - lo[1]) / big]);
            }
        }
    }
    finish_soup(soup, ccw, convex, Some(crease), tex, opts)
}

fn line_set(
    doc: &X3dDocument,
    node: &X3dNode,
    opts: &mut GeomOptions,
) -> Option<(Primitive, Vec<u32>, bool)> {
    let pts = coords(doc, node);
    if pts.is_empty() || pts.len() > opts.vertex_budget {
        return None;
    }
    let cols = colors(doc, node);
    let n = pts.len();
    // Polylines as (coord index, colour) sequences.
    let mut lines: Vec<Vec<(u32, Option<[f32; 4]>)>> = Vec::new();
    if node.type_name == "LineSet" {
        let mut at = 0usize;
        for c in i32s(node, "vertexCount") {
            let c = c.max(0) as usize;
            let end = (at + c).min(n);
            lines.push(
                (at..end)
                    .map(|i| (i as u32, cols.as_ref().and_then(|cc| cc.get(i).copied())))
                    .collect(),
            );
            at = end;
        }
    } else {
        let ci = i32s(node, "coordIndex");
        let cpv = bool_field(node, "colorPerVertex", true);
        let col_idx = i32s(node, "colorIndex");
        let mut flat = 0usize;
        let mut poly_no = 0usize;
        let mut cur = Vec::new();
        for &i in &ci {
            if i < 0 {
                if !cur.is_empty() {
                    lines.push(std::mem::take(&mut cur));
                }
                poly_no += 1;
                flat += 1;
                continue;
            }
            if (i as usize) < n {
                let color = cols.as_ref().and_then(|cc| {
                    let k = if cpv {
                        if col_idx.is_empty() {
                            i
                        } else {
                            col_idx.get(flat).copied().unwrap_or(i)
                        }
                    } else if col_idx.is_empty() {
                        poly_no as i32
                    } else {
                        col_idx.get(poly_no).copied().unwrap_or(poly_no as i32)
                    };
                    cc.get(k.max(0) as usize).copied()
                });
                cur.push((i as u32, color));
            }
            flat += 1;
        }
        if !cur.is_empty() {
            lines.push(cur);
        }
    }
    let has_colors = cols.is_some();
    let mut map: HashMap<(u32, [u32; 4]), u32> = HashMap::new();
    let mut positions = Vec::new();
    let mut colors_out = Vec::new();
    let mut source = Vec::new();
    let mut indices = Vec::new();
    for l in &lines {
        let mut ids = Vec::with_capacity(l.len());
        for &(ci, col) in l {
            let col = col.unwrap_or([1.0; 4]);
            let key = (ci, col.map(f32::to_bits));
            let next = positions.len() as u32;
            let v = *map.entry(key).or_insert(next);
            if v == next {
                positions.push(pts[ci as usize]);
                colors_out.push(col);
                source.push(ci);
            }
            ids.push(v);
        }
        for w in ids.windows(2) {
            indices.extend_from_slice(&[w[0], w[1]]);
        }
    }
    if indices.is_empty() {
        return None;
    }
    opts.vertex_budget = opts.vertex_budget.saturating_sub(positions.len());
    let mut prim = Primitive::new(Topology::Lines);
    prim.positions = positions;
    if has_colors {
        prim.colors = vec![colors_out];
    }
    prim.indices = Some(Indices::U32(indices));
    Some((prim, source, has_colors))
}

fn point_set(
    doc: &X3dDocument,
    node: &X3dNode,
    opts: &mut GeomOptions,
) -> Option<(Primitive, Vec<u32>, bool)> {
    let pts = coords(doc, node);
    if pts.is_empty() || pts.len() > opts.vertex_budget {
        return None;
    }
    opts.vertex_budget -= pts.len();
    let cols = colors(doc, node);
    let n = pts.len();
    let mut prim = Primitive::new(Topology::Points);
    if let Some(c) = &cols {
        let mut c = c.clone();
        c.resize(n, [1.0; 4]);
        prim.colors = vec![c];
    }
    if let Some(ns) = normals(doc, node) {
        if ns.len() >= n {
            prim.normals = Some(ns[..n].to_vec());
        }
    }
    prim.positions = pts;
    Some((prim, (0..n as u32).collect(), cols.is_some()))
}

/// Builder for analytic shapes: vertices with normal + (s, t).
#[derive(Default)]
struct Direct {
    pos: Vec<[f32; 3]>,
    nrm: Vec<[f32; 3]>,
    st: Vec<[f32; 2]>,
    idx: Vec<u32>,
}

impl Direct {
    fn v(&mut self, p: [f32; 3], n: [f32; 3], st: [f32; 2]) -> u32 {
        self.pos.push(p);
        self.nrm.push(n);
        self.st.push(st);
        (self.pos.len() - 1) as u32
    }

    fn quad(&mut self, a: u32, b: u32, c: u32, d: u32) {
        self.idx.extend_from_slice(&[a, b, c, a, c, d]);
    }

    fn finish(self, tex: Option<&TexTransform2>) -> Primitive {
        let mut prim = Primitive::new(Topology::Triangles);
        prim.uvs = vec![self
            .st
            .iter()
            .map(|&st| {
                let st = tex.map(|t| t.apply(st)).unwrap_or(st);
                [st[0], 1.0 - st[1]]
            })
            .collect()];
        prim.positions = self.pos;
        prim.normals = Some(self.nrm);
        prim.indices = Some(Indices::U32(self.idx));
        prim
    }
}

fn box_prim(node: &X3dNode, tex: Option<&TexTransform2>) -> Primitive {
    let s = node
        .value("size")
        .and_then(|v| v.as_tuple::<3>())
        .unwrap_or([2.0, 2.0, 2.0]);
    let h = [s[0] / 2.0, s[1] / 2.0, s[2] / 2.0];
    let mut d = Direct::default();
    // (normal, right axis, up axis) per face, viewed from outside with
    // the orientations of ISO/IEC 19775-1 13.3.1.
    let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ];
    for (n, r, u) in faces {
        let mut corner = |sr: f32, su: f32| -> u32 {
            let p = [
                (n[0] + r[0] * sr + u[0] * su) * h[0],
                (n[1] + r[1] * sr + u[1] * su) * h[1],
                (n[2] + r[2] * sr + u[2] * su) * h[2],
            ];
            d.v(p, n, [(sr + 1.0) / 2.0, (su + 1.0) / 2.0])
        };
        let a = corner(-1.0, -1.0);
        let b = corner(1.0, -1.0);
        let c = corner(1.0, 1.0);
        let e = corner(-1.0, 1.0);
        d.quad(a, b, c, e);
    }
    d.finish(tex)
}

/// Point on the unit circle for texture parameter `u` (0 at −Z,
/// counter-clockwise seen from +Y).
fn ring(u: f32) -> (f32, f32) {
    let a = 2.0 * PI * u;
    (-a.sin(), -a.cos())
}

fn sphere_prim(node: &X3dNode, tex: Option<&TexTransform2>, seg: u32) -> Primitive {
    let r = f32_field(node, "radius", 1.0);
    let slices = seg.clamp(3, 1024);
    let stacks = (seg / 2).clamp(2, 512);
    let mut d = Direct::default();
    for j in 0..=stacks {
        let t = j as f32 / stacks as f32;
        let phi = PI * t; // 0 at bottom
        let (sp, cp) = phi.sin_cos();
        for i in 0..=slices {
            let u = i as f32 / slices as f32;
            let (x, z) = ring(u);
            let n = [x * sp, -cp, z * sp];
            d.v(scale(n, r), n, [u, t]);
        }
    }
    let w = slices + 1;
    for j in 0..stacks {
        for i in 0..slices {
            let a = j * w + i;
            let b = a + 1;
            let c = a + w + 1;
            let e = a + w;
            // Counter-clockwise seen from outside.
            d.quad(a, b, c, e);
        }
    }
    d.finish(tex)
}

fn cylinder_prim(node: &X3dNode, tex: Option<&TexTransform2>, seg: u32, cone: bool) -> Primitive {
    let h = f32_field(node, "height", 2.0);
    let r = if cone {
        f32_field(node, "bottomRadius", 1.0)
    } else {
        f32_field(node, "radius", 1.0)
    };
    let side = bool_field(node, "side", true);
    let bottom = bool_field(node, "bottom", true);
    let top = !cone && bool_field(node, "top", true);
    let n = seg.clamp(3, 4096);
    let y0 = -h / 2.0;
    let y1 = h / 2.0;
    let mut d = Direct::default();
    if side {
        // Cone slant normal: (x, r/h, z) normalised.
        let ny = if cone { r / h.max(1e-12) } else { 0.0 };
        for i in 0..=n {
            let u = i as f32 / n as f32;
            let (x, z) = ring(u);
            let nrm = normalize([x, ny, z]).unwrap_or([x, 0.0, z]);
            d.v([x * r, y0, z * r], nrm, [u, 0.0]);
            let top_p = if cone {
                [0.0, y1, 0.0]
            } else {
                [x * r, y1, z * r]
            };
            d.v(top_p, nrm, [u, 1.0]);
        }
        for i in 0..n {
            let a = 2 * i;
            let (b0, t0, b1, t1) = (a, a + 1, a + 2, a + 3);
            if cone {
                d.idx.extend_from_slice(&[b0, b1, t0]);
            } else {
                d.quad(b0, b1, t1, t0);
            }
        }
    }
    let cap = |d: &mut Direct, y: f32, up: bool| {
        let nrm = if up {
            [0.0, 1.0, 0.0]
        } else {
            [0.0, -1.0, 0.0]
        };
        let c = d.v([0.0, y, 0.0], nrm, [0.5, 0.5]);
        let first = d.pos.len() as u32;
        for i in 0..n {
            let (x, z) = ring(i as f32 / n as f32);
            let st = if up {
                [0.5 + x / 2.0, 0.5 - z / 2.0]
            } else {
                [0.5 + x / 2.0, 0.5 + z / 2.0]
            };
            d.v([x * r, y, z * r], nrm, st);
        }
        for i in 0..n {
            let a = first + i;
            let b = first + (i + 1) % n;
            if up {
                d.idx.extend_from_slice(&[c, a, b]);
            } else {
                d.idx.extend_from_slice(&[c, b, a]);
            }
        }
    };
    if bottom {
        cap(&mut d, y0, false);
    }
    if top {
        cap(&mut d, y1, true);
    }
    d.finish(tex)
}

fn surface_2d(
    node: &X3dNode,
    tex: Option<&TexTransform2>,
    opts: &mut GeomOptions,
) -> Option<Primitive> {
    let seg = opts.segments.clamp(3, 4096);
    let nrm = [0.0, 0.0, 1.0];
    let mut d = Direct::default();
    match node.type_name.as_str() {
        "Rectangle2D" => {
            let s = node
                .value("size")
                .and_then(|v| v.as_tuple::<2>())
                .unwrap_or([2.0, 2.0]);
            let (w, h) = (s[0] / 2.0, s[1] / 2.0);
            let a = d.v([-w, -h, 0.0], nrm, [0.0, 0.0]);
            let b = d.v([w, -h, 0.0], nrm, [1.0, 0.0]);
            let c = d.v([w, h, 0.0], nrm, [1.0, 1.0]);
            let e = d.v([-w, h, 0.0], nrm, [0.0, 1.0]);
            d.quad(a, b, c, e);
        }
        "Disk2D" => {
            let ro = f32_field(node, "outerRadius", 1.0);
            let ri = f32_field(node, "innerRadius", 0.0).clamp(0.0, ro);
            let big = ro.max(1e-12) * 2.0;
            let st = |x: f32, y: f32| [0.5 + x / big, 0.5 + y / big];
            for i in 0..=seg {
                let a = 2.0 * PI * i as f32 / seg as f32;
                let (s, c) = a.sin_cos();
                d.v([c * ri, s * ri, 0.0], nrm, st(c * ri, s * ri));
                d.v([c * ro, s * ro, 0.0], nrm, st(c * ro, s * ro));
            }
            for i in 0..seg {
                let a = 2 * i;
                d.quad(a, a + 1, a + 3, a + 2);
            }
        }
        "ArcClose2D" => {
            let r = f32_field(node, "radius", 1.0);
            let a0 = f32_field(node, "startAngle", 0.0) * opts.angle_factor;
            let a1 = f32_field(node, "endAngle", PI / 2.0) * opts.angle_factor;
            let pie = node
                .value("closureType")
                .and_then(|v| v.as_str().map(|s| s.eq_ignore_ascii_case("PIE")))
                .unwrap_or(true);
            let span = if a1 > a0 { a1 - a0 } else { a1 - a0 + 2.0 * PI };
            let big = r.max(1e-12) * 2.0;
            let st = |x: f32, y: f32| [0.5 + x / big, 0.5 + y / big];
            let first = if pie {
                Some(d.v([0.0, 0.0, 0.0], nrm, [0.5, 0.5]))
            } else {
                None
            };
            let start = d.pos.len() as u32;
            for i in 0..=seg {
                let a = a0 + span * i as f32 / seg as f32;
                let (s, c) = a.sin_cos();
                d.v([c * r, s * r, 0.0], nrm, st(c * r, s * r));
            }
            let hub = first.unwrap_or(start);
            for i in 0..seg {
                let a = start + i;
                if a != hub && a + 1 != hub {
                    d.idx.extend_from_slice(&[hub, a, a + 1]);
                }
            }
        }
        _ => {
            // TriangleSet2D
            let v = tuples::<2>(node, "vertices");
            let (mut lo, mut hi) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]);
            for p in &v {
                for k in 0..2 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
            let big = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1e-12);
            for t in v.chunks_exact(3) {
                for p in t {
                    d.v(
                        [p[0], p[1], 0.0],
                        nrm,
                        [(p[0] - lo[0]) / big, (p[1] - lo[1]) / big],
                    );
                }
                let k = d.pos.len() as u32;
                d.idx.extend_from_slice(&[k - 3, k - 2, k - 1]);
            }
        }
    }
    if d.idx.is_empty() || d.pos.len() > opts.vertex_budget {
        return None;
    }
    opts.vertex_budget -= d.pos.len();
    Some(d.finish(tex))
}

fn lines_2d(node: &X3dNode, opts: &mut GeomOptions) -> Option<Primitive> {
    let seg = opts.segments.clamp(3, 4096);
    let (pts, topo): (Vec<[f32; 3]>, Topology) = match node.type_name.as_str() {
        "Circle2D" => {
            let r = f32_field(node, "radius", 1.0);
            (
                (0..seg)
                    .map(|i| {
                        let (s, c) = (2.0 * PI * i as f32 / seg as f32).sin_cos();
                        [c * r, s * r, 0.0]
                    })
                    .collect(),
                Topology::LineLoop,
            )
        }
        "Arc2D" => {
            let r = f32_field(node, "radius", 1.0);
            let a0 = f32_field(node, "startAngle", 0.0) * opts.angle_factor;
            let a1 = f32_field(node, "endAngle", PI / 2.0) * opts.angle_factor;
            let span = if a1 > a0 { a1 - a0 } else { a1 - a0 + 2.0 * PI };
            (
                (0..=seg)
                    .map(|i| {
                        let (s, c) = (a0 + span * i as f32 / seg as f32).sin_cos();
                        [c * r, s * r, 0.0]
                    })
                    .collect(),
                Topology::LineStrip,
            )
        }
        "Polyline2D" => (
            tuples::<2>(node, "lineSegments")
                .into_iter()
                .map(|p| [p[0], p[1], 0.0])
                .collect(),
            Topology::LineStrip,
        ),
        _ => (
            tuples::<2>(node, "point")
                .into_iter()
                .map(|p| [p[0], p[1], 0.0])
                .collect(),
            Topology::Points,
        ),
    };
    if pts.is_empty() || pts.len() > opts.vertex_budget {
        return None;
    }
    opts.vertex_budget -= pts.len();
    let mut prim = Primitive::new(topo);
    prim.positions = pts;
    Some(prim)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ear_clipping_concave() {
        // L-shape (concave) in the XY plane, CCW.
        let pts = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ];
        let tris = triangulate(&pts, false);
        assert_eq!(tris.len(), 4);
        let area: f32 = tris
            .iter()
            .map(|t| {
                let n = cross(sub(pts[t[1]], pts[t[0]]), sub(pts[t[2]], pts[t[0]]));
                n[2] / 2.0
            })
            .sum();
        assert!((area - 3.0).abs() < 1e-5, "area {area}");
    }

    #[test]
    fn tex_transform() {
        let t = TexTransform2 {
            center: [0.0, 0.0],
            rotation: 0.0,
            scale: [2.0, 2.0],
            translation: [0.5, 0.0],
        };
        assert_eq!(t.apply([1.0, 1.0]), [2.5, 2.0]);
    }
}
