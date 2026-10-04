//! ROUTE graphs → mesh3d animations.
//!
//! The classic X3D keyframe pattern is
//! `TimeSensor.fraction_changed → Interpolator.set_fraction` followed
//! by `Interpolator.value_changed → target.set_field`. Each TimeSensor
//! becomes one [`Animation`]; keyframe times are `key × cycleInterval`
//! seconds. Supported targets:
//!
//! * Position(Spline)Interpolator → `translation` / `scale` of
//!   Transform-like nodes, `position` of viewpoints, `location` of
//!   point / spot lights;
//! * Orientation / SquadOrientation interpolators → `rotation` /
//!   `orientation`;
//! * CoordinateInterpolator → a Coordinate's `point`: the keyframes
//!   become morph targets of every mesh built from that Coordinate and
//!   the animation drives one-hot morph weights, which reproduces the
//!   linear per-vertex interpolation exactly.
//!
//! Other routes (sensors, scripts, colour / scalar interpolators) have
//! no mesh3d counterpart and are reported in `extras["x3d:routes"]`.

use std::collections::{BTreeMap, HashMap, HashSet};

use oxideav_mesh3d::{
    Animation, AnimationChannel, AnimationProperty, AnimationSampler, AnimationValues,
    Interpolation, MorphTarget, Transform,
};
use serde_json::{json, Value};

use super::math::quat_from_axis_angle;
use super::Conv;
use crate::document::{NodeIdx, Route};

fn base_field(f: &str) -> &str {
    let f = f.strip_prefix("set_").unwrap_or(f);
    f.strip_suffix("_changed").unwrap_or(f)
}

/// Strictly increasing keyframe times.
fn times(keys: &[f32], cycle: f32) -> Vec<f32> {
    let mut out: Vec<f32> = keys.iter().map(|k| k * cycle).collect();
    for i in 1..out.len() {
        if out[i].partial_cmp(&out[i - 1]) != Some(std::cmp::Ordering::Greater) {
            out[i] = out[i - 1] + (cycle.abs().max(1.0) * 1e-6);
        }
    }
    out
}

/// X3D nodes whose fields are driven by an interpolator.
pub(crate) fn animated_targets(doc: &crate::X3dDocument, routes: &[Route]) -> HashSet<NodeIdx> {
    routes
        .iter()
        .filter(|r| {
            doc.node(r.from_node)
                .map(|n| n.type_name.contains("Interpolator"))
                .unwrap_or(false)
        })
        .map(|r| r.to_node)
        .collect()
}

pub(crate) fn build_animations(c: &mut Conv<'_>, routes: &[Route]) {
    let doc = c.doc;
    let ty = |i: NodeIdx| doc.node(i).map(|n| n.type_name.as_str()).unwrap_or("");
    let mut ts_of: HashMap<NodeIdx, NodeIdx> = HashMap::new();
    for r in routes {
        if ty(r.from_node) == "TimeSensor"
            && base_field(&r.from_field) == "fraction"
            && base_field(&r.to_field) == "fraction"
        {
            ts_of.insert(r.to_node, r.from_node);
        }
    }
    let mut channels: BTreeMap<NodeIdx, Vec<AnimationChannel>> = BTreeMap::new();
    let mut unmapped: Vec<Value> = Vec::new();
    for r in routes {
        let Some(&ts) = ts_of.get(&r.from_node) else {
            if !(ty(r.from_node) == "TimeSensor" && ts_of.contains_key(&r.to_node)) {
                unmapped.push(json!(format!(
                    "{}.{} -> {}.{}",
                    r.from_def, r.from_field, r.to_def, r.to_field
                )));
            }
            continue;
        };
        if base_field(&r.from_field) != "value" {
            continue;
        }
        let Some(interp) = doc.node(r.from_node) else {
            continue;
        };
        let Some(ts_node) = doc.node(ts) else {
            continue;
        };
        let cycle = ts_node
            .value("cycleInterval")
            .and_then(|v| v.as_f32())
            .filter(|c| *c > 0.0 && c.is_finite())
            .unwrap_or(1.0);
        let keys = interp.value("key").map(|v| v.as_f32s()).unwrap_or_default();
        if keys.is_empty() {
            continue;
        }
        let target_ty = ty(r.to_node);
        let field = base_field(&r.to_field);
        let made = match interp.type_name.as_str() {
            "PositionInterpolator" | "SplinePositionInterpolator" => {
                let vals = interp
                    .value("keyValue")
                    .map(|v| v.as_tuples::<3>())
                    .unwrap_or_default();
                let n = keys.len().min(vals.len());
                let prop = match (target_ty, field) {
                    (_, "translation") | (_, "position") | (_, "location") => {
                        Some(AnimationProperty::Translation)
                    }
                    (_, "scale") => Some(AnimationProperty::Scale),
                    _ => None,
                };
                prop.filter(|_| n > 0).map(|p| {
                    (
                        p,
                        AnimationSampler {
                            keyframes: times(&keys[..n], cycle),
                            values: AnimationValues::Vec3(vals[..n].to_vec()),
                            interpolation: Interpolation::Linear,
                        },
                    )
                })
            }
            "OrientationInterpolator" | "SquadOrientationInterpolator" => {
                let vals = interp
                    .value("keyValue")
                    .map(|v| v.as_tuples::<4>())
                    .unwrap_or_default();
                let n = keys.len().min(vals.len());
                let ok = matches!(field, "rotation" | "orientation");
                (ok && n > 0).then(|| {
                    let mut quats: Vec<[f32; 4]> = Vec::with_capacity(n);
                    for v in &vals[..n] {
                        let mut q = quat_from_axis_angle([v[0], v[1], v[2], v[3] * c.angle]);
                        if let Some(p) = quats.last() {
                            let d = p[0] * q[0] + p[1] * q[1] + p[2] * q[2] + p[3] * q[3];
                            if d < 0.0 {
                                q = [-q[0], -q[1], -q[2], -q[3]];
                            }
                        }
                        quats.push(q);
                    }
                    (
                        AnimationProperty::Rotation,
                        AnimationSampler {
                            keyframes: times(&keys[..n], cycle),
                            values: AnimationValues::Quat(quats),
                            interpolation: Interpolation::Linear,
                        },
                    )
                })
            }
            "CoordinateInterpolator" if field == "point" => {
                if morph(c, r.from_node, r.to_node, &keys, cycle, &mut channels, ts) {
                    continue;
                }
                None
            }
            _ => None,
        };
        let Some((prop, sampler)) = made else {
            unmapped.push(json!(format!(
                "{}.{} -> {}.{}",
                r.from_def, r.from_field, r.to_def, r.to_field
            )));
            continue;
        };
        let targets = c.instances.get(&r.to_node).cloned().unwrap_or_default();
        if targets.is_empty() {
            continue;
        }
        for t in targets {
            let trs = matches!(
                c.scene.node(t).map(|n| n.transform),
                Some(Transform::Trs { .. })
            );
            if !trs {
                c.warnings.push(format!(
                    "animation target {} uses center/scaleOrientation; channel skipped",
                    r.to_def
                ));
                continue;
            }
            let mut sampler = sampler.clone();
            if let (Some(&(_, c0)), AnimationProperty::Translation, AnimationValues::Vec3(v)) =
                (c.pivots.get(&t), prop, &mut sampler.values)
            {
                for p in v.iter_mut() {
                    *p = [p[0] + c0[0], p[1] + c0[1], p[2] + c0[2]];
                }
            }
            channels
                .entry(ts)
                .or_default()
                .push(AnimationChannel::new(t, prop, sampler));
        }
    }
    let mut info = Vec::new();
    for (ts, ch) in channels {
        let n = doc.node(ts);
        let name = n
            .and_then(|n| n.def.clone())
            .unwrap_or_else(|| format!("TimeSensor_{}", ts.0));
        let mut a = Animation::new(Some(name.clone()));
        a.channels = ch;
        c.scene.add_animation(a);
        if let Some(n) = n {
            info.push(json!({
                "name": name,
                "cycleInterval": n.value("cycleInterval").and_then(|v| v.as_f64()).unwrap_or(1.0),
                "loop": n.value("loop").and_then(|v| v.as_bool()).unwrap_or(false),
            }));
        }
    }
    if !info.is_empty() {
        c.scene
            .extras
            .insert("x3d:animations".into(), Value::Array(info));
    }
    if !unmapped.is_empty() {
        unmapped.truncate(1024);
        c.scene
            .extras
            .insert("x3d:routes".into(), Value::Array(unmapped));
    }
}

/// CoordinateInterpolator → morph targets. Returns `true` when at
/// least one mesh was animated.
fn morph(
    c: &mut Conv<'_>,
    interp: NodeIdx,
    coord: NodeIdx,
    keys: &[f32],
    cycle: f32,
    channels: &mut BTreeMap<NodeIdx, Vec<AnimationChannel>>,
    ts: NodeIdx,
) -> bool {
    let doc = c.doc;
    let Some(inode) = doc.node(interp) else {
        return false;
    };
    let Some(cnode) = doc.node(coord) else {
        return false;
    };
    let base = cnode
        .value("point")
        .map(|v| v.as_tuples::<3>())
        .unwrap_or_default();
    let np = base.len();
    let vals = inode
        .value("keyValue")
        .map(|v| v.as_tuples::<3>())
        .unwrap_or_default();
    if np == 0 {
        return false;
    }
    let nk = keys.len().min(vals.len() / np).min(256);
    if nk == 0 {
        return false;
    }
    let users: Vec<_> = c
        .prim_sources
        .iter()
        .filter(|(_, (ci, _))| *ci == coord)
        .map(|(k, (_, s))| (*k, s.clone()))
        .collect();
    let mut any = false;
    for ((mesh_id, prim_i), src) in users {
        let nv = src.len();
        if nk.saturating_mul(nv) > c.geom.vertex_budget {
            c.warnings
                .push("CoordinateInterpolator morph exceeds vertex budget".into());
            continue;
        }
        c.geom.vertex_budget -= nk * nv;
        let Some(mesh) = c.scene.meshes.get_mut(mesh_id.0 as usize) else {
            continue;
        };
        if mesh.primitives.len() != 1 || !mesh.primitives[0].targets.is_empty() {
            continue;
        }
        let prim = &mut mesh.primitives[prim_i];
        for k in 0..nk {
            let mut t = MorphTarget::default();
            t.position = Some(
                src.iter()
                    .map(|&s| {
                        let s = s as usize;
                        let a = vals[k * np + s];
                        let b = base[s];
                        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
                    })
                    .collect(),
            );
            prim.targets.push(t);
        }
        mesh.weights = vec![0.0; nk];
        let weights: Vec<Vec<f32>> = (0..nk)
            .map(|k| {
                let mut w = vec![0.0; nk];
                w[k] = 1.0;
                w
            })
            .collect();
        let Some(sampler) = AnimationSampler::morph_weights(
            times(&keys[..nk], cycle),
            weights,
            Interpolation::Linear,
        ) else {
            continue;
        };
        let nodes: Vec<_> = c
            .scene
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.mesh == Some(mesh_id))
            .map(|(i, _)| oxideav_mesh3d::NodeId(i as u32))
            .collect();
        for nid in nodes {
            channels.entry(ts).or_default().push(AnimationChannel::new(
                nid,
                AnimationProperty::MorphWeights,
                sampler.clone(),
            ));
        }
        any = true;
    }
    any
}
