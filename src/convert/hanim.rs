//! H-Anim (ISO/IEC 19774) humanoid skin binding → mesh3d skeleton +
//! skin.
//!
//! An `HAnimHumanoid` lists its joints (`joints`, or every
//! `HAnimJoint` under `skeleton` in depth-first order when empty), a
//! shared skin coordinate node (`skinCoord`) and the skin shapes
//! (`skin`) whose geometry uses that coordinate node. Each joint
//! carries `skinCoordIndex` / `skinCoordWeight` pairs. The skin
//! coordinates are authored in the humanoid's default (rest) pose, so
//! the inverse bind matrix of a joint is the inverse of its rest
//! transform relative to the humanoid. Every vertex keeps its four
//! strongest influences (renormalised); vertices no joint influences
//! are bound to the humanoid node itself so they stay rigid.

use std::collections::HashMap;

use oxideav_mesh3d::{NodeId, Skeleton, Skin};
use serde_json::json;

use super::math::{mat_inverse, mat_mul, Mat4, IDENTITY};
use super::Conv;
use crate::document::NodeIdx;

/// Humanoid-relative rest matrices of every node under `root`.
fn relative_matrices(c: &Conv<'_>, root: NodeId) -> HashMap<NodeId, Mat4> {
    let mut out = HashMap::new();
    let mut stack = vec![(root, IDENTITY)];
    let mut guard = 0usize;
    while let Some((id, m)) = stack.pop() {
        guard += 1;
        if guard > c.scene.nodes.len() + 1 {
            break;
        }
        out.insert(id, m);
        if let Some(n) = c.scene.node(id) {
            for &ch in &n.children {
                if let Some(cn) = c.scene.node(ch) {
                    let cm = mat_mul(&m, &cn.transform.to_matrix());
                    stack.push((ch, cm));
                }
            }
        }
    }
    out
}

fn dfs_joints(c: &Conv<'_>, idx: NodeIdx, out: &mut Vec<NodeIdx>, depth: usize) {
    if depth > 512 || out.len() > 4096 {
        return;
    }
    let Some(n) = c.doc.node(idx) else { return };
    if n.type_name == "HAnimJoint" {
        if out.contains(&idx) {
            return;
        }
        out.push(idx);
    }
    for &ch in n.children_of("children") {
        dfs_joints(c, ch, out, depth + 1);
    }
}

pub(crate) fn bind_skins(c: &mut Conv<'_>) {
    let humanoids = c.humanoids.clone();
    for (h_idx, h_id) in humanoids {
        let Some(h) = c.doc.node(h_idx) else { continue };
        let Some(&skin_coord) = h.children_of("skinCoord").first() else {
            continue;
        };
        let mut joints: Vec<NodeIdx> = h.children_of("joints").to_vec();
        if joints.is_empty() {
            for &s in h.children_of("skeleton") {
                dfs_joints(c, s, &mut joints, 0);
            }
        }
        if joints.is_empty() {
            continue;
        }
        let rel = relative_matrices(c, h_id);
        // mesh3d joint nodes: first instance under this humanoid.
        let mut joint_nodes: Vec<NodeId> = Vec::new();
        let mut ibms: Vec<Mat4> = Vec::new();
        let mut joint_slot: HashMap<NodeIdx, u16> = HashMap::new();
        for &j in &joints {
            let Some(outer) = c
                .instances
                .get(&j)
                .and_then(|v| v.iter().find(|id| rel.contains_key(id)).copied())
            else {
                continue;
            };
            // Split transforms: the pivot carries the full joint frame.
            let inst = c.pivots.get(&outer).map(|p| p.0).unwrap_or(outer);
            if !rel.contains_key(&inst) {
                continue;
            }
            let ibm = mat_inverse(&rel[&inst]).unwrap_or(IDENTITY);
            joint_slot.insert(j, joint_nodes.len() as u16);
            joint_nodes.push(inst);
            ibms.push(ibm);
            if joint_nodes.len() >= u16::MAX as usize - 1 {
                break;
            }
        }
        if joint_nodes.is_empty() {
            continue;
        }
        // Root slot for unbound vertices.
        let root_slot = joint_nodes.len() as u16;
        joint_nodes.push(h_id);
        ibms.push(IDENTITY);
        // Influences per skin coordinate.
        let mut infl: HashMap<u32, Vec<(u16, f32)>> = HashMap::new();
        for (&j, &slot) in &joint_slot {
            let Some(jn) = c.doc.node(j) else { continue };
            let idx = jn
                .value("skinCoordIndex")
                .map(|v| v.as_i32s().to_vec())
                .unwrap_or_default();
            let w = jn
                .value("skinCoordWeight")
                .map(|v| v.as_f32s())
                .unwrap_or_default();
            for (k, &ci) in idx.iter().enumerate() {
                if ci < 0 {
                    continue;
                }
                let weight = w.get(k).or(w.first()).copied().unwrap_or(1.0);
                if weight > 0.0 && weight.is_finite() {
                    infl.entry(ci as u32).or_default().push((slot, weight));
                }
            }
        }
        let skel = Skeleton {
            name: h.def.clone(),
            joints: joint_nodes.clone(),
            inverse_bind_matrices: ibms,
        };
        let skel_id = c.scene.add_skeleton(skel);
        let skin_id = c.scene.add_skin(Skin::new(skel_id).with_root(h_id));
        let mut bound = 0usize;
        let users: Vec<_> = c
            .prim_sources
            .iter()
            .filter(|(_, (ci, _))| *ci == skin_coord)
            .map(|(k, (_, s))| (*k, s.clone()))
            .collect();
        for ((mesh_id, prim_i), src) in users {
            let Some(prim) = c
                .scene
                .meshes
                .get_mut(mesh_id.0 as usize)
                .and_then(|m| m.primitives.get_mut(prim_i))
            else {
                continue;
            };
            let mut js = Vec::with_capacity(src.len());
            let mut ws = Vec::with_capacity(src.len());
            for s in &src {
                let mut list = infl.get(s).cloned().unwrap_or_default();
                list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                list.truncate(4);
                let total: f32 = list.iter().map(|x| x.1).sum();
                let mut j4 = [root_slot, 0, 0, 0];
                let mut w4 = [1.0, 0.0, 0.0, 0.0];
                if total > 0.0 {
                    for (k, (slot, w)) in list.iter().enumerate() {
                        j4[k] = *slot;
                        w4[k] = w / total;
                    }
                    for k in list.len()..4 {
                        j4[k] = 0;
                        w4[k] = 0.0;
                    }
                }
                js.push(j4);
                ws.push(w4);
            }
            prim.joints = Some(js);
            prim.weights = Some(ws);
            bound += 1;
            let nodes: Vec<usize> = c
                .scene
                .nodes
                .iter()
                .enumerate()
                .filter(|(i, n)| n.mesh == Some(mesh_id) && rel.contains_key(&NodeId(*i as u32)))
                .map(|(i, _)| i)
                .collect();
            for i in nodes {
                c.scene.nodes[i].skin = Some(skin_id);
            }
        }
        if let Some(hn) = c.scene.node_mut(h_id) {
            hn.extras.insert(
                "x3d:hanimSkin".into(),
                json!({"joints": joint_nodes.len() - 1, "skinnedPrimitives": bound}),
            );
        }
    }
}
