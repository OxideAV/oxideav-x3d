//! Prototype expansion (ISO/IEC 19775-1 4.4.4).
//!
//! Every reachable `ProtoInstance` whose declaration has a body is
//! replaced by a deep copy of the body's first node (its *type node*),
//! with `IS` connections substituted by the instance's `fieldValue`s
//! or the interface defaults. Sharing inside the body is preserved per
//! instance; node-valued field values coming from the instance are
//! referenced, not copied. Nested instances are expanded lazily up to
//! [`Limits::max_proto_depth`](crate::Limits::max_proto_depth), and
//! every allocation counts against
//! [`Limits::max_nodes`](crate::Limits::max_nodes), so recursive
//! prototypes terminate.
//!
//! `ExternProtoDeclare` instances cannot be resolved without file
//! access; they stay in the graph as prototype instances and are
//! reported by the scene converter.

use std::collections::{HashMap, HashSet};

use crate::document::{NodeIdx, NodeKind, Route, X3dDocument};
use crate::error::{Error, Result};
use crate::field::{AccessType, FieldData};
use crate::Limits;

/// Expand all reachable prototype instances in place. Returns the
/// routes of the expanded bodies (endpoints remapped to the copies).
pub fn expand_all(doc: &mut X3dDocument, limits: &Limits) -> Result<Vec<Route>> {
    let mut ex = Expander {
        limits,
        memo: HashMap::new(),
        depth_of: HashMap::new(),
        routes: Vec::new(),
    };
    let roots = doc.scene.roots.clone();
    let mut new_roots = Vec::with_capacity(roots.len());
    for r in roots {
        if let Some(n) = ex.resolve(doc, r)? {
            new_roots.push(n);
        }
    }
    doc.scene.roots = new_roots;
    // Rewrite node references reachable from the roots.
    let mut seen: HashSet<NodeIdx> = HashSet::new();
    let mut work: Vec<NodeIdx> = doc.scene.roots.clone();
    while let Some(i) = work.pop() {
        if !seen.insert(i) {
            continue;
        }
        let nfields = doc.nodes[i.index()].fields.len();
        for f in 0..nfields {
            let list = match &doc.nodes[i.index()].fields[f].1.data {
                FieldData::Node(v) => v.clone(),
                _ => continue,
            };
            let mut out = Vec::with_capacity(list.len());
            for c in list {
                if let Some(n) = ex.resolve(doc, c)? {
                    out.push(n);
                }
            }
            for &c in &out {
                work.push(c);
            }
            if let FieldData::Node(v) = &mut doc.nodes[i.index()].fields[f].1.data {
                *v = out;
            }
        }
    }
    Ok(ex.routes)
}

struct Expander<'l> {
    limits: &'l Limits,
    memo: HashMap<NodeIdx, Option<NodeIdx>>,
    depth_of: HashMap<NodeIdx, usize>,
    routes: Vec<Route>,
}

impl Expander<'_> {
    /// Resolve `idx` to a non-prototype node when possible.
    fn resolve(&mut self, doc: &mut X3dDocument, idx: NodeIdx) -> Result<Option<NodeIdx>> {
        let mut cur = idx;
        let mut hops = 0usize;
        loop {
            let Some(node) = doc.node(cur) else {
                return Ok(None);
            };
            let NodeKind::ProtoInstance { proto: Some(p) } = node.kind else {
                return Ok(Some(cur));
            };
            if doc.protos.get(p).and_then(|d| d.body.as_ref()).is_none() {
                // Extern / unresolvable: keep as is.
                return Ok(Some(cur));
            }
            if let Some(m) = self.memo.get(&cur) {
                match m {
                    Some(n) => {
                        cur = *n;
                        hops += 1;
                        if hops > self.limits.max_proto_depth {
                            return Ok(None);
                        }
                        continue;
                    }
                    None => return Ok(None),
                }
            }
            let depth = self.depth_of.get(&cur).copied().unwrap_or(0);
            if depth >= self.limits.max_proto_depth {
                doc.warn(format!(
                    "prototype '{}' nested deeper than {} — dropped",
                    node.type_name, self.limits.max_proto_depth
                ));
                self.memo.insert(cur, None);
                return Ok(None);
            }
            // Mark in progress (guards self-instantiation through
            // node-valued interface defaults).
            self.memo.insert(cur, None);
            let expanded = self.expand(doc, cur, p, depth)?;
            self.memo.insert(cur, expanded);
            match expanded {
                Some(n) => {
                    cur = n;
                    hops += 1;
                    if hops > self.limits.max_proto_depth {
                        return Ok(None);
                    }
                }
                None => return Ok(None),
            }
        }
    }

    fn expand(
        &mut self,
        doc: &mut X3dDocument,
        inst: NodeIdx,
        proto: usize,
        depth: usize,
    ) -> Result<Option<NodeIdx>> {
        let body = match doc.protos[proto].body.clone() {
            Some(b) => b,
            None => return Ok(None),
        };
        let Some(&first) = body.roots.first() else {
            return Ok(None);
        };
        let mut map: HashMap<NodeIdx, NodeIdx> = HashMap::new();
        let mut copies = Vec::new();
        for &r in &body.roots {
            copies.push(self.copy(doc, r, inst, proto, depth, &mut map, 0)?);
        }
        for r in &body.routes {
            if let (Some(&f), Some(&t)) = (map.get(&r.from_node), map.get(&r.to_node)) {
                self.routes.push(Route {
                    from_node: f,
                    to_node: t,
                    ..r.clone()
                });
            }
        }
        let root = map.get(&first).copied();
        if let Some(root) = root {
            let def = doc.nodes[inst.index()].def.clone();
            if def.is_some() {
                doc.nodes[root.index()].def = def;
            }
        }
        Ok(root)
    }

    #[allow(clippy::too_many_arguments)]
    fn copy(
        &mut self,
        doc: &mut X3dDocument,
        old: NodeIdx,
        inst: NodeIdx,
        proto: usize,
        depth: usize,
        map: &mut HashMap<NodeIdx, NodeIdx>,
        level: usize,
    ) -> Result<NodeIdx> {
        if let Some(&n) = map.get(&old) {
            return Ok(n);
        }
        if level > self.limits.max_depth {
            return Err(Error::limit("prototype body nesting deeper than max_depth"));
        }
        if doc.nodes.len() >= self.limits.max_nodes {
            return Err(Error::limit("prototype expansion exceeds max_nodes"));
        }
        let mut node = doc.nodes[old.index()].clone();
        let connects = std::mem::take(&mut node.is_connects);
        let new = doc.add_node(node);
        map.insert(old, new);
        // Copy node-valued children.
        let nfields = doc.nodes[new.index()].fields.len();
        for f in 0..nfields {
            let list = match &doc.nodes[new.index()].fields[f].1.data {
                FieldData::Node(v) => v.clone(),
                _ => continue,
            };
            let mut out = Vec::with_capacity(list.len());
            for c in list {
                out.push(self.copy(doc, c, inst, proto, depth, map, level + 1)?);
            }
            if let FieldData::Node(v) = &mut doc.nodes[new.index()].fields[f].1.data {
                *v = out;
            }
        }
        // IS substitution.
        for c in connects {
            let decl = doc.protos[proto]
                .interface
                .iter()
                .find(|d| d.name == c.proto_field)
                .cloned();
            let Some(decl) = decl else { continue };
            if matches!(decl.access, AccessType::InputOnly | AccessType::OutputOnly) {
                continue;
            }
            let value = doc.nodes[inst.index()]
                .get(&c.proto_field)
                .cloned()
                .or(decl.value.clone());
            if let Some(mut v) = value {
                // Keep the destination field's declared type when known.
                if let Some(fd) = doc.nodes[new.index()]
                    .def_table()
                    .and_then(|t| t.field(&c.node_field))
                {
                    if fd.ty.storage() == v.ty.storage() {
                        v.ty = fd.ty;
                    }
                }
                doc.nodes[new.index()].set(c.node_field.clone(), v);
            }
        }
        if matches!(doc.nodes[new.index()].kind, NodeKind::ProtoInstance { .. }) {
            self.depth_of.insert(new, depth + 1);
        }
        Ok(new)
    }
}
