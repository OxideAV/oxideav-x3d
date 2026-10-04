//! XML encoding writer (ISO/IEC 19776-1): [`X3dDocument`] → text.
//!
//! The first reference to a node writes it in full; later references
//! write `<Type USE='name'/>`. Nodes referenced more than once (or by
//! a `ROUTE`) without a `DEF` get a synthesised one. A
//! `containerField` attribute is emitted only when the parent field
//! differs from the child type's default.

use std::collections::{HashMap, HashSet};

use crate::document::{FieldDecl, NodeIdx, NodeKind, ProtoDecl, SceneBody, X3dDocument};
use crate::field::{format_xml_value, FieldData, FieldType};
use crate::nodes;
use crate::xml::write_attr;

/// Serialise a document with the XML encoding.
pub fn write_xml(doc: &X3dDocument) -> String {
    let mut w = Writer::new(doc);
    w.document();
    w.out
}

struct Writer<'d> {
    doc: &'d X3dDocument,
    out: String,
    names: HashMap<NodeIdx, String>,
    written: HashSet<NodeIdx>,
}

impl<'d> Writer<'d> {
    fn new(doc: &'d X3dDocument) -> Self {
        // Count references to decide which nodes need a DEF.
        let mut refs: HashMap<NodeIdx, usize> = HashMap::new();
        for n in &doc.nodes {
            for c in n.all_children() {
                *refs.entry(c).or_default() += 1;
            }
        }
        let count_body = |b: &SceneBody, refs: &mut HashMap<NodeIdx, usize>| {
            for r in &b.roots {
                *refs.entry(*r).or_default() += 1;
            }
            for r in &b.routes {
                *refs.entry(r.from_node).or_default() += 2;
                *refs.entry(r.to_node).or_default() += 2;
            }
        };
        count_body(&doc.scene, &mut refs);
        for p in &doc.protos {
            if let Some(b) = &p.body {
                count_body(b, &mut refs);
            }
        }
        for p in &doc.protos {
            for d in &p.interface {
                if let Some(v) = &d.value {
                    for c in v.as_nodes() {
                        *refs.entry(*c).or_default() += 1;
                    }
                }
            }
        }
        let mut names = HashMap::new();
        let mut used: HashSet<String> = doc.nodes.iter().filter_map(|n| n.def.clone()).collect();
        for (i, n) in doc.nodes.iter().enumerate() {
            let idx = NodeIdx(i as u32);
            if let Some(d) = &n.def {
                names.insert(idx, d.clone());
            } else if refs.get(&idx).copied().unwrap_or(0) > 1 {
                let mut k = i;
                let name = loop {
                    let cand = format!("{}_{k}", n.type_name);
                    if !used.contains(&cand) {
                        break cand;
                    }
                    k += 1_000_000;
                };
                used.insert(name.clone());
                names.insert(idx, name);
            }
        }
        Self {
            doc,
            out: String::new(),
            names,
            written: HashSet::new(),
        }
    }

    fn indent(&mut self, depth: usize) {
        for _ in 0..depth {
            self.out.push_str("  ");
        }
    }

    fn document(&mut self) {
        let d = self.doc;
        let version = if d.version.is_empty() {
            "4.0"
        } else {
            &d.version
        };
        let profile = if d.profile.is_empty() {
            "Full"
        } else {
            &d.profile
        };
        self.out
            .push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        self.out.push_str(&format!(
            "<!DOCTYPE X3D PUBLIC \"ISO//Web3D//DTD X3D {version}//EN\" \"https://www.web3d.org/specifications/x3d-{version}.dtd\">\n"
        ));
        self.out.push_str("<X3D");
        write_attr(&mut self.out, "profile", profile);
        write_attr(&mut self.out, "version", version);
        write_attr(
            &mut self.out,
            "xmlns:xsd",
            "http://www.w3.org/2001/XMLSchema-instance",
        );
        write_attr(
            &mut self.out,
            "xsd:noNamespaceSchemaLocation",
            &format!("https://www.web3d.org/specifications/x3d-{version}.xsd"),
        );
        self.out.push_str(">\n");
        if !d.components.is_empty() || !d.units.is_empty() || !d.meta.is_empty() {
            self.out.push_str("  <head>\n");
            for c in &d.components {
                self.out.push_str("    <component");
                write_attr(&mut self.out, "name", &c.name);
                write_attr(&mut self.out, "level", &c.level.to_string());
                self.out.push_str("/>\n");
            }
            for u in &d.units {
                self.out.push_str("    <unit");
                write_attr(&mut self.out, "category", &u.category);
                write_attr(&mut self.out, "name", &u.name);
                write_attr(
                    &mut self.out,
                    "conversionFactor",
                    &crate::field::fmt_f64(u.conversion_factor),
                );
                self.out.push_str("/>\n");
            }
            for (n, c) in &d.meta {
                self.out.push_str("    <meta");
                write_attr(&mut self.out, "name", n);
                write_attr(&mut self.out, "content", c);
                self.out.push_str("/>\n");
            }
            self.out.push_str("  </head>\n");
        }
        self.out.push_str("  <Scene>\n");
        self.body(&d.scene, 2);
        self.out.push_str("  </Scene>\n</X3D>\n");
    }

    fn body(&mut self, b: &SceneBody, depth: usize) {
        for &p in &b.protos {
            if let Some(decl) = self.doc.protos.get(p) {
                self.proto(decl, depth);
            }
        }
        for &r in &b.roots {
            self.node(r, None, depth);
        }
        for imp in &b.imports {
            self.indent(depth);
            self.out.push_str("<IMPORT");
            write_attr(&mut self.out, "inlineDEF", &imp.inline_def);
            write_attr(&mut self.out, "importedDEF", &imp.imported_def);
            if let Some(a) = &imp.as_name {
                write_attr(&mut self.out, "AS", a);
            }
            self.out.push_str("/>\n");
        }
        for exp in &b.exports {
            self.indent(depth);
            self.out.push_str("<EXPORT");
            write_attr(&mut self.out, "localDEF", &exp.local_def);
            if let Some(a) = &exp.as_name {
                write_attr(&mut self.out, "AS", a);
            }
            self.out.push_str("/>\n");
        }
        for r in &b.routes {
            let from = self
                .names
                .get(&r.from_node)
                .cloned()
                .unwrap_or_else(|| r.from_def.clone());
            let to = self
                .names
                .get(&r.to_node)
                .cloned()
                .unwrap_or_else(|| r.to_def.clone());
            self.indent(depth);
            self.out.push_str("<ROUTE");
            write_attr(&mut self.out, "fromNode", &from);
            write_attr(&mut self.out, "fromField", &r.from_field);
            write_attr(&mut self.out, "toNode", &to);
            write_attr(&mut self.out, "toField", &r.to_field);
            self.out.push_str("/>\n");
        }
    }

    fn proto(&mut self, p: &ProtoDecl, depth: usize) {
        let external = p.body.is_none();
        self.indent(depth);
        self.out.push_str(if external {
            "<ExternProtoDeclare"
        } else {
            "<ProtoDeclare"
        });
        write_attr(&mut self.out, "name", &p.name);
        if let Some(a) = &p.appinfo {
            write_attr(&mut self.out, "appinfo", a);
        }
        if external {
            let url = crate::field::FieldValue {
                ty: FieldType::MFString,
                data: FieldData::String(p.url.clone()),
            };
            write_attr(&mut self.out, "url", &format_xml_value(&url));
            self.out.push_str(">\n");
            for d in &p.interface {
                self.field_decl(d, depth + 1, false);
            }
            self.indent(depth);
            self.out.push_str("</ExternProtoDeclare>\n");
            return;
        }
        self.out.push_str(">\n");
        self.indent(depth + 1);
        self.out.push_str("<ProtoInterface>\n");
        for d in &p.interface {
            self.field_decl(d, depth + 2, true);
        }
        self.indent(depth + 1);
        self.out.push_str("</ProtoInterface>\n");
        self.indent(depth + 1);
        self.out.push_str("<ProtoBody>\n");
        if let Some(b) = &p.body {
            self.body(b, depth + 2);
        }
        self.indent(depth + 1);
        self.out.push_str("</ProtoBody>\n");
        self.indent(depth);
        self.out.push_str("</ProtoDeclare>\n");
    }

    fn field_decl(&mut self, d: &FieldDecl, depth: usize, with_value: bool) {
        self.indent(depth);
        self.out.push_str("<field");
        write_attr(&mut self.out, "name", &d.name);
        write_attr(&mut self.out, "type", d.ty.name());
        write_attr(&mut self.out, "accessType", d.access.name());
        let mut node_children: &[NodeIdx] = &[];
        if with_value {
            if let Some(v) = &d.value {
                if matches!(v.data, FieldData::Node(_)) {
                    node_children = v.as_nodes();
                } else {
                    write_attr(&mut self.out, "value", &format_xml_value(v));
                }
            }
        }
        if node_children.is_empty() {
            self.out.push_str("/>\n");
        } else {
            self.out.push_str(">\n");
            for &c in node_children {
                self.node(c, None, depth + 1);
            }
            self.indent(depth);
            self.out.push_str("</field>\n");
        }
    }

    /// Write node `idx` as the value of parent field `parent_field`
    /// (`None` at body level).
    fn node(&mut self, idx: NodeIdx, parent_field: Option<&str>, depth: usize) {
        let Some(n) = self.doc.node(idx) else {
            return;
        };
        let is_proto = matches!(n.kind, NodeKind::ProtoInstance { .. });
        let elem = if is_proto {
            "ProtoInstance"
        } else {
            n.type_name.as_str()
        };
        let default_cf = if is_proto {
            "children"
        } else {
            nodes::lookup(&n.type_name)
                .map(|d| d.container_field)
                .unwrap_or("children")
        };
        let cf_attr = match parent_field {
            Some(f) if f != default_cf => Some(f.to_string()),
            None => n.container_field.clone().filter(|c| c != default_cf),
            _ => None,
        };
        self.indent(depth);
        self.out.push('<');
        self.out.push_str(elem);
        if self.written.contains(&idx) {
            if let Some(name) = self.names.get(&idx).cloned() {
                write_attr(&mut self.out, "USE", &name);
            }
            if is_proto {
                write_attr(&mut self.out, "name", &n.type_name);
            }
            if let Some(cf) = cf_attr {
                write_attr(&mut self.out, "containerField", &cf);
            }
            self.out.push_str("/>\n");
            return;
        }
        self.written.insert(idx);
        if let Some(name) = self.names.get(&idx).cloned() {
            write_attr(&mut self.out, "DEF", &name);
        }
        if is_proto {
            write_attr(&mut self.out, "name", &n.type_name);
        }
        if let Some(cf) = &cf_attr {
            write_attr(&mut self.out, "containerField", cf);
        }
        let mut node_fields: Vec<(&str, &[NodeIdx])> = Vec::new();
        let mut proto_values: Vec<(&str, String)> = Vec::new();
        for (name, v) in &n.fields {
            if let FieldData::Node(list) = &v.data {
                node_fields.push((name.as_str(), list.as_slice()));
            } else if is_proto {
                proto_values.push((name.as_str(), format_xml_value(v)));
            } else {
                write_attr(&mut self.out, name, &format_xml_value(v));
            }
        }
        let has_content = !node_fields.is_empty()
            || !proto_values.is_empty()
            || !n.decls.is_empty()
            || !n.is_connects.is_empty()
            || n.source_text.is_some();
        if !has_content {
            self.out.push_str("/>\n");
            return;
        }
        self.out.push_str(">\n");
        for d in &n.decls {
            self.field_decl(d, depth + 1, true);
        }
        if !n.is_connects.is_empty() {
            self.indent(depth + 1);
            self.out.push_str("<IS>\n");
            for c in &n.is_connects {
                self.indent(depth + 2);
                self.out.push_str("<connect");
                write_attr(&mut self.out, "nodeField", &c.node_field);
                write_attr(&mut self.out, "protoField", &c.proto_field);
                self.out.push_str("/>\n");
            }
            self.indent(depth + 1);
            self.out.push_str("</IS>\n");
        }
        for (name, val) in proto_values {
            self.indent(depth + 1);
            self.out.push_str("<fieldValue");
            write_attr(&mut self.out, "name", name);
            write_attr(&mut self.out, "value", &val);
            self.out.push_str("/>\n");
        }
        for (name, list) in node_fields {
            if is_proto {
                self.indent(depth + 1);
                self.out.push_str("<fieldValue");
                write_attr(&mut self.out, "name", name);
                self.out.push_str(">\n");
                for &c in list {
                    self.node(c, None, depth + 2);
                }
                self.indent(depth + 1);
                self.out.push_str("</fieldValue>\n");
            } else {
                for &c in list {
                    self.node(c, Some(name), depth + 1);
                }
            }
        }
        if let Some(t) = &n.source_text {
            self.indent(depth + 1);
            self.out.push_str("<![CDATA[");
            self.out.push_str(&t.replace("]]>", "]]]]><![CDATA[>"));
            self.out.push_str("]]>\n");
        }
        self.indent(depth);
        self.out.push_str("</");
        self.out.push_str(elem);
        self.out.push_str(">\n");
    }
}
