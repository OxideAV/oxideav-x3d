//! XML encoding reader (ISO/IEC 19776-1): XML tree → [`X3dDocument`].
//!
//! Handles the `<X3D>` header (`profile`, `version`, `<head>` with
//! `component` / `unit` / `meta`), the `<Scene>` node tree with
//! `DEF`/`USE` and `containerField`, `ROUTE`, `IMPORT`/`EXPORT`,
//! `ProtoDeclare` (`ProtoInterface` / `ProtoBody`),
//! `ExternProtoDeclare`, `ProtoInstance` + `fieldValue`, `IS` /
//! `connect`, and user-defined `field` declarations on `Script` and
//! shader nodes. Field values are parsed according to the node table
//! or the declared interface; unparsable values are skipped with a
//! warning instead of failing the whole document.

use std::collections::HashMap;

use crate::document::{
    ComponentDecl, Encoding, Export, FieldDecl, Import, IsConnect, NodeIdx, NodeKind, ProtoDecl,
    Route, SceneBody, UnitDecl, X3dDocument, X3dNode,
};
use crate::error::{Error, Result};
use crate::field::{parse_xml_value, AccessType, FieldType, FieldValue};
use crate::nodes;
use crate::xml::{self, Element, XmlNode};
use crate::Limits;

/// Parse an XML-encoded X3D document from text.
pub fn read_xml(src: &str, limits: &Limits) -> Result<X3dDocument> {
    if src.len() > limits.max_input_bytes {
        return Err(Error::limit("input larger than max_input_bytes"));
    }
    let root = xml::parse(src, limits)?;
    read_tree(&root, limits)
}

/// Convert an already-parsed XML tree into an [`X3dDocument`].
pub fn read_tree(root: &Element, limits: &Limits) -> Result<X3dDocument> {
    if root.local_name() != "X3D" {
        return Err(Error::invalid(format!(
            "root element is <{}>, expected <X3D>",
            root.name
        )));
    }
    let mut r = Reader {
        doc: X3dDocument {
            encoding: Encoding::Xml,
            version: root.attr("version").unwrap_or("").to_string(),
            profile: root.attr("profile").unwrap_or("").to_string(),
            ..X3dDocument::default()
        },
        scopes: Vec::new(),
        limits,
    };
    let mut scene: Option<&Element> = None;
    for el in root.elements() {
        match el.local_name() {
            "head" => r.read_head(el),
            "Scene" => {
                if scene.is_none() {
                    scene = Some(el);
                }
            }
            other => r.doc.warn(format!("ignoring <{other}> under <X3D>")),
        }
    }
    let scene = scene.ok_or_else(|| Error::invalid("missing <Scene> element"))?;
    r.scopes.push(Scope::default());
    let mut body = SceneBody::default();
    r.read_body(scene, &mut body)?;
    r.finish_scope(&mut body);
    r.doc.scene = body;
    Ok(r.doc)
}

#[derive(Default)]
struct Scope {
    defs: HashMap<String, NodeIdx>,
    protos: HashMap<String, usize>,
    pending_routes: Vec<(String, String, String, String)>,
}

struct Reader<'l> {
    doc: X3dDocument,
    scopes: Vec<Scope>,
    limits: &'l Limits,
}

/// Statement elements that may appear among children anywhere.
fn is_statement(name: &str) -> bool {
    matches!(
        name,
        "ROUTE" | "ProtoDeclare" | "ExternProtoDeclare" | "IMPORT" | "EXPORT"
    )
}

impl Reader<'_> {
    fn read_head(&mut self, head: &Element) {
        for el in head.elements() {
            match el.local_name() {
                "component" => {
                    let name = el.attr("name").unwrap_or("").to_string();
                    let level = el
                        .attr("level")
                        .and_then(|l| l.trim().parse().ok())
                        .unwrap_or(1);
                    self.doc.components.push(ComponentDecl { name, level });
                }
                "unit" => {
                    let factor = el
                        .attr("conversionFactor")
                        .and_then(|f| f.trim().parse::<f64>().ok())
                        .unwrap_or(1.0);
                    self.doc.units.push(UnitDecl {
                        category: el.attr("category").unwrap_or("").to_string(),
                        name: el.attr("name").unwrap_or("").to_string(),
                        conversion_factor: factor,
                    });
                }
                "meta" => {
                    self.doc.meta.push((
                        el.attr("name").unwrap_or("").to_string(),
                        el.attr("content").unwrap_or("").to_string(),
                    ));
                }
                _ => {}
            }
        }
    }

    fn scope(&mut self) -> &mut Scope {
        if self.scopes.is_empty() {
            self.scopes.push(Scope::default());
        }
        let n = self.scopes.len();
        &mut self.scopes[n - 1]
    }

    fn find_proto(&self, name: &str) -> Option<usize> {
        self.scopes
            .iter()
            .rev()
            .find_map(|s| s.protos.get(name).copied())
    }

    fn finish_scope(&mut self, body: &mut SceneBody) {
        let pending = std::mem::take(&mut self.scope().pending_routes);
        for (fnode, ffield, tnode, tfield) in pending {
            let defs = &self.scopes.last().map(|s| &s.defs);
            let from = defs.and_then(|d| d.get(&fnode).copied());
            let to = defs.and_then(|d| d.get(&tnode).copied());
            match (from, to) {
                (Some(from_node), Some(to_node)) => body.routes.push(Route {
                    from_node,
                    from_def: fnode,
                    from_field: ffield,
                    to_node,
                    to_def: tnode,
                    to_field: tfield,
                }),
                _ => self.doc.warn(format!(
                    "ROUTE {fnode}.{ffield} TO {tnode}.{tfield}: unknown DEF"
                )),
            }
        }
    }

    /// Read the children of a `<Scene>` / `<ProtoBody>` as a body.
    fn read_body(&mut self, el: &Element, body: &mut SceneBody) -> Result<()> {
        for child in el.elements() {
            let name = child.local_name();
            if is_statement(name) {
                self.read_statement(child, body)?;
            } else if matches!(name, "IS" | "field" | "fieldValue" | "connect") {
                self.doc
                    .warn(format!("<{name}> outside of its context ignored"));
            } else if let Some((idx, _)) = self.read_node(child, 1, body)? {
                body.roots.push(idx);
            }
        }
        Ok(())
    }

    fn read_statement(&mut self, el: &Element, body: &mut SceneBody) -> Result<()> {
        match el.local_name() {
            "ROUTE" => {
                let g = |k: &str| el.attr(k).unwrap_or("").trim().to_string();
                self.scope().pending_routes.push((
                    g("fromNode"),
                    g("fromField"),
                    g("toNode"),
                    g("toField"),
                ));
            }
            "IMPORT" => body.imports.push(Import {
                inline_def: el.attr("inlineDEF").unwrap_or("").to_string(),
                imported_def: el.attr("importedDEF").unwrap_or("").to_string(),
                as_name: el.attr("AS").map(str::to_string),
            }),
            "EXPORT" => body.exports.push(Export {
                local_def: el.attr("localDEF").unwrap_or("").to_string(),
                as_name: el.attr("AS").map(str::to_string),
            }),
            "ProtoDeclare" => {
                let idx = self.read_proto(el, false)?;
                body.protos.push(idx);
            }
            "ExternProtoDeclare" => {
                let idx = self.read_proto(el, true)?;
                body.protos.push(idx);
            }
            _ => {}
        }
        Ok(())
    }

    fn read_proto(&mut self, el: &Element, external: bool) -> Result<usize> {
        let name = el.attr("name").unwrap_or("").to_string();
        let mut decl = ProtoDecl {
            name: name.clone(),
            interface: Vec::new(),
            body: None,
            url: Vec::new(),
            appinfo: el.attr("appinfo").map(str::to_string),
        };
        if external {
            if let Some(u) = el.attr("url") {
                if let Ok(v) = parse_xml_value(FieldType::MFString, u) {
                    decl.url = v.as_strings().to_vec();
                }
            }
            for f in el.elements().filter(|e| e.local_name() == "field") {
                if let Some(d) = self.read_field_decl(f, 1)? {
                    decl.interface.push(d);
                }
            }
        } else {
            for part in el.elements() {
                if part.local_name() == "ProtoInterface" {
                    for f in part.elements().filter(|e| e.local_name() == "field") {
                        if let Some(d) = self.read_field_decl(f, 1)? {
                            decl.interface.push(d);
                        }
                    }
                }
            }
        }
        let idx = self.doc.protos.len();
        self.doc.protos.push(decl);
        // Register before reading the body so nested instances resolve
        // (recursion is bounded at expansion time).
        self.scope().protos.insert(name, idx);
        if !external {
            if let Some(b) = el.elements().find(|e| e.local_name() == "ProtoBody") {
                self.scopes.push(Scope::default());
                let mut body = SceneBody::default();
                let res = self.read_body(b, &mut body);
                self.finish_scope(&mut body);
                self.scopes.pop();
                res?;
                self.doc.protos[idx].body = Some(body);
            } else {
                self.doc.protos[idx].body = Some(SceneBody::default());
            }
        }
        Ok(idx)
    }

    /// `<field name type accessType value>` (with node children for
    /// SFNode / MFNode).
    fn read_field_decl(&mut self, el: &Element, depth: usize) -> Result<Option<FieldDecl>> {
        let name = el.attr("name").unwrap_or("").to_string();
        let Some(ty) = el.attr("type").and_then(FieldType::from_name) else {
            self.doc
                .warn(format!("field '{name}' has a missing/unknown type"));
            return Ok(None);
        };
        let access = el
            .attr("accessType")
            .and_then(AccessType::from_name)
            .unwrap_or(AccessType::InitializeOnly);
        let mut value = None;
        if matches!(ty, FieldType::SFNode | FieldType::MFNode) {
            let mut nodes_v = Vec::new();
            let mut scratch = SceneBody::default();
            for c in el.elements() {
                if is_statement(c.local_name()) {
                    continue;
                }
                if let Some((i, _)) = self.read_node(c, depth + 1, &mut scratch)? {
                    nodes_v.push(i);
                }
            }
            // `value="DEFname"` initialisation (19776-1 5.13 example 2).
            if let Some(v) = el.attr("value").map(str::trim) {
                if !v.is_empty() && v != "NULL" {
                    if let Some(&i) = self.scope().defs.get(v) {
                        nodes_v.push(i);
                    }
                }
            }
            if ty == FieldType::SFNode {
                nodes_v.truncate(1);
            }
            if access != AccessType::InputOnly && access != AccessType::OutputOnly {
                value = Some(FieldValue::nodes(ty, nodes_v));
            }
        } else if let Some(v) = el.attr("value") {
            match parse_xml_value(ty, v) {
                Ok(fv) => value = Some(fv),
                Err(e) => self.doc.warn(format!("field '{name}': {}", e.0)),
            }
        } else if matches!(access, AccessType::InitializeOnly | AccessType::InputOutput) {
            value = Some(FieldValue::empty(ty));
        }
        Ok(Some(FieldDecl {
            name,
            ty,
            access,
            value,
        }))
    }

    /// Parse one node element. Returns the node index and the effective
    /// container field (explicit `containerField`, else the type's
    /// default). `None` for elements that are not nodes.
    fn read_node(
        &mut self,
        el: &Element,
        depth: usize,
        body: &mut SceneBody,
    ) -> Result<Option<(NodeIdx, String)>> {
        if depth > self.limits.max_depth {
            return Err(Error::limit("node nesting deeper than max_depth"));
        }
        let tname = el.local_name();
        let explicit_cf = el.attr("containerField").map(|s| s.trim().to_string());

        if let Some(use_name) = el.attr("USE") {
            let use_name = use_name.trim();
            let found = self.scope().defs.get(use_name).copied();
            return match found {
                Some(i) => {
                    let cf = explicit_cf.unwrap_or_else(|| self.default_container(i));
                    Ok(Some((i, cf)))
                }
                None => {
                    self.doc
                        .warn(format!("line {}: USE '{use_name}' has no DEF", el.line));
                    Ok(None)
                }
            };
        }

        if self.doc.nodes.len() >= self.limits.max_nodes {
            return Err(Error::limit("more than max_nodes nodes"));
        }

        let mut node = if tname == "ProtoInstance" {
            let pname = el.attr("name").unwrap_or("").trim().to_string();
            let proto = self.find_proto(&pname);
            if proto.is_none() {
                self.doc.warn(format!(
                    "ProtoInstance '{pname}' has no declaration in scope"
                ));
            }
            let mut n = X3dNode::new(pname);
            n.kind = NodeKind::ProtoInstance { proto };
            n
        } else {
            X3dNode::new(tname)
        };
        node.line = el.line;
        node.container_field = explicit_cf.clone();
        node.def = el
            .attr("DEF")
            .map(|d| d.trim().to_string())
            .filter(|d| !d.is_empty());

        // Register the DEF before children so a (pathological) USE of an
        // ancestor resolves; consumers guard against the cycle.
        let idx = self.doc.add_node(node);
        if let Some(d) = self.doc.nodes[idx.index()].def.clone() {
            self.scope().defs.insert(d, idx);
        }

        let table = self.doc.nodes[idx.index()].def_table();
        let proto_iface: Option<Vec<(String, FieldType)>> = match self.doc.nodes[idx.index()].kind {
            NodeKind::ProtoInstance { proto: Some(p) } => Some(
                self.doc.protos[p]
                    .interface
                    .iter()
                    .map(|d| (d.name.clone(), d.ty))
                    .collect(),
            ),
            _ => None,
        };

        // User-defined field declarations first (Script / shaders), so
        // their attribute values can be typed.
        let mut decls = Vec::new();
        for c in el.elements().filter(|c| c.local_name() == "field") {
            if let Some(d) = self.read_field_decl(c, depth + 1)? {
                decls.push(d);
            }
        }

        // Attributes.
        let mut fields: Vec<(String, FieldValue)> = Vec::new();
        for (k, v) in &el.attrs {
            if matches!(
                k.as_str(),
                "DEF" | "USE" | "containerField" | "class" | "id" | "style"
            ) || k.starts_with("xmlns")
                || k.contains(':')
            {
                continue;
            }
            if tname == "ProtoInstance" && k == "name" {
                continue;
            }
            let ty = table
                .and_then(|t| t.field(k))
                .map(|f| f.ty)
                .or_else(|| decls.iter().find(|d| d.name == *k).map(|d| d.ty));
            match ty {
                Some(FieldType::SFNode | FieldType::MFNode) => {}
                Some(ty) => match parse_xml_value(ty, v) {
                    Ok(fv) => fields.push((k.clone(), fv)),
                    Err(e) => self.doc.warn(format!(
                        "line {}: {tname}.{k}: {} (value skipped)",
                        el.line, e.0
                    )),
                },
                None => fields.push((k.clone(), FieldValue::sf_string(v.clone()))),
            }
        }

        // Child content.
        let mut text = String::new();
        let mut cdata: Option<String> = None;
        for c in &el.children {
            let c = match c {
                XmlNode::Element(e) => e,
                XmlNode::Text(t) => {
                    text.push_str(t);
                    continue;
                }
                XmlNode::CData(t) => {
                    cdata.get_or_insert_with(String::new).push_str(t);
                    continue;
                }
            };
            let cname = c.local_name();
            match cname {
                "field" => {}
                "IS" => {
                    for con in c.elements().filter(|e| e.local_name() == "connect") {
                        self.doc.nodes[idx.index()].is_connects.push(IsConnect {
                            node_field: con.attr("nodeField").unwrap_or("").to_string(),
                            proto_field: con.attr("protoField").unwrap_or("").to_string(),
                        });
                    }
                }
                "fieldValue" => {
                    let fname = c.attr("name").unwrap_or("").to_string();
                    let fty = proto_iface
                        .as_ref()
                        .and_then(|i| i.iter().find(|(n, _)| *n == fname).map(|(_, t)| *t));
                    let has_nodes = c.elements().next().is_some();
                    match fty {
                        Some(t @ (FieldType::SFNode | FieldType::MFNode)) => {
                            let mut v = Vec::new();
                            for n in c.elements() {
                                if let Some((i, _)) = self.read_node(n, depth + 1, body)? {
                                    v.push(i);
                                }
                            }
                            if t == FieldType::SFNode {
                                v.truncate(1);
                            }
                            fields.push((fname, FieldValue::nodes(t, v)));
                        }
                        Some(t) => match c.attr("value").map(|v| parse_xml_value(t, v)) {
                            Some(Ok(fv)) => fields.push((fname, fv)),
                            Some(Err(e)) => self.doc.warn(format!("fieldValue '{fname}': {}", e.0)),
                            None => {}
                        },
                        None if has_nodes => {
                            let mut v = Vec::new();
                            for n in c.elements() {
                                if let Some((i, _)) = self.read_node(n, depth + 1, body)? {
                                    v.push(i);
                                }
                            }
                            fields.push((fname, FieldValue::nodes(FieldType::MFNode, v)));
                        }
                        None => {
                            if let Some(v) = c.attr("value") {
                                fields.push((fname, FieldValue::sf_string(v)));
                            }
                        }
                    }
                }
                n if is_statement(n) => self.read_statement(c, body)?,
                "connect" => {}
                _ => {
                    let Some((child, cf)) = self.read_node(c, depth + 1, body)? else {
                        continue;
                    };
                    let fty = table
                        .and_then(|t| t.field(&cf))
                        .map(|f| f.ty)
                        .or_else(|| decls.iter().find(|d| d.name == cf).map(|d| d.ty))
                        .filter(|t| matches!(t, FieldType::SFNode | FieldType::MFNode))
                        .unwrap_or(FieldType::MFNode);
                    if let Some(slot) = fields.iter_mut().find(|(n, _)| *n == cf) {
                        if let crate::field::FieldData::Node(v) = &mut slot.1.data {
                            if fty == FieldType::SFNode {
                                v.clear();
                            }
                            v.push(child);
                        }
                    } else {
                        fields.push((cf, FieldValue::nodes(fty, vec![child])));
                    }
                }
            }
        }

        let n = &mut self.doc.nodes[idx.index()];
        n.fields = fields;
        n.decls = decls;
        n.source_text = cdata.or_else(|| {
            let t = text.trim();
            (!t.is_empty()).then(|| t.to_string())
        });
        let cf = explicit_cf.unwrap_or_else(|| self.default_container(idx));
        Ok(Some((idx, cf)))
    }

    fn default_container(&self, idx: NodeIdx) -> String {
        let n = &self.doc.nodes[idx.index()];
        if let NodeKind::ProtoInstance { .. } = n.kind {
            return "children".into();
        }
        nodes::lookup(&n.type_name)
            .map(|d| d.container_field.to_string())
            .unwrap_or_else(|| "children".into())
    }
}
