//! ClassicVRML encoding (ISO/IEC 19776-2, `.x3dv`) via the shared
//! VRML-syntax layer of [`oxideav_vrml`].
//!
//! The ClassicVRML grammar is the VRML97 grammar plus the X3D access
//! keywords, field types and header statements (`PROFILE`,
//! `COMPONENT`, `UNIT`, `META`, `IMPORT`, `EXPORT`). Rather than
//! duplicating a lexer and parser, this module reuses
//! `oxideav_vrml::syntax` in its `Dialect::X3dClassic` mode, supplying
//! the X3D 4.0 node interfaces through [`X3dCatalog`] (built from the
//! same X3DUOM-generated table the XML reader uses), and translates
//! between the VRML AST and [`X3dDocument`] in both directions.
//! Prototype expansion stays in this crate (on the document model) so
//! both encodings share one conversion path.

use std::collections::HashMap;
use std::sync::OnceLock;

use oxideav_vrml::ast as v;
use oxideav_vrml::syntax::{self, Dialect, FieldSchema, NodeCatalog, NodeSchema};

use crate::document::{
    ComponentDecl, Encoding, Export, FieldDecl, Import, IsConnect, NodeIdx, NodeKind, ProtoDecl,
    Route, SceneBody, UnitDecl, X3dDocument, X3dNode,
};
use crate::error::{Error, Result};
use crate::field::{format_value, AccessType, FieldData, FieldType, FieldValue, SFImage};
use crate::{nodes, Limits};

fn to_v_access(a: AccessType) -> v::AccessType {
    match a {
        AccessType::InitializeOnly => v::AccessType::Field,
        AccessType::InputOnly => v::AccessType::EventIn,
        AccessType::OutputOnly => v::AccessType::EventOut,
        AccessType::InputOutput => v::AccessType::ExposedField,
    }
}

fn from_v_access(a: v::AccessType) -> AccessType {
    match a {
        v::AccessType::Field => AccessType::InitializeOnly,
        v::AccessType::EventIn => AccessType::InputOnly,
        v::AccessType::EventOut => AccessType::OutputOnly,
        v::AccessType::ExposedField => AccessType::InputOutput,
    }
}

fn to_v_type(t: FieldType) -> Option<v::FieldType> {
    v::FieldType::from_name(t.name())
}

fn from_v_type(t: v::FieldType) -> Option<FieldType> {
    FieldType::from_name(t.name())
}

/// ClassicVRML spelling of a value (brackets around MF values).
fn classic_text(val: &FieldValue) -> String {
    let body = format_value(val, true);
    if val.ty.is_mf() {
        format!("[ {body} ]")
    } else {
        body
    }
}

static DEFAULTS: OnceLock<Vec<Vec<String>>> = OnceLock::new();
static FIELDS: OnceLock<Vec<Vec<FieldSchema>>> = OnceLock::new();
static SCHEMAS: OnceLock<Vec<NodeSchema>> = OnceLock::new();

fn schemas() -> &'static [NodeSchema] {
    SCHEMAS.get_or_init(|| {
        let defaults = DEFAULTS.get_or_init(|| {
            nodes::all()
                .iter()
                .map(|d| {
                    d.fields
                        .iter()
                        .map(|f| match f.default {
                            Some("NULL") => "NULL".to_string(),
                            Some(_) => classic_text(&f.default_value()),
                            None => String::new(),
                        })
                        .collect()
                })
                .collect()
        });
        let fields = FIELDS.get_or_init(|| {
            nodes::all()
                .iter()
                .zip(defaults)
                .map(|(d, defs)| {
                    d.fields
                        .iter()
                        .zip(defs)
                        .filter_map(|(f, def)| {
                            Some(FieldSchema {
                                access: to_v_access(f.access),
                                field_type: to_v_type(f.ty)?,
                                name: f.name,
                                default: def.as_str(),
                            })
                        })
                        .collect()
                })
                .collect()
        });
        nodes::all()
            .iter()
            .zip(fields)
            .map(|(d, f)| NodeSchema {
                name: d.name,
                fields: f.as_slice(),
            })
            .collect()
    })
}

/// The X3D 4.0 node interfaces as an `oxideav_vrml` node catalogue.
#[derive(Clone, Copy, Debug, Default)]
pub struct X3dCatalog;

impl NodeCatalog for X3dCatalog {
    fn node(&self, name: &str) -> Option<&NodeSchema> {
        let s = schemas();
        s.binary_search_by(|n| n.name.cmp(name)).ok().map(|i| &s[i])
    }
}

/// `true` when the text starts like a ClassicVRML file (`#X3D` header).
pub fn looks_classic(text: &str) -> bool {
    text.trim_start_matches('\u{feff}')
        .trim_start()
        .starts_with("#X3D")
}

fn map_err(e: oxideav_vrml::Error) -> Error {
    let s = e.to_string();
    if s.contains("limit") || s.contains("more than") {
        Error::LimitExceeded(s)
    } else {
        Error::Invalid(format!("ClassicVRML: {s}"))
    }
}

/// Parse ClassicVRML text into an [`X3dDocument`].
pub fn read_classic(src: &str, limits: &Limits) -> Result<X3dDocument> {
    if src.len() > limits.max_input_bytes {
        return Err(Error::limit("input larger than max_input_bytes"));
    }
    let opts = syntax::ParseOptions {
        dialect: Dialect::X3dClassic,
        limits: syntax::ParseLimits {
            max_depth: limits
                .max_depth
                .min(syntax::ParseLimits::default().max_depth),
            max_nodes: limits.max_nodes,
            ..syntax::ParseLimits::default()
        },
        catalog: &X3dCatalog,
        require_header: true,
    };
    let vdoc = syntax::parse_with(src, &opts).map_err(map_err)?;
    if vdoc.header.format != "X3D" {
        return Err(Error::invalid(format!(
            "header '#{}' is not a ClassicVRML X3D header",
            vdoc.header.format
        )));
    }
    Ok(Importer::new(&vdoc).run())
}

struct Importer<'v> {
    v: &'v v::Document,
    doc: X3dDocument,
}

impl<'v> Importer<'v> {
    fn new(v: &'v v::Document) -> Self {
        Self {
            v,
            doc: X3dDocument {
                encoding: Encoding::ClassicVrml,
                version: v.header.version.trim_start_matches(['V', 'v']).to_string(),
                ..X3dDocument::default()
            },
        }
    }

    fn proto_index(&self, o: v::NodeOrigin) -> Option<usize> {
        match o {
            v::NodeOrigin::Proto(p) => Some(p.0 as usize),
            v::NodeOrigin::ExternProto(e) => Some(self.v.protos.len() + e.0 as usize),
            _ => None,
        }
    }

    fn value(&mut self, val: &v::FieldValue) -> Option<FieldValue> {
        let ty = from_v_type(val.ty)?;
        let data = match &val.data {
            v::FieldData::Bools(b) => FieldData::Bool(b.clone()),
            v::FieldData::Int32s(b) => FieldData::Int32(b.clone()),
            v::FieldData::Floats(b) => FieldData::Float(b.clone()),
            v::FieldData::Doubles(b) => FieldData::Double(b.clone()),
            v::FieldData::Strings(b) => FieldData::String(b.clone()),
            v::FieldData::Images(b) => FieldData::Image(
                b.iter()
                    .map(|i| SFImage {
                        width: i.width,
                        height: i.height,
                        components: i.components,
                        pixels: i.pixels.clone(),
                    })
                    .collect(),
            ),
            v::FieldData::Nodes(b) => FieldData::Node(b.iter().map(|n| NodeIdx(n.0)).collect()),
        };
        Some(FieldValue { ty, data })
    }

    fn decl(&mut self, d: &v::InterfaceDecl) -> Option<FieldDecl> {
        Some(FieldDecl {
            name: d.name.clone(),
            ty: from_v_type(d.field_type)?,
            access: from_v_access(d.access),
            value: d.value.as_ref().and_then(|x| self.value(x)),
        })
    }

    fn run(mut self) -> X3dDocument {
        // Prototype table: PROTOs then EXTERNPROTOs.
        for p in &self.v.protos {
            let iface = p.interface.iter().filter_map(|d| self.decl(d)).collect();
            self.doc.protos.push(ProtoDecl {
                name: p.name.clone(),
                interface: iface,
                body: Some(SceneBody::default()),
                url: Vec::new(),
                appinfo: None,
            });
        }
        for p in &self.v.extern_protos {
            let iface = p.interface.iter().filter_map(|d| self.decl(d)).collect();
            self.doc.protos.push(ProtoDecl {
                name: p.name.clone(),
                interface: iface,
                body: None,
                url: p.urls.clone(),
                appinfo: None,
            });
        }
        // Node arena, 1:1.
        for (i, n) in self.v.nodes.iter().enumerate() {
            let mut x = X3dNode::new(n.type_name.clone());
            x.kind = match n.origin {
                v::NodeOrigin::Builtin if nodes::lookup(&n.type_name).is_some() => {
                    NodeKind::Builtin
                }
                v::NodeOrigin::Proto(_) | v::NodeOrigin::ExternProto(_) => {
                    NodeKind::ProtoInstance {
                        proto: self.proto_index(n.origin),
                    }
                }
                _ => NodeKind::Unknown,
            };
            x.def = n.def_name.clone();
            for f in &n.fields {
                match &f.binding {
                    v::FieldBinding::Value(val) => match self.value(val) {
                        Some(fv) => x.fields.push((f.name.clone(), fv)),
                        None => self.doc.warn(format!(
                            "node {i} ({}): field {} has an unsupported type",
                            n.type_name, f.name
                        )),
                    },
                    v::FieldBinding::Is(target) => x.is_connects.push(IsConnect {
                        node_field: f.name.clone(),
                        proto_field: target.clone(),
                    }),
                }
            }
            for d in &n.interface {
                if let Some(is) = &d.is {
                    x.is_connects.push(IsConnect {
                        node_field: d.name.clone(),
                        proto_field: is.clone(),
                    });
                }
                if let Some(fd) = self.decl(d) {
                    x.decls.push(fd);
                }
            }
            self.doc.nodes.push(x);
        }
        // Bodies.
        let top = self.body(&self.v.statements.clone(), true);
        self.doc.scene = top;
        for (pi, p) in self.v.protos.iter().enumerate() {
            let b = self.body(&p.body, false);
            self.doc.protos[pi].body = Some(b);
        }
        self.doc
    }

    /// Convert a statement list (plus statements nested in node
    /// bodies) into a [`SceneBody`].
    fn body(&mut self, stmts: &[v::Statement], top: bool) -> SceneBody {
        let mut b = SceneBody::default();
        let mut nested: Vec<v::Statement> = Vec::new();
        for s in stmts {
            if let v::Statement::Node(id) = s {
                b.roots.push(NodeIdx(id.0));
                self.collect_nested(*id, &mut nested, &mut Vec::new(), 0);
            }
        }
        for s in stmts.iter().chain(nested.iter()) {
            match s {
                v::Statement::Node(_) => {}
                v::Statement::Proto(p) => b.protos.push(p.0 as usize),
                v::Statement::ExternProto(e) => b.protos.push(self.v.protos.len() + e.0 as usize),
                v::Statement::Route(r) => match (r.from_id, r.to_id) {
                    (Some(f), Some(t)) => b.routes.push(Route {
                        from_node: NodeIdx(f.0),
                        from_def: r.from_node.clone(),
                        from_field: r.from_field.clone(),
                        to_node: NodeIdx(t.0),
                        to_def: r.to_node.clone(),
                        to_field: r.to_field.clone(),
                    }),
                    _ => self.doc.warn(format!(
                        "ROUTE {}.{} TO {}.{}: unresolved",
                        r.from_node, r.from_field, r.to_node, r.to_field
                    )),
                },
                v::Statement::Profile(p) if top => self.doc.profile = p.clone(),
                v::Statement::Component { name, level } if top => {
                    self.doc.components.push(ComponentDecl {
                        name: name.clone(),
                        level: (*level).max(0) as u32,
                    })
                }
                v::Statement::Meta { name, content } if top => {
                    self.doc.meta.push((name.clone(), content.clone()))
                }
                v::Statement::Unit {
                    category,
                    name,
                    factor,
                } if top => self.doc.units.push(UnitDecl {
                    category: category.clone(),
                    name: name.clone(),
                    conversion_factor: *factor,
                }),
                v::Statement::Import {
                    inline_def,
                    exported,
                    as_name,
                } => b.imports.push(Import {
                    inline_def: inline_def.clone(),
                    imported_def: exported.clone(),
                    as_name: as_name.clone(),
                }),
                v::Statement::Export { local, as_name } => b.exports.push(Export {
                    local_def: local.clone(),
                    as_name: as_name.clone(),
                }),
                _ => {}
            }
        }
        b
    }

    /// Statements written inside node bodies belong to the enclosing
    /// scope.
    fn collect_nested(
        &self,
        id: v::NodeId,
        out: &mut Vec<v::Statement>,
        seen: &mut Vec<v::NodeId>,
        depth: usize,
    ) {
        if depth > 1024 || seen.contains(&id) || seen.len() > 1 << 20 {
            return;
        }
        seen.push(id);
        let Some(n) = self.v.node(id) else { return };
        out.extend(n.statements.iter().cloned());
        for c in n.child_ids() {
            self.collect_nested(c, out, seen, depth + 1);
        }
    }
}

/// Serialise a document with the ClassicVRML encoding.
pub fn write_classic(doc: &X3dDocument) -> String {
    let mut out = v::Document::new();
    let version = if doc.version.is_empty() {
        "4.0"
    } else {
        &doc.version
    };
    out.header = v::Header {
        format: "X3D".into(),
        version: format!("V{version}"),
        encoding: "utf8".into(),
        comment: String::new(),
    };
    let to_value = |val: &FieldValue| -> Option<v::FieldValue> {
        let ty = to_v_type(val.ty)?;
        let data = match &val.data {
            FieldData::Bool(b) => v::FieldData::Bools(b.clone()),
            FieldData::Int32(b) => v::FieldData::Int32s(b.clone()),
            FieldData::Float(b) => v::FieldData::Floats(b.clone()),
            FieldData::Double(b) => v::FieldData::Doubles(b.clone()),
            FieldData::String(b) => v::FieldData::Strings(b.clone()),
            FieldData::Image(b) => v::FieldData::Images(
                b.iter()
                    .map(|i| v::Image {
                        width: i.width,
                        height: i.height,
                        components: i.components,
                        pixels: i.pixels.clone(),
                    })
                    .collect(),
            ),
            FieldData::Node(b) => v::FieldData::Nodes(b.iter().map(|n| v::NodeId(n.0)).collect()),
        };
        Some(v::FieldValue { ty, data })
    };
    let to_decl = |d: &FieldDecl, is: Option<String>| -> Option<v::InterfaceDecl> {
        Some(v::InterfaceDecl {
            access: to_v_access(d.access),
            field_type: to_v_type(d.ty)?,
            name: d.name.clone(),
            value: if is.is_some() {
                None
            } else {
                d.value.as_ref().and_then(to_value)
            },
            is,
        })
    };
    // Protos first so instance origins can reference them.
    let mut proto_ids: HashMap<usize, v::NodeOrigin> = HashMap::new();
    for (i, p) in doc.protos.iter().enumerate() {
        let mut iface: Vec<v::InterfaceDecl> = p
            .interface
            .iter()
            .filter_map(|d| to_decl(d, None))
            .collect();
        if p.body.is_none() {
            // EXTERNPROTO interfaces carry no values.
            for d in &mut iface {
                d.value = None;
            }
        }
        match &p.body {
            Some(_) => {
                let id = out.add_proto(v::ProtoDecl {
                    name: p.name.clone(),
                    interface: iface,
                    body: Vec::new(),
                });
                proto_ids.insert(i, v::NodeOrigin::Proto(id));
            }
            None => {
                let id = out.add_extern_proto(v::ExternProtoDecl {
                    name: p.name.clone(),
                    interface: iface,
                    urls: p.url.clone(),
                });
                proto_ids.insert(i, v::NodeOrigin::ExternProto(id));
            }
        }
    }
    for n in &doc.nodes {
        let mut vn = v::Node::new(n.type_name.clone());
        vn.def_name = n.def.clone();
        vn.origin = match n.kind {
            NodeKind::Builtin => v::NodeOrigin::Builtin,
            NodeKind::ProtoInstance { proto: Some(p) } => {
                proto_ids.get(&p).copied().unwrap_or(v::NodeOrigin::Unknown)
            }
            _ => v::NodeOrigin::Unknown,
        };
        let is_of = |name: &str| {
            n.is_connects
                .iter()
                .find(|c| c.node_field == name)
                .map(|c| c.proto_field.clone())
        };
        for d in &n.decls {
            if let Some(vd) = to_decl(d, is_of(&d.name)) {
                vn.interface.push(vd);
            }
        }
        for (name, val) in &n.fields {
            if n.decls.iter().any(|d| d.name == *name) {
                // Script field values live in the interface block.
                if let Some(vd) = vn.interface.iter_mut().find(|d| d.name == *name) {
                    vd.value = to_value(val);
                }
                continue;
            }
            if let Some(vv) = to_value(val) {
                vn.fields.push(v::Field {
                    name: name.clone(),
                    binding: v::FieldBinding::Value(vv),
                    inferred: n.kind == NodeKind::Unknown,
                });
            }
        }
        if let Some(src) = &n.source_text {
            // ClassicVRML has no CDATA: inline source goes in `url`.
            if n.get("url").is_none() {
                vn.fields.push(v::Field::value(
                    "url",
                    v::FieldValue::mf_string(vec![src.trim().to_string()]),
                ));
            }
        }
        for c in &n.is_connects {
            if n.decls.iter().any(|d| d.name == c.node_field) {
                continue;
            }
            vn.fields
                .push(v::Field::is(c.node_field.clone(), c.proto_field.clone()));
        }
        out.add_node(vn);
    }
    let body_statements = |b: &SceneBody, out: &mut v::Document, top: bool| -> Vec<v::Statement> {
        let mut st = Vec::new();
        if top {
            if !doc.profile.is_empty() {
                st.push(v::Statement::Profile(doc.profile.clone()));
            }
            for c in &doc.components {
                st.push(v::Statement::Component {
                    name: c.name.clone(),
                    level: c.level as i32,
                });
            }
            for u in &doc.units {
                st.push(v::Statement::Unit {
                    category: u.category.clone(),
                    name: u.name.clone(),
                    factor: u.conversion_factor,
                });
            }
            for (n, c) in &doc.meta {
                st.push(v::Statement::Meta {
                    name: n.clone(),
                    content: c.clone(),
                });
            }
        }
        for p in &b.protos {
            match proto_ids.get(p) {
                Some(v::NodeOrigin::Proto(id)) => st.push(v::Statement::Proto(*id)),
                Some(v::NodeOrigin::ExternProto(id)) => st.push(v::Statement::ExternProto(*id)),
                _ => {}
            }
        }
        for r in &b.roots {
            st.push(v::Statement::Node(v::NodeId(r.0)));
        }
        for i in &b.imports {
            st.push(v::Statement::Import {
                inline_def: i.inline_def.clone(),
                exported: i.imported_def.clone(),
                as_name: i.as_name.clone(),
            });
        }
        for e in &b.exports {
            st.push(v::Statement::Export {
                local: e.local_def.clone(),
                as_name: e.as_name.clone(),
            });
        }
        for r in &b.routes {
            let def = |i: NodeIdx, fallback: &str| -> String {
                out.node(v::NodeId(i.0))
                    .and_then(|n| n.def_name.clone())
                    .unwrap_or_else(|| fallback.to_string())
            };
            st.push(v::Statement::Route(v::Route {
                from_node: def(r.from_node, &r.from_def),
                from_field: r.from_field.clone(),
                to_node: def(r.to_node, &r.to_def),
                to_field: r.to_field.clone(),
                from_id: Some(v::NodeId(r.from_node.0)),
                to_id: Some(v::NodeId(r.to_node.0)),
            }));
        }
        st
    };
    for (i, p) in doc.protos.iter().enumerate() {
        if let (Some(b), Some(v::NodeOrigin::Proto(id))) = (&p.body, proto_ids.get(&i)) {
            let st = body_statements(b, &mut out, false);
            out.protos[id.0 as usize].body = st;
        }
    }
    let st = body_statements(&doc.scene, &mut out, true);
    out.statements = st;
    syntax::write_document_with(
        &out,
        &syntax::WriteOptions {
            dialect: Dialect::X3dClassic,
            ..syntax::WriteOptions::default()
        },
    )
}
