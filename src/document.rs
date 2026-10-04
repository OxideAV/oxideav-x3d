//! Encoding-independent X3D document model.
//!
//! Both the XML and the ClassicVRML readers produce an [`X3dDocument`];
//! both writers consume one. Nodes live in an arena
//! ([`X3dDocument::nodes`]) addressed by [`NodeIdx`], so `DEF`/`USE`
//! sharing is represented by two parents referencing the same index —
//! the graph is a DAG in well-formed input (a `USE` of an enclosing
//! `DEF` would make it cyclic; every consumer in this crate guards
//! against that).
//!
//! Unknown node types are preserved: their attributes are kept as
//! `SFString` values and their child nodes under the `containerField`
//! names they were given, so a decode → encode round trip keeps them.

use crate::field::{AccessType, FieldType, FieldValue};
use crate::nodes::{self, NodeDef};

/// Index into [`X3dDocument::nodes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeIdx(pub u32);

impl NodeIdx {
    /// Arena slot as `usize`.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// What kind of node an [`X3dNode`] is.
#[derive(Clone, Debug, PartialEq)]
pub enum NodeKind {
    /// A node type from the X3D node table.
    Builtin,
    /// An instance of a prototype. `proto` indexes
    /// [`X3dDocument::protos`] when the declaration was found in scope.
    ProtoInstance {
        /// Index of the resolved `ProtoDeclare` / `ExternProtoDeclare`.
        proto: Option<usize>,
    },
    /// Node type unknown to the table and not a declared prototype.
    Unknown,
}

/// `IS` connection between a node field inside a prototype body and a
/// field of the prototype interface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IsConnect {
    /// Field of the node inside the body.
    pub node_field: String,
    /// Field of the enclosing prototype's interface.
    pub proto_field: String,
}

/// User-defined field declaration (prototype interfaces, `Script`,
/// shader nodes).
#[derive(Clone, Debug, PartialEq)]
pub struct FieldDecl {
    /// Field name.
    pub name: String,
    /// Field type.
    pub ty: FieldType,
    /// Access type.
    pub access: AccessType,
    /// Initial value (`None` for `inputOnly` / `outputOnly` and extern
    /// prototype interfaces).
    pub value: Option<FieldValue>,
}

/// One node instance.
#[derive(Clone, Debug, PartialEq)]
pub struct X3dNode {
    /// Node type name (prototype name for prototype instances).
    pub type_name: String,
    /// Built-in, prototype instance or unknown.
    pub kind: NodeKind,
    /// `DEF` name.
    pub def: Option<String>,
    /// Explicitly specified field values in source order. Node-valued
    /// fields hold [`FieldData::Node`](crate::field::FieldData::Node).
    pub fields: Vec<(String, FieldValue)>,
    /// `IS` connections (only meaningful inside prototype bodies).
    pub is_connects: Vec<IsConnect>,
    /// User-defined fields of `Script` / shader nodes.
    pub decls: Vec<FieldDecl>,
    /// Inline source text (`Script` / shader `CDATA`).
    pub source_text: Option<String>,
    /// Explicit `containerField` given in the XML encoding.
    pub container_field: Option<String>,
    /// 1-based source line (0 when synthesised).
    pub line: usize,
}

impl X3dNode {
    /// Fresh node of type `type_name` with no fields set.
    pub fn new(type_name: impl Into<String>) -> Self {
        let type_name = type_name.into();
        let kind = if nodes::lookup(&type_name).is_some() {
            NodeKind::Builtin
        } else {
            NodeKind::Unknown
        };
        Self {
            type_name,
            kind,
            def: None,
            fields: Vec::new(),
            is_connects: Vec::new(),
            decls: Vec::new(),
            source_text: None,
            container_field: None,
            line: 0,
        }
    }

    /// Node-table entry for built-in nodes.
    pub fn def_table(&self) -> Option<&'static NodeDef> {
        match self.kind {
            NodeKind::Builtin => nodes::lookup(&self.type_name),
            _ => None,
        }
    }

    /// Explicitly set value of `name`.
    pub fn get(&self, name: &str) -> Option<&FieldValue> {
        self.fields.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }

    /// Set (replace) a field value.
    pub fn set(&mut self, name: impl Into<String>, value: FieldValue) {
        let name = name.into();
        if let Some(slot) = self.fields.iter_mut().find(|(n, _)| *n == name) {
            slot.1 = value;
        } else {
            self.fields.push((name, value));
        }
    }

    /// Value of `name`: the explicit value, else the node-table default,
    /// else the declared default of a user-defined field.
    pub fn value(&self, name: &str) -> Option<FieldValue> {
        if let Some(v) = self.get(name) {
            return Some(v.clone());
        }
        if let Some(t) = self.def_table() {
            if let Some(f) = t.field(name) {
                return Some(f.default_value());
            }
        }
        self.decls
            .iter()
            .find(|d| d.name == name)
            .and_then(|d| d.value.clone())
    }

    /// Node children stored in field `name`.
    pub fn children_of(&self, name: &str) -> &[NodeIdx] {
        self.get(name).map(|v| v.as_nodes()).unwrap_or(&[])
    }

    /// Every node referenced from any field, in field order.
    pub fn all_children(&self) -> impl Iterator<Item = NodeIdx> + '_ {
        self.fields
            .iter()
            .flat_map(|(_, v)| v.as_nodes().iter().copied())
    }
}

/// `ROUTE` statement. Endpoints are resolved to node indices at parse
/// time (DEF names are kept for writing).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Route {
    /// Source node.
    pub from_node: NodeIdx,
    /// Source DEF name.
    pub from_def: String,
    /// Source field (output event).
    pub from_field: String,
    /// Destination node.
    pub to_node: NodeIdx,
    /// Destination DEF name.
    pub to_def: String,
    /// Destination field (input event).
    pub to_field: String,
}

/// `IMPORT` statement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Import {
    /// DEF name of the `Inline` node.
    pub inline_def: String,
    /// Name exported from the inlined scene.
    pub imported_def: String,
    /// Local alias (`AS`).
    pub as_name: Option<String>,
}

/// `EXPORT` statement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Export {
    /// Local DEF name.
    pub local_def: String,
    /// Exported alias (`AS`).
    pub as_name: Option<String>,
}

/// Body of a scene or a prototype.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SceneBody {
    /// Top-level nodes in order.
    pub roots: Vec<NodeIdx>,
    /// Routes declared in this body.
    pub routes: Vec<Route>,
    /// Indices into [`X3dDocument::protos`] declared in this body.
    pub protos: Vec<usize>,
    /// `IMPORT` statements.
    pub imports: Vec<Import>,
    /// `EXPORT` statements.
    pub exports: Vec<Export>,
}

/// `ProtoDeclare` or `ExternProtoDeclare`.
#[derive(Clone, Debug, PartialEq)]
pub struct ProtoDecl {
    /// Prototype name.
    pub name: String,
    /// Interface declarations.
    pub interface: Vec<FieldDecl>,
    /// Body (`None` for `ExternProtoDeclare`).
    pub body: Option<SceneBody>,
    /// `url` of an `ExternProtoDeclare`.
    pub url: Vec<String>,
    /// `appinfo` / `documentation` attributes when present.
    pub appinfo: Option<String>,
}

/// `component` statement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentDecl {
    /// Component name.
    pub name: String,
    /// Support level.
    pub level: u32,
}

/// `unit` statement (ISO/IEC 19775-1 4.3.6).
#[derive(Clone, Debug, PartialEq)]
pub struct UnitDecl {
    /// `angle`, `force`, `length` or `mass`.
    pub category: String,
    /// Unit name.
    pub name: String,
    /// Multiplier to the base unit (radian, newton, metre, kilogram).
    pub conversion_factor: f64,
}

/// Which encoding a document was read from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Encoding {
    /// ISO/IEC 19776-1 XML encoding (`.x3d`).
    #[default]
    Xml,
    /// ISO/IEC 19776-2 ClassicVRML encoding (`.x3dv`).
    ClassicVrml,
}

/// A whole X3D document.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct X3dDocument {
    /// Encoding the document came from.
    pub encoding: Encoding,
    /// `version` (e.g. `"4.0"`).
    pub version: String,
    /// `profile` (e.g. `"Interchange"`).
    pub profile: String,
    /// `component` statements.
    pub components: Vec<ComponentDecl>,
    /// `unit` statements.
    pub units: Vec<UnitDecl>,
    /// `meta` statements (`name`, `content`).
    pub meta: Vec<(String, String)>,
    /// Node arena.
    pub nodes: Vec<X3dNode>,
    /// Every prototype declaration (scene-level and nested).
    pub protos: Vec<ProtoDecl>,
    /// The scene.
    pub scene: SceneBody,
    /// Non-fatal problems found while reading (unresolved `USE`,
    /// unparsable values that were skipped, ...), capped in length.
    pub warnings: Vec<String>,
}

/// Maximum number of warnings kept on a document.
pub const MAX_WARNINGS: usize = 256;

impl X3dDocument {
    /// Empty X3D 4.0 Interchange-profile document.
    pub fn new() -> Self {
        Self {
            version: "4.0".into(),
            profile: "Interchange".into(),
            ..Self::default()
        }
    }

    /// Borrow a node.
    pub fn node(&self, idx: NodeIdx) -> Option<&X3dNode> {
        self.nodes.get(idx.index())
    }

    /// Mutably borrow a node.
    pub fn node_mut(&mut self, idx: NodeIdx) -> Option<&mut X3dNode> {
        self.nodes.get_mut(idx.index())
    }

    /// Push a node and return its index.
    pub fn add_node(&mut self, node: X3dNode) -> NodeIdx {
        let i = NodeIdx(self.nodes.len() as u32);
        self.nodes.push(node);
        i
    }

    /// Record a non-fatal warning (bounded).
    pub fn warn(&mut self, msg: impl Into<String>) {
        if self.warnings.len() < MAX_WARNINGS {
            self.warnings.push(msg.into());
        }
    }

    /// Find a node by `DEF` name anywhere in the arena (first match;
    /// prototype bodies have their own DEF scopes, so names may repeat).
    pub fn find_def(&self, name: &str) -> Option<NodeIdx> {
        self.nodes
            .iter()
            .position(|n| n.def.as_deref() == Some(name))
            .map(|i| NodeIdx(i as u32))
    }

    /// Conversion factor declared for a unit `category` (1.0 when
    /// absent).
    pub fn unit_factor(&self, category: &str) -> f64 {
        self.units
            .iter()
            .rev()
            .find(|u| u.category == category)
            .map(|u| u.conversion_factor)
            .filter(|f| f.is_finite() && *f > 0.0)
            .unwrap_or(1.0)
    }
}
