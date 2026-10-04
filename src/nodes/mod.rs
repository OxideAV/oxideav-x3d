//! Node-type table: every concrete X3D 4.0 node with its component,
//! default `containerField` and field interface (name, type, access
//! type, default value).
//!
//! The table in `generated.rs` is generated from the Web3D X3D Unified
//! Object Model (`X3dUnifiedObjectModel-4.0.xml`, mirrored in the
//! OxideAV docs repository under `3d/x3d/schema/`). Defaults are kept
//! as their XML-encoding attribute text and parsed on demand with
//! [`parse_xml_value`](crate::field::parse_xml_value).

#[rustfmt::skip]
mod generated;

use crate::field::{parse_xml_value, FieldValue};
pub use crate::field::{AccessType, FieldType};

/// One field of a node interface.
#[derive(Clone, Copy, Debug)]
pub struct FieldDef {
    /// Field name.
    pub name: &'static str,
    /// Field type.
    pub ty: FieldType,
    /// Access type.
    pub access: AccessType,
    /// Default value in XML attribute syntax (`None` when the X3DUOM
    /// lists no default, e.g. `inputOnly` events or empty MF fields).
    pub default: Option<&'static str>,
}

impl FieldDef {
    /// Parsed default value (empty value when none is listed or the
    /// listed text does not parse).
    pub fn default_value(&self) -> FieldValue {
        match self.default {
            Some(d) if d != "NULL" => {
                parse_xml_value(self.ty, d).unwrap_or_else(|_| FieldValue::empty(self.ty))
            }
            _ => FieldValue::empty(self.ty),
        }
    }
}

/// One concrete node type.
#[derive(Clone, Copy, Debug)]
pub struct NodeDef {
    /// Node type name.
    pub name: &'static str,
    /// Component the node belongs to.
    pub component: &'static str,
    /// Component level that introduces the node.
    pub level: u32,
    /// Default `containerField` of the XML encoding.
    pub container_field: &'static str,
    /// Field interface sorted by name.
    pub fields: &'static [FieldDef],
}

impl NodeDef {
    /// Look up a field by name (also resolving the `set_x` / `x_changed`
    /// event aliases of `inputOutput` fields).
    pub fn field(&self, name: &str) -> Option<&'static FieldDef> {
        if let Ok(i) = self.fields.binary_search_by(|f| f.name.cmp(name)) {
            return Some(&self.fields[i]);
        }
        let base = name
            .strip_prefix("set_")
            .or_else(|| name.strip_suffix("_changed"))?;
        let f = self.field_exact(base)?;
        (f.access == AccessType::InputOutput).then_some(f)
    }

    fn field_exact(&self, name: &str) -> Option<&'static FieldDef> {
        self.fields
            .binary_search_by(|f| f.name.cmp(name))
            .ok()
            .map(|i| &self.fields[i])
    }
}

/// All node types, sorted by name.
pub fn all() -> &'static [NodeDef] {
    generated::NODES
}

/// Look up a node type by name.
pub fn lookup(name: &str) -> Option<&'static NodeDef> {
    let n = generated::NODES;
    n.binary_search_by(|d| d.name.cmp(name)).ok().map(|i| &n[i])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_sorted_and_findable() {
        let n = all();
        assert!(n.len() > 250);
        for w in n.windows(2) {
            assert!(w[0].name < w[1].name);
        }
        for d in n {
            for w in d.fields.windows(2) {
                assert!(w[0].name < w[1].name, "{} fields unsorted", d.name);
            }
        }
        let t = lookup("Transform").unwrap();
        assert_eq!(t.component, "Grouping");
        let s = t.field("scale").unwrap();
        assert_eq!(s.default_value().as_tuple::<3>(), Some([1.0, 1.0, 1.0]));
        assert!(t.field("set_translation").is_some());
        assert!(t.field("translation_changed").is_some());
        assert_eq!(lookup("Material").unwrap().container_field, "material");
        assert_eq!(lookup("Box").unwrap().container_field, "geometry");
    }

    #[test]
    fn every_default_parses() {
        for d in all() {
            for f in d.fields {
                if let Some(text) = f.default {
                    if text == "NULL" {
                        continue;
                    }
                    assert!(
                        parse_xml_value(f.ty, text).is_ok(),
                        "{}.{} default {:?} does not parse as {:?}",
                        d.name,
                        f.name,
                        text,
                        f.ty
                    );
                }
            }
        }
    }
}
