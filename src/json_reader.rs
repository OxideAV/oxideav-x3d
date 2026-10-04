//! JSON encoding reader (ISO/IEC 19776-5 draft, `.x3dj` / `.json`).
//!
//! The X3D JSON encoding is a direct transliteration of the XML
//! encoding: `{"X3D": {...}}` at the root, `"@name"` members for
//! attributes (typed JSON values), `"-containerField"` members for
//! child nodes (an object for SFNode, an array for MFNode), statement
//! and helper elements (`head`, `meta`, `ProtoDeclare`, `field`,
//! `fieldValue`, `IS`, `connect`, `ROUTE`, ...) as plain members, and
//! `"#sourceCode"` for Script / shader text. This module maps the JSON
//! tree onto the [`xml::Element`](crate::xml::Element) tree and reuses
//! the XML reader, so both encodings share every rule.

use serde_json::Value;

use crate::document::{Encoding, X3dDocument};
use crate::error::{Error, Result};
use crate::xml::{Element, XmlNode};
use crate::Limits;

/// Parse JSON-encoded X3D text.
pub fn read_json(src: &str, limits: &Limits) -> Result<X3dDocument> {
    if src.len() > limits.max_input_bytes {
        return Err(Error::limit("input larger than max_input_bytes"));
    }
    // serde_json bounds its own recursion (128 levels).
    let v: Value = serde_json::from_str(src).map_err(|e| Error::invalid(format!("JSON: {e}")))?;
    let root = v
        .get("X3D")
        .and_then(Value::as_object)
        .ok_or_else(|| Error::invalid("JSON: no top-level \"X3D\" object"))?;
    let mut count = 0u64;
    let el = element("X3D", root, None, limits, &mut count, 0)?;
    let mut doc = crate::xml_reader::read_tree(&el, limits)?;
    doc.encoding = Encoding::Json;
    Ok(doc)
}

/// JSON value → XML attribute text.
fn attr_text(v: &Value) -> String {
    match v {
        Value::Null => "NULL".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        Value::Array(a) => {
            if !a.is_empty() && a.iter().all(Value::is_string) {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(crate::field::quote_string)
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                a.iter().map(attr_text).collect::<Vec<_>>().join(" ")
            }
        }
        Value::Object(_) => String::new(),
    }
}

fn element(
    name: &str,
    obj: &serde_json::Map<String, Value>,
    container: Option<&str>,
    limits: &Limits,
    count: &mut u64,
    depth: usize,
) -> Result<Element> {
    *count += 1;
    if *count > limits.max_elements {
        return Err(Error::limit("too many JSON elements"));
    }
    if depth > limits.max_depth {
        return Err(Error::limit("JSON nesting deeper than max_depth"));
    }
    let mut el = Element {
        name: name.to_string(),
        ..Element::default()
    };
    if let Some(c) = container {
        el.attrs.push(("containerField".into(), c.to_string()));
    }
    for (k, v) in obj {
        if let Some(a) = k.strip_prefix('@') {
            el.attrs.push((a.to_string(), attr_text(v)));
        } else if let Some(cf) = k.strip_prefix('-') {
            let items: Vec<&Value> = match v {
                Value::Array(a) => a.iter().collect(),
                other => vec![other],
            };
            for item in items {
                let Some(o) = item.as_object() else { continue };
                for (ty, body) in o {
                    if ty.starts_with('#') {
                        continue;
                    }
                    let Some(b) = body.as_object() else { continue };
                    // Statements inside child lists keep no container.
                    let cfo = if matches!(
                        ty.as_str(),
                        "ROUTE" | "ProtoDeclare" | "ExternProtoDeclare" | "IMPORT" | "EXPORT"
                    ) {
                        None
                    } else {
                        Some(cf)
                    };
                    el.children.push(XmlNode::Element(element(
                        ty,
                        b,
                        cfo,
                        limits,
                        count,
                        depth + 1,
                    )?));
                }
            }
        } else if k == "#sourceCode" {
            let text = match v {
                Value::Array(a) => a
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join("\n"),
                other => other.as_str().unwrap_or("").to_string(),
            };
            el.children.push(XmlNode::CData(text));
        } else if k.starts_with('#') || k == "encoding" || k == "JSON schema" {
            continue;
        } else {
            let items: Vec<&Value> = match v {
                Value::Array(a) => a.iter().collect(),
                other => vec![other],
            };
            for item in items {
                if let Some(o) = item.as_object() {
                    el.children.push(XmlNode::Element(element(
                        k,
                        o,
                        None,
                        limits,
                        count,
                        depth + 1,
                    )?));
                }
            }
        }
    }
    Ok(el)
}
