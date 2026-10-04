//! Minimal, hostile-input-hardened XML 1.0 reader and writer helpers.
//!
//! The X3D XML encoding (ISO/IEC 19776-1) only needs the core XML
//! surface: elements, attributes, character data, CDATA sections
//! (Script / shader source), comments, processing instructions and a
//! `<!DOCTYPE>` declaration that is skipped (internal subsets
//! included). External or DTD-declared entities are **never**
//! expanded — only the five predefined entities and numeric character
//! references are decoded — so entity-expansion bombs are impossible.
//!
//! The tree is built iteratively (explicit stack), so element nesting
//! depth costs heap, not native stack, and is additionally capped by
//! [`Limits::max_depth`](crate::Limits::max_depth).

use crate::error::{Error, Result};
use crate::Limits;

/// One element of a parsed XML document.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Element {
    /// Tag name as written (prefix included, e.g. `x3d:Shape`).
    pub name: String,
    /// `(name, value)` attribute pairs in source order, entity-decoded.
    pub attrs: Vec<(String, String)>,
    /// Child content in source order.
    pub children: Vec<XmlNode>,
    /// 1-based source line of the start tag.
    pub line: usize,
    /// 1-based source column of the start tag.
    pub column: usize,
}

impl Element {
    /// Value of attribute `name`, if present.
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    /// Local part of the tag name (text after the last `:`).
    pub fn local_name(&self) -> &str {
        local(&self.name)
    }

    /// Iterator over the element children (text skipped).
    pub fn elements(&self) -> impl Iterator<Item = &Element> {
        self.children.iter().filter_map(|c| match c {
            XmlNode::Element(e) => Some(e),
            _ => None,
        })
    }

    /// Concatenated character data (text + CDATA) directly inside this
    /// element.
    pub fn text(&self) -> String {
        let mut s = String::new();
        for c in &self.children {
            if let XmlNode::Text(t) | XmlNode::CData(t) = c {
                s.push_str(t);
            }
        }
        s
    }
}

/// Local part of a possibly prefixed XML name.
pub fn local(name: &str) -> &str {
    name.rsplit(':').next().unwrap_or(name)
}

/// Content node inside an [`Element`].
#[derive(Clone, Debug, PartialEq)]
pub enum XmlNode {
    /// Child element.
    Element(Element),
    /// Character data (entity-decoded).
    Text(String),
    /// `<![CDATA[ ... ]]>` section, verbatim.
    CData(String),
}

struct Cursor<'a> {
    s: &'a str,
    b: &'a [u8],
    pos: usize,
    line: usize,
    line_start: usize,
}

impl<'a> Cursor<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            s,
            b: s.as_bytes(),
            pos: 0,
            line: 1,
            line_start: 0,
        }
    }

    fn eof(&self) -> bool {
        self.pos >= self.b.len()
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.pos).copied()
    }

    fn starts_with(&self, pat: &str) -> bool {
        self.b[self.pos..].starts_with(pat.as_bytes())
    }

    fn column(&self) -> usize {
        // Count characters, not bytes, for a friendlier column.
        self.s
            .get(self.line_start..self.pos)
            .map(|t| t.chars().count())
            .unwrap_or(0)
            + 1
    }

    fn err(&self, msg: impl Into<String>) -> Error {
        Error::Syntax {
            line: self.line,
            column: self.column(),
            message: msg.into(),
        }
    }

    /// Advance `n` bytes, tracking line numbers. `n` must land on a
    /// char boundary (callers only skip ASCII or whole matches).
    fn advance(&mut self, n: usize) {
        let end = (self.pos + n).min(self.b.len());
        for i in self.pos..end {
            if self.b[i] == b'\n' {
                self.line += 1;
                self.line_start = i + 1;
            }
        }
        self.pos = end;
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if matches!(c, b' ' | b'\t' | b'\r' | b'\n') {
                self.advance(1);
            } else {
                break;
            }
        }
    }

    /// Advance past the next occurrence of `pat`, returning the text
    /// before it.
    fn take_until(&mut self, pat: &str, what: &str) -> Result<&'a str> {
        let hay = &self.s[self.pos..];
        match hay.find(pat) {
            Some(i) => {
                let out = &hay[..i];
                self.advance(i + pat.len());
                Ok(out)
            }
            None => Err(self.err(format!("unterminated {what}"))),
        }
    }

    fn name(&mut self) -> Result<&'a str> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || matches!(c, b'_' | b':' | b'-' | b'.') || c >= 0x80 {
                self.advance(1);
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(self.err("expected a name"));
        }
        // Multi-byte UTF-8 continuation bytes are all >= 0x80, so the
        // loop never stops mid-character.
        Ok(&self.s[start..self.pos])
    }
}

/// Decode the predefined entities and numeric character references.
/// Unknown entity references are kept verbatim.
pub fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        let window = rest
            .char_indices()
            .nth(16)
            .map(|(i, _)| i)
            .unwrap_or(rest.len());
        let semi = rest[..window].find(';');
        let Some(semi) = semi else {
            out.push('&');
            rest = &rest[1..];
            continue;
        };
        let ent = &rest[1..semi];
        let rep = match ent {
            "lt" => Some('<'),
            "gt" => Some('>'),
            "amp" => Some('&'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            _ => {
                let num = if let Some(h) = ent.strip_prefix("#x").or(ent.strip_prefix("#X")) {
                    u32::from_str_radix(h, 16).ok()
                } else if let Some(d) = ent.strip_prefix('#') {
                    d.parse::<u32>().ok()
                } else {
                    None
                };
                num.and_then(char::from_u32)
            }
        };
        match rep {
            Some(c) => {
                out.push(c);
                rest = &rest[semi + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// Escape text for use inside a double-quoted attribute value.
pub fn escape_attr(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\n' => out.push_str("&#10;"),
            '\r' => out.push_str("&#13;"),
            '\t' => out.push_str("&#9;"),
            c => out.push(c),
        }
    }
    out
}

/// Append ` name="value"` to `out`, picking the quote character that
/// needs no escaping when possible (MFString values contain `"`).
pub fn write_attr(out: &mut String, name: &str, value: &str) {
    out.push(' ');
    out.push_str(name);
    if value.contains('"') && !value.contains('\'') {
        out.push_str("='");
        for c in value.chars() {
            match c {
                '&' => out.push_str("&amp;"),
                '<' => out.push_str("&lt;"),
                '>' => out.push_str("&gt;"),
                '\n' => out.push_str("&#10;"),
                '\r' => out.push_str("&#13;"),
                '\t' => out.push_str("&#9;"),
                c => out.push(c),
            }
        }
        out.push('\'');
    } else {
        out.push_str("=\"");
        out.push_str(&escape_attr(value));
        out.push('"');
    }
}

/// Parse an XML document and return its root element.
///
/// Leading prolog (XML declaration, comments, PIs, DOCTYPE) and
/// trailing misc content are skipped; exactly one root element is
/// required.
pub fn parse(src: &str, limits: &Limits) -> Result<Element> {
    let src = src.strip_prefix('\u{feff}').unwrap_or(src);
    let mut cur = Cursor::new(src);
    let mut stack: Vec<Element> = Vec::new();
    let mut root: Option<Element> = None;
    let mut elements: u64 = 0;

    while !cur.eof() {
        if cur.peek() != Some(b'<') {
            // Character data.
            let hay = &cur.s[cur.pos..];
            let n = hay.find('<').unwrap_or(hay.len());
            let raw = &hay[..n];
            let line = cur.line;
            let col = cur.column();
            cur.advance(n);
            match stack.last_mut() {
                Some(top) => {
                    if !raw.trim().is_empty() || !top.children.is_empty() {
                        top.children.push(XmlNode::Text(decode_entities(raw)));
                    }
                }
                None => {
                    if !raw.trim().is_empty() {
                        return Err(Error::Syntax {
                            line,
                            column: col,
                            message: "character data outside the root element".into(),
                        });
                    }
                }
            }
            continue;
        }
        if cur.starts_with("<!--") {
            cur.advance(4);
            cur.take_until("-->", "comment")?;
            continue;
        }
        if cur.starts_with("<![CDATA[") {
            cur.advance(9);
            let t = cur.take_until("]]>", "CDATA section")?;
            match stack.last_mut() {
                Some(top) => top.children.push(XmlNode::CData(t.to_string())),
                None => return Err(cur.err("CDATA outside the root element")),
            }
            continue;
        }
        if cur.starts_with("<?") {
            cur.advance(2);
            cur.take_until("?>", "processing instruction")?;
            continue;
        }
        if cur.starts_with("<!") {
            skip_markup_decl(&mut cur)?;
            continue;
        }
        if cur.starts_with("</") {
            cur.advance(2);
            let name = cur.name()?;
            cur.skip_ws();
            if cur.peek() != Some(b'>') {
                return Err(cur.err("expected '>' in end tag"));
            }
            cur.advance(1);
            let Some(el) = stack.pop() else {
                return Err(cur.err(format!("unexpected end tag </{name}>")));
            };
            if el.name != name {
                return Err(cur.err(format!(
                    "end tag </{name}> does not match start tag <{}>",
                    el.name
                )));
            }
            finish(el, &mut stack, &mut root, &cur)?;
            continue;
        }
        // Start tag.
        let line = cur.line;
        let column = cur.column();
        cur.advance(1);
        let name = cur.name()?.to_string();
        elements += 1;
        if elements > limits.max_elements {
            return Err(Error::limit(format!(
                "more than {} XML elements",
                limits.max_elements
            )));
        }
        let mut el = Element {
            name,
            attrs: Vec::new(),
            children: Vec::new(),
            line,
            column,
        };
        let self_closing;
        loop {
            let had_ws = matches!(cur.peek(), Some(b' ' | b'\t' | b'\r' | b'\n'));
            cur.skip_ws();
            match cur.peek() {
                None => return Err(cur.err("unterminated start tag")),
                Some(b'>') => {
                    cur.advance(1);
                    self_closing = false;
                    break;
                }
                Some(b'/') => {
                    cur.advance(1);
                    if cur.peek() != Some(b'>') {
                        return Err(cur.err("expected '>' after '/'"));
                    }
                    cur.advance(1);
                    self_closing = true;
                    break;
                }
                Some(_) => {
                    if !had_ws {
                        return Err(cur.err("expected whitespace before attribute"));
                    }
                    let an = cur.name()?.to_string();
                    cur.skip_ws();
                    if cur.peek() != Some(b'=') {
                        return Err(cur.err(format!("expected '=' after attribute {an}")));
                    }
                    cur.advance(1);
                    cur.skip_ws();
                    let q = match cur.peek() {
                        Some(q @ (b'"' | b'\'')) => q,
                        _ => return Err(cur.err("expected quoted attribute value")),
                    };
                    cur.advance(1);
                    let pat = if q == b'"' { "\"" } else { "'" };
                    let raw = cur.take_until(pat, "attribute value")?;
                    if raw.contains('<') {
                        return Err(cur.err("'<' inside attribute value"));
                    }
                    if el.attrs.len() >= limits.max_attributes {
                        return Err(Error::limit("too many attributes on one element"));
                    }
                    // Attribute-value normalisation (XML 1.0 §3.3.3):
                    // literal whitespace characters become spaces.
                    let norm: String = raw
                        .chars()
                        .map(|c| {
                            if matches!(c, '\t' | '\n' | '\r') {
                                ' '
                            } else {
                                c
                            }
                        })
                        .collect();
                    el.attrs.push((an, decode_entities(&norm)));
                }
            }
        }
        if self_closing {
            finish(el, &mut stack, &mut root, &cur)?;
        } else {
            if stack.len() >= limits.max_depth {
                return Err(Error::limit(format!(
                    "XML nesting deeper than {}",
                    limits.max_depth
                )));
            }
            stack.push(el);
        }
    }
    if let Some(open) = stack.last() {
        return Err(Error::Syntax {
            line: open.line,
            column: open.column,
            message: format!("element <{}> is never closed", open.name),
        });
    }
    root.ok_or_else(|| Error::invalid("no root element"))
}

fn finish(
    el: Element,
    stack: &mut [Element],
    root: &mut Option<Element>,
    cur: &Cursor<'_>,
) -> Result<()> {
    match stack.last_mut() {
        Some(parent) => {
            parent.children.push(XmlNode::Element(el));
            Ok(())
        }
        None => {
            if root.is_some() {
                return Err(cur.err("more than one root element"));
            }
            *root = Some(el);
            Ok(())
        }
    }
}

/// Skip `<!DOCTYPE ...>` (with an optional `[...]` internal subset) or
/// any other `<!...>` markup declaration.
fn skip_markup_decl(cur: &mut Cursor<'_>) -> Result<()> {
    cur.advance(2);
    let mut bracket = 0usize;
    let mut quote: Option<u8> = None;
    while let Some(c) = cur.peek() {
        if let Some(q) = quote {
            if c == q {
                quote = None;
            }
        } else if c == b'"' || c == b'\'' {
            quote = Some(c);
        } else if c == b'[' {
            bracket += 1;
        } else if c == b']' {
            bracket = bracket.saturating_sub(1);
        } else if c == b'>' && bracket == 0 {
            cur.advance(1);
            return Ok(());
        } else if c == b'<' && cur.starts_with("<!--") {
            cur.advance(4);
            cur.take_until("-->", "comment")?;
            continue;
        }
        cur.advance(1);
    }
    Err(cur.err("unterminated markup declaration"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Result<Element> {
        parse(s, &Limits::default())
    }

    #[test]
    fn basic_tree() {
        let e = p("<?xml version='1.0'?>\n<!DOCTYPE X3D PUBLIC \"a\" \"b\" [ <!ENTITY x 'y'> ]>\n<a x=\"1\" y='&lt;2&#x41;'><b/>t<![CDATA[ <s> ]]></a>").unwrap();
        assert_eq!(e.name, "a");
        assert_eq!(e.attr("y"), Some("<2A"));
        assert_eq!(e.elements().count(), 1);
        assert!(e.text().contains("<s>"));
    }

    #[test]
    fn errors() {
        assert!(p("<a><b></a>").is_err());
        assert!(p("<a>").is_err());
        assert!(p("<a x=1/>").is_err());
        assert!(p("<a/><b/>").is_err());
        assert!(p("").is_err());
        assert!(p("<a x='<'/>").is_err());
    }

    #[test]
    fn depth_limit() {
        let s = "<a>".repeat(10_000);
        let lim = Limits {
            max_depth: 64,
            ..Limits::default()
        };
        assert!(matches!(parse(&s, &lim), Err(Error::LimitExceeded(_))));
    }

    #[test]
    fn entities() {
        assert_eq!(decode_entities("a&amp;b&unknown;c&#65;"), "a&b&unknown;cA");
        assert_eq!(decode_entities("&"), "&");
        assert_eq!(decode_entities("&#xFFFFFFFF;"), "&#xFFFFFFFF;");
    }
}
