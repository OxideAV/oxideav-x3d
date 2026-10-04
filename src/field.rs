//! X3D field types (ISO/IEC 19775-1 clause 5) and typed field values.
//!
//! [`FieldType`] enumerates the 42 SF/MF types. A [`FieldValue`]
//! pairs a type with its payload ([`FieldData`]), stored in the
//! narrowest uniform shape: booleans, `i32`s, `f32`s (single-precision
//! vectors, colours, rotations, matrices), `f64`s (doubles, times,
//! double-precision vectors/matrices), strings, images, or node
//! references. Tuple types are stored flattened (an `MFVec3f` with two
//! values holds six floats); [`FieldType::arity`] gives the tuple
//! width.
//!
//! [`parse_xml_value`] implements the attribute-value syntax of the XML
//! encoding (ISO/IEC 19776-1 clause 5) and [`format_xml_value`] its
//! inverse.

use crate::document::NodeIdx;

/// Access type of a field (ISO/IEC 19775-1 4.4.2.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AccessType {
    /// `initializeOnly` (VRML97 `field`).
    InitializeOnly,
    /// `inputOnly` (VRML97 `eventIn`).
    InputOnly,
    /// `outputOnly` (VRML97 `eventOut`).
    OutputOnly,
    /// `inputOutput` (VRML97 `exposedField`).
    InputOutput,
}

impl AccessType {
    /// Parse the XML-encoding spelling (`initializeOnly`, ...) or the
    /// ClassicVRML / VRML97 keywords (`field`, `eventIn`, ...).
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "initializeOnly" | "field" => Self::InitializeOnly,
            "inputOnly" | "eventIn" => Self::InputOnly,
            "outputOnly" | "eventOut" => Self::OutputOnly,
            "inputOutput" | "exposedField" => Self::InputOutput,
            _ => return None,
        })
    }

    /// XML-encoding spelling.
    pub fn name(self) -> &'static str {
        match self {
            Self::InitializeOnly => "initializeOnly",
            Self::InputOnly => "inputOnly",
            Self::OutputOnly => "outputOnly",
            Self::InputOutput => "inputOutput",
        }
    }
}

macro_rules! field_types {
    ($( $v:ident ),* $(,)?) => {
        /// The X3D field data types (ISO/IEC 19775-1 clause 5).
        #[allow(missing_docs)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum FieldType { $( $v ),* }

        impl FieldType {
            /// Every field type.
            pub const ALL: &'static [FieldType] = &[ $( FieldType::$v ),* ];

            /// Canonical type name (`"SFVec3f"`).
            pub fn name(self) -> &'static str {
                match self { $( FieldType::$v => stringify!($v) ),* }
            }

            /// Look a type up by its canonical name.
            pub fn from_name(s: &str) -> Option<Self> {
                match s { $( stringify!($v) => Some(FieldType::$v), )* _ => None }
            }
        }
    };
}

field_types!(
    SFBool,
    MFBool,
    SFColor,
    MFColor,
    SFColorRGBA,
    MFColorRGBA,
    SFDouble,
    MFDouble,
    SFFloat,
    MFFloat,
    SFImage,
    MFImage,
    SFInt32,
    MFInt32,
    SFMatrix3d,
    MFMatrix3d,
    SFMatrix3f,
    MFMatrix3f,
    SFMatrix4d,
    MFMatrix4d,
    SFMatrix4f,
    MFMatrix4f,
    SFNode,
    MFNode,
    SFRotation,
    MFRotation,
    SFString,
    MFString,
    SFTime,
    MFTime,
    SFVec2d,
    MFVec2d,
    SFVec2f,
    MFVec2f,
    SFVec3d,
    MFVec3d,
    SFVec3f,
    MFVec3f,
    SFVec4d,
    MFVec4d,
    SFVec4f,
    MFVec4f,
);

/// Storage class of a field type's payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage {
    /// [`FieldData::Bool`].
    Bool,
    /// [`FieldData::Int32`].
    Int32,
    /// [`FieldData::Float`].
    Float,
    /// [`FieldData::Double`].
    Double,
    /// [`FieldData::String`].
    String,
    /// [`FieldData::Image`].
    Image,
    /// [`FieldData::Node`].
    Node,
}

impl FieldType {
    /// `true` for the multiple-valued (`MF*`) types.
    pub fn is_mf(self) -> bool {
        self.name().starts_with("MF")
    }

    /// Single-valued counterpart (`MFVec3f` → `SFVec3f`; SF types map
    /// to themselves).
    pub fn single(self) -> Self {
        if self.is_mf() {
            let n = format!("SF{}", &self.name()[2..]);
            Self::from_name(&n).unwrap_or(self)
        } else {
            self
        }
    }

    /// Multiple-valued counterpart (`SFVec3f` → `MFVec3f`).
    pub fn multi(self) -> Self {
        if self.is_mf() {
            self
        } else {
            let n = format!("MF{}", &self.name()[2..]);
            Self::from_name(&n).unwrap_or(self)
        }
    }

    /// Number of scalar components per single value (3 for `SFVec3f`
    /// / `MFVec3f`, 16 for the 4×4 matrices, 1 for scalars, strings,
    /// images and nodes).
    pub fn arity(self) -> usize {
        use FieldType::*;
        match self.single() {
            SFColor | SFVec3f | SFVec3d => 3,
            SFColorRGBA | SFRotation | SFVec4f | SFVec4d => 4,
            SFVec2f | SFVec2d => 2,
            SFMatrix3f | SFMatrix3d => 9,
            SFMatrix4f | SFMatrix4d => 16,
            _ => 1,
        }
    }

    /// Payload storage class.
    pub fn storage(self) -> Storage {
        use FieldType::*;
        match self.single() {
            SFBool => Storage::Bool,
            SFInt32 => Storage::Int32,
            SFColor | SFColorRGBA | SFFloat | SFMatrix3f | SFMatrix4f | SFRotation | SFVec2f
            | SFVec3f | SFVec4f => Storage::Float,
            SFDouble | SFTime | SFMatrix3d | SFMatrix4d | SFVec2d | SFVec3d | SFVec4d => {
                Storage::Double
            }
            SFString => Storage::String,
            SFImage => Storage::Image,
            _ => Storage::Node,
        }
    }
}

/// An `SFImage` value (ISO/IEC 19775-1 5.3.6): `width × height` pixels
/// of `components` (1–4) bytes each, packed into one `u32` per pixel
/// with the first component in the most significant used byte, stored
/// bottom row first.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SFImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Components per pixel: 1 (grey), 2 (grey+alpha), 3 (RGB) or
    /// 4 (RGBA).
    pub components: u32,
    /// `width × height` packed pixels, lower-left first.
    pub pixels: Vec<u32>,
}

impl SFImage {
    /// Expand to 8-bit RGBA, rows **top to bottom** (image order).
    pub fn to_rgba8_top_down(&self) -> Vec<u8> {
        let w = self.width as usize;
        let h = self.height as usize;
        let mut out = vec![0u8; w * h * 4];
        for y in 0..h {
            let src_row = h - 1 - y;
            for x in 0..w {
                let p = self.pixels.get(src_row * w + x).copied().unwrap_or(0);
                let (r, g, b, a) = match self.components {
                    1 => {
                        let v = (p & 0xff) as u8;
                        (v, v, v, 255)
                    }
                    2 => {
                        let v = ((p >> 8) & 0xff) as u8;
                        (v, v, v, (p & 0xff) as u8)
                    }
                    3 => (
                        ((p >> 16) & 0xff) as u8,
                        ((p >> 8) & 0xff) as u8,
                        (p & 0xff) as u8,
                        255,
                    ),
                    _ => (
                        ((p >> 24) & 0xff) as u8,
                        ((p >> 16) & 0xff) as u8,
                        ((p >> 8) & 0xff) as u8,
                        (p & 0xff) as u8,
                    ),
                };
                let o = (y * w + x) * 4;
                out[o..o + 4].copy_from_slice(&[r, g, b, a]);
            }
        }
        out
    }
}

/// Typed payload of a [`FieldValue`].
#[derive(Clone, Debug, PartialEq)]
pub enum FieldData {
    /// `SFBool` / `MFBool`.
    Bool(Vec<bool>),
    /// `SFInt32` / `MFInt32`.
    Int32(Vec<i32>),
    /// Single-precision types, flattened.
    Float(Vec<f32>),
    /// Double-precision and time types, flattened.
    Double(Vec<f64>),
    /// `SFString` / `MFString`.
    String(Vec<String>),
    /// `SFImage` / `MFImage`.
    Image(Vec<SFImage>),
    /// `SFNode` / `MFNode` (an empty vec is `NULL`).
    Node(Vec<NodeIdx>),
}

/// A typed field value.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldValue {
    /// Declared type.
    pub ty: FieldType,
    /// Payload.
    pub data: FieldData,
}

impl FieldValue {
    /// Empty value of `ty` (empty MF list / `NULL` node / zero SF —
    /// SF scalars hold no components until set).
    pub fn empty(ty: FieldType) -> Self {
        let data = match ty.storage() {
            Storage::Bool => FieldData::Bool(Vec::new()),
            Storage::Int32 => FieldData::Int32(Vec::new()),
            Storage::Float => FieldData::Float(Vec::new()),
            Storage::Double => FieldData::Double(Vec::new()),
            Storage::String => FieldData::String(Vec::new()),
            Storage::Image => FieldData::Image(Vec::new()),
            Storage::Node => FieldData::Node(Vec::new()),
        };
        Self { ty, data }
    }

    /// `SFNode`/`MFNode` value referencing `nodes`.
    pub fn nodes(ty: FieldType, nodes: Vec<NodeIdx>) -> Self {
        Self {
            ty,
            data: FieldData::Node(nodes),
        }
    }

    /// Convenience constructor for an `SFString`.
    pub fn sf_string(s: impl Into<String>) -> Self {
        Self {
            ty: FieldType::SFString,
            data: FieldData::String(vec![s.into()]),
        }
    }

    /// Number of scalar components (or strings / images / nodes).
    pub fn len(&self) -> usize {
        match &self.data {
            FieldData::Bool(v) => v.len(),
            FieldData::Int32(v) => v.len(),
            FieldData::Float(v) => v.len(),
            FieldData::Double(v) => v.len(),
            FieldData::String(v) => v.len(),
            FieldData::Image(v) => v.len(),
            FieldData::Node(v) => v.len(),
        }
    }

    /// `true` when the payload holds nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// First boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match &self.data {
            FieldData::Bool(v) => v.first().copied(),
            _ => None,
        }
    }

    /// All booleans.
    pub fn as_bools(&self) -> &[bool] {
        match &self.data {
            FieldData::Bool(v) => v,
            _ => &[],
        }
    }

    /// All integers.
    pub fn as_i32s(&self) -> &[i32] {
        match &self.data {
            FieldData::Int32(v) => v,
            _ => &[],
        }
    }

    /// First integer.
    pub fn as_i32(&self) -> Option<i32> {
        self.as_i32s().first().copied()
    }

    /// All numeric components as `f32` (floats, doubles and integers).
    pub fn as_f32s(&self) -> Vec<f32> {
        match &self.data {
            FieldData::Float(v) => v.clone(),
            FieldData::Double(v) => v.iter().map(|&d| d as f32).collect(),
            FieldData::Int32(v) => v.iter().map(|&d| d as f32).collect(),
            _ => Vec::new(),
        }
    }

    /// All numeric components as `f64`.
    pub fn as_f64s(&self) -> Vec<f64> {
        match &self.data {
            FieldData::Float(v) => v.iter().map(|&d| d as f64).collect(),
            FieldData::Double(v) => v.clone(),
            FieldData::Int32(v) => v.iter().map(|&d| d as f64).collect(),
            _ => Vec::new(),
        }
    }

    /// First numeric component as `f32`.
    pub fn as_f32(&self) -> Option<f32> {
        match &self.data {
            FieldData::Float(v) => v.first().copied(),
            FieldData::Double(v) => v.first().map(|&d| d as f32),
            FieldData::Int32(v) => v.first().map(|&d| d as f32),
            _ => None,
        }
    }

    /// First numeric component as `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        match &self.data {
            FieldData::Float(v) => v.first().map(|&d| d as f64),
            FieldData::Double(v) => v.first().copied(),
            FieldData::Int32(v) => v.first().map(|&d| d as f64),
            _ => None,
        }
    }

    /// Numeric components grouped into `N`-tuples (incomplete trailing
    /// tuple dropped).
    pub fn as_tuples<const N: usize>(&self) -> Vec<[f32; N]> {
        let f = self.as_f32s();
        f.chunks_exact(N)
            .map(|c| {
                let mut a = [0.0f32; N];
                a.copy_from_slice(c);
                a
            })
            .collect()
    }

    /// First `N`-tuple, if complete.
    pub fn as_tuple<const N: usize>(&self) -> Option<[f32; N]> {
        let f = self.as_f32s();
        if f.len() < N {
            return None;
        }
        let mut a = [0.0f32; N];
        a.copy_from_slice(&f[..N]);
        Some(a)
    }

    /// First string.
    pub fn as_str(&self) -> Option<&str> {
        match &self.data {
            FieldData::String(v) => v.first().map(String::as_str),
            _ => None,
        }
    }

    /// All strings.
    pub fn as_strings(&self) -> &[String] {
        match &self.data {
            FieldData::String(v) => v,
            _ => &[],
        }
    }

    /// Referenced nodes.
    pub fn as_nodes(&self) -> &[NodeIdx] {
        match &self.data {
            FieldData::Node(v) => v,
            _ => &[],
        }
    }

    /// Images.
    pub fn as_images(&self) -> &[SFImage] {
        match &self.data {
            FieldData::Image(v) => v,
            _ => &[],
        }
    }
}

/// Field-value parse failure (the caller adds node/field context).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueError(pub String);

fn is_sep(c: char) -> bool {
    c.is_whitespace() || c == ','
}

/// Parse an integer token: decimal or `0x`-prefixed hexadecimal
/// (wrapped to 32 bits, so `0xFFFFFFFF` is `-1`).
pub fn parse_int_token(t: &str) -> Option<i64> {
    let (neg, body) = match t.strip_prefix('-') {
        Some(b) => (true, b),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    let v = if let Some(h) = body.strip_prefix("0x").or(body.strip_prefix("0X")) {
        i64::from_str_radix(h, 16).ok()?
    } else {
        body.parse::<i64>().ok()?
    };
    Some(if neg { -v } else { v })
}

/// Parse a floating-point token (ISO C syntax; Rust's grammar is a
/// superset that also takes `inf`/`nan`).
pub fn parse_float_token(t: &str) -> Option<f64> {
    if let Ok(v) = t.parse::<f64>() {
        return Some(v);
    }
    // Tolerate integers in hex (seen in hand-written files).
    parse_int_token(t).map(|i| i as f64)
}

/// Parse a bool token (`true`/`false`, case-insensitive so the
/// ClassicVRML `TRUE`/`FALSE` spellings are accepted too).
pub fn parse_bool_token(t: &str) -> Option<bool> {
    if t.eq_ignore_ascii_case("true") {
        Some(true)
    } else if t.eq_ignore_ascii_case("false") {
        Some(false)
    } else {
        None
    }
}

/// Parse the body of an MFString attribute: a whitespace-separated
/// list of double-quoted strings with `\"` / `\\` escapes. A value
/// without any double quote is treated as a single unquoted string
/// (the singleton relaxation of ISO/IEC 19776-1 V4.0 5.15).
pub fn parse_mfstring(s: &str) -> Result<Vec<String>, ValueError> {
    if !s.contains('"') {
        let t = s.trim();
        return Ok(if t.is_empty() {
            Vec::new()
        } else {
            vec![t.to_string()]
        });
    }
    let mut out = Vec::new();
    let mut it = s.chars().peekable();
    loop {
        while matches!(it.peek(), Some(c) if is_sep(*c)) {
            it.next();
        }
        match it.next() {
            None => break,
            Some('"') => {
                let mut cur = String::new();
                let mut closed = false;
                while let Some(c) = it.next() {
                    match c {
                        '\\' => {
                            if let Some(n) = it.next() {
                                cur.push(n);
                            }
                        }
                        '"' => {
                            closed = true;
                            break;
                        }
                        c => cur.push(c),
                    }
                }
                if !closed {
                    return Err(ValueError("unterminated string in MFString".into()));
                }
                out.push(cur);
            }
            Some(c) => {
                return Err(ValueError(format!(
                    "unexpected '{c}' outside quotes in MFString"
                )));
            }
        }
    }
    Ok(out)
}

/// Parse an XML-encoding attribute value as `ty`.
///
/// `SFNode`/`MFNode` values cannot be expressed as attributes (except
/// `NULL`); they parse to an empty node list.
pub fn parse_xml_value(ty: FieldType, s: &str) -> Result<FieldValue, ValueError> {
    let toks = || s.split(is_sep).filter(|t| !t.is_empty());
    let data = match ty.storage() {
        Storage::Bool => {
            let mut v = Vec::new();
            for t in toks() {
                v.push(
                    parse_bool_token(t).ok_or_else(|| ValueError(format!("bad boolean '{t}'")))?,
                );
            }
            FieldData::Bool(v)
        }
        Storage::Int32 => {
            let mut v = Vec::new();
            for t in toks() {
                let i =
                    parse_int_token(t).ok_or_else(|| ValueError(format!("bad integer '{t}'")))?;
                v.push(i as i32);
            }
            FieldData::Int32(v)
        }
        Storage::Float => {
            let mut v = Vec::new();
            for t in toks() {
                let f =
                    parse_float_token(t).ok_or_else(|| ValueError(format!("bad number '{t}'")))?;
                v.push(f as f32);
            }
            FieldData::Float(v)
        }
        Storage::Double => {
            let mut v = Vec::new();
            for t in toks() {
                v.push(
                    parse_float_token(t).ok_or_else(|| ValueError(format!("bad number '{t}'")))?,
                );
            }
            FieldData::Double(v)
        }
        Storage::String => {
            if ty.is_mf() {
                FieldData::String(parse_mfstring(s)?)
            } else {
                FieldData::String(vec![s.to_string()])
            }
        }
        Storage::Image => {
            let mut ints = Vec::new();
            for t in toks() {
                ints.push(
                    parse_int_token(t).ok_or_else(|| ValueError(format!("bad pixel '{t}'")))?,
                );
            }
            FieldData::Image(images_from_ints(&ints, ty.is_mf())?)
        }
        Storage::Node => FieldData::Node(Vec::new()),
    };
    let mut v = FieldValue { ty, data };
    normalise_count(&mut v);
    Ok(v)
}

/// Upper bound on the pixels of one `SFImage` (64 Mi pixels).
pub const MAX_IMAGE_PIXELS: u64 = 1 << 26;

/// Build SFImage values from a flat integer list.
pub fn images_from_ints(ints: &[i64], multi: bool) -> Result<Vec<SFImage>, ValueError> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < ints.len() {
        if i + 3 > ints.len() {
            return Err(ValueError("truncated SFImage header".into()));
        }
        let (w, h, c) = (ints[i], ints[i + 1], ints[i + 2]);
        if !(0..=1 << 20).contains(&w) || !(0..=1 << 20).contains(&h) || !(0..=4).contains(&c) {
            return Err(ValueError(format!("bad SFImage header {w} {h} {c}")));
        }
        let n = (w as u64) * (h as u64);
        if n > MAX_IMAGE_PIXELS {
            return Err(ValueError("SFImage too large".into()));
        }
        i += 3;
        let n = n as usize;
        let avail = (ints.len() - i).min(n);
        let mut pixels: Vec<u32> = ints[i..i + avail].iter().map(|&p| p as u32).collect();
        pixels.resize(n, 0);
        i += avail;
        out.push(SFImage {
            width: w as u32,
            height: h as u32,
            components: c as u32,
            pixels,
        });
        if !multi {
            break;
        }
    }
    if !multi && out.is_empty() {
        out.push(SFImage::default());
    }
    Ok(out)
}

/// Drop an incomplete trailing tuple and clamp SF values to one tuple.
pub fn normalise_count(v: &mut FieldValue) {
    let ar = v.ty.arity();
    let keep = |len: usize| -> usize {
        if v.ty.is_mf() {
            len - len % ar
        } else {
            len.min(ar)
        }
    };
    match &mut v.data {
        FieldData::Float(x) => {
            let k = keep(x.len());
            x.truncate(k)
        }
        FieldData::Double(x) => {
            let k = keep(x.len());
            x.truncate(k)
        }
        FieldData::Bool(x) if !v.ty.is_mf() => x.truncate(1),
        FieldData::Int32(x) if !v.ty.is_mf() => x.truncate(1),
        FieldData::String(x) if !v.ty.is_mf() => x.truncate(1),
        FieldData::Node(x) if !v.ty.is_mf() => x.truncate(1),
        _ => {}
    }
}

/// Shortest round-tripping decimal for an `f32`.
pub fn fmt_f32(f: f32) -> String {
    if f == 0.0 {
        return "0".into();
    }
    if !f.is_finite() {
        return "0".into();
    }
    format!("{f}")
}

/// Shortest round-tripping decimal for an `f64`.
pub fn fmt_f64(f: f64) -> String {
    if f == 0.0 || !f.is_finite() {
        return "0".into();
    }
    format!("{f}")
}

/// Quote one string for MFString / ClassicVRML syntax.
pub fn quote_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        if c == '"' || c == '\\' {
            out.push('\\');
        }
        out.push(c);
    }
    out.push('"');
    out
}

/// Format a value with the XML-encoding attribute syntax (unescaped —
/// the XML writer escapes it). Node values format as an empty string.
pub fn format_xml_value(v: &FieldValue) -> String {
    format_value(v, false)
}

/// Shared formatter: `classic == true` uses ClassicVRML spellings
/// (`TRUE`/`FALSE`, quoted `SFString`); brackets are the caller's job.
pub(crate) fn format_value(v: &FieldValue, classic: bool) -> String {
    let ar = v.ty.arity();
    let sep_tuples = |parts: Vec<String>| -> String {
        let mut s = String::new();
        for (i, p) in parts.iter().enumerate() {
            if i > 0 {
                if i % ar == 0 && v.ty.is_mf() && ar > 1 {
                    s.push_str(", ");
                } else {
                    s.push(' ');
                }
            }
            s.push_str(p);
        }
        s
    };
    match &v.data {
        FieldData::Bool(b) => b
            .iter()
            .map(|&x| match (x, classic) {
                (true, false) => "true",
                (false, false) => "false",
                (true, true) => "TRUE",
                (false, true) => "FALSE",
            })
            .collect::<Vec<_>>()
            .join(" "),
        FieldData::Int32(x) => x
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(" "),
        FieldData::Float(x) => sep_tuples(x.iter().map(|&f| fmt_f32(f)).collect()),
        FieldData::Double(x) => sep_tuples(x.iter().map(|&f| fmt_f64(f)).collect()),
        FieldData::String(x) => {
            if v.ty.is_mf() || classic {
                x.iter()
                    .map(|s| quote_string(s))
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                x.first().cloned().unwrap_or_default()
            }
        }
        FieldData::Image(imgs) => {
            let mut s = String::new();
            for (k, im) in imgs.iter().enumerate() {
                if k > 0 {
                    s.push_str(", ");
                }
                s.push_str(&format!("{} {} {}", im.width, im.height, im.components));
                for p in &im.pixels {
                    s.push_str(&format!(" 0x{p:X}"));
                }
            }
            s
        }
        FieldData::Node(_) => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_names_roundtrip() {
        for &t in FieldType::ALL {
            assert_eq!(FieldType::from_name(t.name()), Some(t));
            assert_eq!(t.single().multi(), t.multi());
        }
        assert_eq!(FieldType::MFVec3f.arity(), 3);
        assert_eq!(FieldType::SFMatrix4d.storage(), Storage::Double);
    }

    #[test]
    fn numbers_and_commas() {
        let v = parse_xml_value(FieldType::MFColor, "1 1 1, 0 0 0,  ").unwrap();
        assert_eq!(v.as_tuples::<3>(), vec![[1.0, 1.0, 1.0], [0.0; 3]]);
        let v = parse_xml_value(FieldType::MFInt32, "0 1 0x10 -1").unwrap();
        assert_eq!(v.as_i32s(), &[0, 1, 16, -1]);
        assert!(parse_xml_value(FieldType::SFFloat, "abc").is_err());
        let v = parse_xml_value(FieldType::MFVec2f, "1 2 3").unwrap();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn strings() {
        assert_eq!(
            parse_mfstring(r#""He said, \"Immel did it!\"" "b""#).unwrap(),
            vec![r#"He said, "Immel did it!""#.to_string(), "b".into()]
        );
        assert_eq!(parse_mfstring("plain.png").unwrap(), vec!["plain.png"]);
        assert_eq!(parse_mfstring("").unwrap(), Vec::<String>::new());
        assert!(parse_mfstring(r#""open"#).is_err());
        let v = parse_xml_value(FieldType::MFString, r#""a" "b\\c""#).unwrap();
        assert_eq!(format_xml_value(&v), r#""a" "b\\c""#);
    }

    #[test]
    fn image() {
        let v = parse_xml_value(
            FieldType::SFImage,
            "2 4 3 0xFF0000 0xFF00 0 0 0 0 0xFFFFFF 0xFFFF00",
        )
        .unwrap();
        let im = &v.as_images()[0];
        assert_eq!((im.width, im.height, im.components), (2, 4, 3));
        let rgba = im.to_rgba8_top_down();
        // Top-left pixel is white (last row in file order).
        assert_eq!(&rgba[0..4], &[255, 255, 255, 255]);
        // Bottom-left is red.
        assert_eq!(&rgba[6 * 4..6 * 4 + 4], &[255, 0, 0, 255]);
        assert!(parse_xml_value(FieldType::SFImage, "99999999 99999999 3").is_err());
    }
}
