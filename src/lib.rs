//! # oxideav-x3d
//!
//! Pure-Rust, clean-room reader and writer for **X3D** — ISO/IEC
//! 19775-1 (architecture and base components, edition 4 / X3D 4.0,
//! compatible with 3.x content) in the **XML encoding** (ISO/IEC
//! 19776-1, `.x3d`) and the **ClassicVRML encoding** (ISO/IEC
//! 19776-2, `.x3dv`), plus their gzip-compressed forms (`.x3dz`,
//! `.x3dvz`).
//!
//! The crate has two layers:
//!
//! * **Document layer** — [`X3dDocument`]: an encoding-independent,
//!   typed node graph (arena + `DEF`/`USE` sharing, prototypes,
//!   routes, header statements) with every field value parsed by its
//!   declared type from a node table generated from the Web3D X3D
//!   Unified Object Model. Unknown nodes and fields are preserved.
//!   [`parse_document`] reads it, [`write_xml`] writes it.
//! * **Scene layer** — conversion to and from
//!   [`oxideav_mesh3d::Scene3D`] (glTF-2.0-aligned): [`X3dDecoder`]
//!   implements [`Mesh3DDecoder`](oxideav_mesh3d::Mesh3DDecoder),
//!   [`X3dEncoder`] implements
//!   [`Mesh3DEncoder`](oxideav_mesh3d::Mesh3DEncoder).
//!
//! X3D is right-handed, Y-up, metres — the same convention as the
//! mesh3d model — so no axis conversion is applied.
//!
//! ## Standalone build
//!
//! ```toml
//! oxideav-x3d = { version = "0.0", default-features = false }
//! ```
//!
//! drops `oxideav-core` and the [`register`] helper; everything else
//! stays available.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod convert;
pub mod decoder;
pub mod document;
pub mod encoder;
pub mod error;
pub mod field;
pub mod nodes;
pub mod xml;
pub mod xml_reader;
pub mod xml_writer;

pub use convert::{document_to_scene, ConvertOptions};
pub use decoder::X3dDecoder;
pub use document::{NodeIdx, X3dDocument, X3dNode};
pub use encoder::{scene_to_document, X3dEncoder};
pub use error::{Error, Result};
pub use field::{AccessType, FieldData, FieldType, FieldValue};
pub use xml_writer::write_xml;

/// Register the X3D decoder and encoders with a
/// [`Mesh3DRegistry`](oxideav_mesh3d::Mesh3DRegistry).
///
/// * decoder `"x3d"` — extensions `x3d`, `x3dz`, `x3dv`, `x3dvz`
///   (encoding and compression are sniffed from the bytes);
/// * encoder `"x3d"` — extension `x3d` (XML);
/// * encoder `"x3dz"` — extension `x3dz` (gzip-compressed XML).
#[cfg(feature = "registry")]
pub fn register(registry: &mut oxideav_mesh3d::Mesh3DRegistry) {
    registry.register_decoder(
        "x3d",
        &["x3d", "x3dz", "x3dv", "x3dvz"],
        Box::new(|| Box::new(X3dDecoder::new())),
    );
    registry.register_encoder("x3d", &["x3d"], Box::new(|| Box::new(X3dEncoder::new())));
    registry.register_encoder(
        "x3dz",
        &["x3dz"],
        Box::new(|| Box::new(X3dEncoder::new().with_gzip(true))),
    );
}

/// Hostile-input bounds applied while reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Largest accepted (decompressed) input, in bytes.
    pub max_input_bytes: usize,
    /// Deepest element / node nesting.
    pub max_depth: usize,
    /// Most XML elements in one document.
    pub max_elements: u64,
    /// Most attributes on one element.
    pub max_attributes: usize,
    /// Most nodes in the document arena (prototype expansion included).
    pub max_nodes: usize,
    /// Deepest nested prototype expansion.
    pub max_proto_depth: usize,
    /// Most vertices produced by geometry conversion, scene-wide.
    pub max_vertices: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_input_bytes: 512 << 20,
            max_depth: 256,
            max_elements: 8 << 20,
            max_attributes: 1024,
            max_nodes: 4 << 20,
            max_proto_depth: 16,
            max_vertices: 64 << 20,
        }
    }
}

/// `true` when `bytes` start with the gzip magic (RFC 1952).
pub fn is_gzip(bytes: &[u8]) -> bool {
    bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b
}

/// Inflate a gzip stream (`.x3dz` / `.x3dvz`), capped at
/// `limits.max_input_bytes` of output.
pub fn gunzip(bytes: &[u8], limits: &Limits) -> Result<Vec<u8>> {
    match compcol::vec::decompress_to_vec_capped::<compcol::gzip::Gzip>(
        bytes,
        limits.max_input_bytes as u64,
    ) {
        Ok(v) => Ok(v),
        Err(compcol::Error::OutputLimitExceeded) => {
            Err(Error::limit("gzip stream inflates past max_input_bytes"))
        }
        Err(e) => Err(Error::invalid(format!("gzip inflate failed: {e}"))),
    }
}

/// Compress bytes as a gzip stream (for `.x3dz` / `.x3dvz` output).
pub fn gzip(bytes: &[u8]) -> Result<Vec<u8>> {
    compcol::vec::compress_to_vec::<compcol::gzip::Gzip>(bytes)
        .map_err(|e| Error::invalid(format!("gzip deflate failed: {e}")))
}

/// Parse an X3D document from raw bytes, auto-detecting gzip
/// compression and the encoding (XML when the first significant
/// character is `<`, otherwise ClassicVRML).
pub fn parse_document(bytes: &[u8]) -> Result<X3dDocument> {
    parse_document_with_limits(bytes, &Limits::default())
}

/// [`parse_document`] with explicit [`Limits`].
pub fn parse_document_with_limits(bytes: &[u8], limits: &Limits) -> Result<X3dDocument> {
    let owned;
    let bytes = if is_gzip(bytes) {
        owned = gunzip(bytes, limits)?;
        &owned[..]
    } else {
        bytes
    };
    if bytes.len() > limits.max_input_bytes {
        return Err(Error::limit("input larger than max_input_bytes"));
    }
    let text = decode_text(bytes)?;
    let first = text
        .trim_start_matches('\u{feff}')
        .trim_start()
        .chars()
        .next();
    match first {
        Some('<') => xml_reader::read_xml(&text, limits),
        Some(_) => Err(Error::unsupported(
            "ClassicVRML encoding is not supported yet",
        )),
        None => Err(Error::invalid("empty input")),
    }
}

/// Decode input bytes to text: UTF-8 (BOM optional) or UTF-16 with a
/// BOM.
fn decode_text(bytes: &[u8]) -> Result<String> {
    if bytes.len() >= 2 && (bytes[..2] == [0xff, 0xfe] || bytes[..2] == [0xfe, 0xff]) {
        let le = bytes[0] == 0xff;
        let units: Vec<u16> = bytes[2..]
            .chunks_exact(2)
            .map(|c| {
                if le {
                    u16::from_le_bytes([c[0], c[1]])
                } else {
                    u16::from_be_bytes([c[0], c[1]])
                }
            })
            .collect();
        return String::from_utf16(&units).map_err(|_| Error::invalid("invalid UTF-16 text"));
    }
    match std::str::from_utf8(bytes) {
        Ok(s) => Ok(s.to_string()),
        // Legacy content is occasionally Latin-1; map bytes 1:1.
        Err(_) => Ok(bytes.iter().map(|&b| b as char).collect()),
    }
}
