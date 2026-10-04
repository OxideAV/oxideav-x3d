//! [`Mesh3DDecoder`] implementation.

use oxideav_mesh3d::{Mesh3DDecoder, Scene3D};

use crate::convert::{document_to_scene, ConvertOptions, UrlResolver};
use crate::Limits;

/// X3D decoder: XML (`.x3d`), ClassicVRML (`.x3dv`) and their gzip
/// forms (`.x3dz`, `.x3dvz`), auto-detected from the bytes.
#[derive(Clone, Debug, Default)]
pub struct X3dDecoder {
    opts: ConvertOptions,
}

impl X3dDecoder {
    /// Decoder with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Circumference segments used to tessellate Sphere / Cone /
    /// Cylinder / Disk2D / Circle2D / Arc2D (default 32).
    pub fn with_segments(mut self, segments: u32) -> Self {
        self.opts.segments = segments.clamp(3, 4096);
        self
    }

    /// Replace the hostile-input [`Limits`].
    pub fn with_limits(mut self, limits: Limits) -> Self {
        self.opts.limits = limits;
        self
    }

    /// Resolve `Inline` URLs to bytes so inlined scenes are spliced in
    /// (each inlined file is decoded with the same options, nesting
    /// bounded).
    pub fn with_inline_resolver(mut self, resolver: UrlResolver) -> Self {
        self.opts.inline_resolver = Some(resolver);
        self
    }

    /// Decode with this decoder's options, reporting the crate-local
    /// error type.
    pub fn decode_scene(&self, bytes: &[u8]) -> crate::Result<Scene3D> {
        let doc = crate::parse_document_with_limits(bytes, &self.opts.limits)?;
        document_to_scene(&doc, &self.opts)
    }
}

impl Mesh3DDecoder for X3dDecoder {
    fn decode(&mut self, bytes: &[u8]) -> oxideav_mesh3d::Result<Scene3D> {
        self.decode_scene(bytes).map_err(Into::into)
    }
}
