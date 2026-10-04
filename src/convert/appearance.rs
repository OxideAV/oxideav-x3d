//! Appearance / material / texture nodes → mesh3d materials and
//! textures.
//!
//! * `PhysicalMaterial` (X3D 4.0 PBR) maps one-to-one onto the
//!   glTF metallic-roughness model.
//! * `Material` (Phong/Blinn) becomes a dielectric (`metallic = 0`)
//!   with `base_color = diffuseColor`, `roughness = 1 − shininess`;
//!   the original Phong parameters ride along in
//!   `extras["x3d:material"]` so an X3D round trip restores them.
//! * `UnlitMaterial` sets [`MaterialExt::unlit`](oxideav_mesh3d::MaterialExt)
//!   with `base_color = emissiveColor`.
//! * `TwoSidedMaterial` (X3D 3) is treated as a double-sided `Material`.
//! * A Shape without Appearance / material is unlit white
//!   (ISO/IEC 19775-1 12.2.2).
//!
//! `transparency` becomes base-colour alpha; `Appearance.alphaMode`
//! maps directly (`AUTO` → `BLEND` when any transparency is present).

use std::collections::HashMap;
use std::sync::Arc;

use oxideav_mesh3d::{
    AlphaMode, InMemoryAsset, MagFilter, Material, MinFilter, Sampler, Texture, TextureRef,
    WrapMode,
};
use serde_json::{json, Value};

use super::geometry::TexTransform2;
use crate::document::{NodeIdx, X3dDocument, X3dNode};

fn f32v(n: &X3dNode, name: &str, d: f32) -> f32 {
    n.value(name).and_then(|v| v.as_f32()).unwrap_or(d)
}

fn color(n: &X3dNode, name: &str, d: [f32; 3]) -> [f32; 3] {
    n.value(name).and_then(|v| v.as_tuple::<3>()).unwrap_or(d)
}

fn string(n: &X3dNode, name: &str) -> Option<String> {
    n.value(name).and_then(|v| v.as_str().map(str::to_string))
}

/// Texture transform of an appearance (first entry of a
/// MultiTextureTransform).
pub fn tex_transform(
    doc: &X3dDocument,
    app: Option<&X3dNode>,
    angle: f32,
) -> Option<TexTransform2> {
    let app = app?;
    let tt = doc.node(*app.children_of("textureTransform").first()?)?;
    let tt = if tt.type_name == "MultiTextureTransform" {
        doc.node(*tt.children_of("textureTransform").first()?)?
    } else {
        tt
    };
    if tt.type_name != "TextureTransform" {
        return None;
    }
    let get2 =
        |name: &str, d: [f32; 2]| tt.value(name).and_then(|v| v.as_tuple::<2>()).unwrap_or(d);
    let t = TexTransform2 {
        center: get2("center", [0.0, 0.0]),
        rotation: f32v(tt, "rotation", 0.0) * angle,
        scale: get2("scale", [1.0, 1.0]),
        translation: get2("translation", [0.0, 0.0]),
    };
    let identity = TexTransform2 {
        center: t.center,
        rotation: 0.0,
        scale: [1.0, 1.0],
        translation: [0.0, 0.0],
    };
    (t != identity).then_some(t)
}

/// Material cache key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatKey {
    /// Appearance node.
    pub appearance: Option<NodeIdx>,
    /// Geometry `solid` field.
    pub solid: bool,
    /// Geometry has per-vertex colours.
    pub has_colors: bool,
    /// Geometry is lines/points (unlit emissive in X3D).
    pub unlit_geometry: bool,
}

/// Builds mesh3d materials and textures with caching.
#[derive(Debug, Default)]
pub struct MaterialBuilder {
    materials: HashMap<MatKey, oxideav_mesh3d::MaterialId>,
    textures: HashMap<NodeIdx, Option<oxideav_mesh3d::TextureId>>,
}

impl MaterialBuilder {
    /// Material for a Shape.
    pub fn material(
        &mut self,
        doc: &X3dDocument,
        scene: &mut oxideav_mesh3d::Scene3D,
        key: MatKey,
        warnings: &mut Vec<String>,
    ) -> oxideav_mesh3d::MaterialId {
        if let Some(&m) = self.materials.get(&key) {
            return m;
        }
        let mat = self.build(doc, scene, key, warnings);
        let id = scene.add_material(mat);
        self.materials.insert(key, id);
        id
    }

    fn texture(
        &mut self,
        doc: &X3dDocument,
        scene: &mut oxideav_mesh3d::Scene3D,
        idx: NodeIdx,
        warnings: &mut Vec<String>,
    ) -> Option<oxideav_mesh3d::TextureId> {
        if let Some(t) = self.textures.get(&idx) {
            return *t;
        }
        let t = build_texture(doc, idx, warnings).map(|t| scene.add_texture(t));
        self.textures.insert(idx, t);
        t
    }

    fn tex_ref(
        &mut self,
        doc: &X3dDocument,
        scene: &mut oxideav_mesh3d::Scene3D,
        owner: &X3dNode,
        field: &str,
        warnings: &mut Vec<String>,
    ) -> Option<TextureRef> {
        let idx = *owner.children_of(field).first()?;
        let id = self.texture(doc, scene, idx, warnings)?;
        Some(TextureRef::new(id))
    }

    fn build(
        &mut self,
        doc: &X3dDocument,
        scene: &mut oxideav_mesh3d::Scene3D,
        key: MatKey,
        warnings: &mut Vec<String>,
    ) -> Material {
        let mut m = Material::new();
        m.double_sided = !key.solid;
        let app = key.appearance.and_then(|a| doc.node(a));
        let Some(app) = app else {
            // No appearance: unlit white.
            m.metallic = 0.0;
            m.ext.unlit = true;
            m.extras
                .insert("x3d:material".into(), json!({"type": null}));
            return m;
        };
        m.name = app.def.clone();
        let mat_node = app
            .children_of("material")
            .first()
            .and_then(|&i| doc.node(i));
        let app_tex = app.children_of("texture").first().copied();
        let mut transparency = 0.0f32;
        let mut extras_mat = serde_json::Map::new();
        match mat_node {
            None => {
                m.metallic = 0.0;
                m.ext.unlit = true;
                extras_mat.insert("type".into(), Value::Null);
                if let Some(t) = app_tex {
                    m.base_color_texture =
                        self.texture(doc, scene, t, warnings).map(TextureRef::new);
                }
            }
            Some(mn) => {
                if m.name.is_none() {
                    m.name = mn.def.clone();
                }
                extras_mat.insert("type".into(), json!(mn.type_name));
                match mn.type_name.as_str() {
                    "PhysicalMaterial" => {
                        let bc = color(mn, "baseColor", [1.0; 3]);
                        transparency = f32v(mn, "transparency", 0.0);
                        m.base_color = [bc[0], bc[1], bc[2], 1.0 - transparency];
                        m.metallic = f32v(mn, "metallic", 1.0);
                        m.roughness = f32v(mn, "roughness", 1.0);
                        m.emissive_factor = color(mn, "emissiveColor", [0.0; 3]);
                        m.normal_scale = f32v(mn, "normalScale", 1.0);
                        m.occlusion_strength = f32v(mn, "occlusionStrength", 1.0);
                        m.base_color_texture =
                            self.tex_ref(doc, scene, mn, "baseTexture", warnings);
                        m.metallic_roughness_texture =
                            self.tex_ref(doc, scene, mn, "metallicRoughnessTexture", warnings);
                        m.emissive_texture =
                            self.tex_ref(doc, scene, mn, "emissiveTexture", warnings);
                        m.normal_texture = self.tex_ref(doc, scene, mn, "normalTexture", warnings);
                        m.occlusion_texture =
                            self.tex_ref(doc, scene, mn, "occlusionTexture", warnings);
                        if m.base_color_texture.is_none() {
                            if let Some(t) = app_tex {
                                m.base_color_texture =
                                    self.texture(doc, scene, t, warnings).map(TextureRef::new);
                            }
                        }
                        if key.has_colors {
                            // Colour nodes replace baseColor.
                            m.base_color[0] = 1.0;
                            m.base_color[1] = 1.0;
                            m.base_color[2] = 1.0;
                            extras_mat.insert("baseColor".into(), json!(bc));
                        }
                        if key.unlit_geometry {
                            m.ext.unlit = true;
                            let e = m.emissive_factor;
                            if !key.has_colors {
                                m.base_color = [e[0], e[1], e[2], m.base_color[3]];
                            }
                        }
                    }
                    "UnlitMaterial" => {
                        let ec = color(mn, "emissiveColor", [1.0; 3]);
                        transparency = f32v(mn, "transparency", 0.0);
                        m.metallic = 0.0;
                        m.ext.unlit = true;
                        m.base_color = [ec[0], ec[1], ec[2], 1.0 - transparency];
                        m.normal_scale = f32v(mn, "normalScale", 1.0);
                        m.base_color_texture = self
                            .tex_ref(doc, scene, mn, "emissiveTexture", warnings)
                            .or_else(|| {
                                app_tex.and_then(|t| {
                                    self.texture(doc, scene, t, warnings).map(TextureRef::new)
                                })
                            });
                        m.normal_texture = self.tex_ref(doc, scene, mn, "normalTexture", warnings);
                        if key.has_colors {
                            m.base_color[0] = 1.0;
                            m.base_color[1] = 1.0;
                            m.base_color[2] = 1.0;
                        }
                    }
                    other => {
                        // Material, TwoSidedMaterial, or an unknown
                        // material node: Phong mapping.
                        let dc = color(mn, "diffuseColor", [0.8; 3]);
                        let ec = color(mn, "emissiveColor", [0.0; 3]);
                        let sc = color(mn, "specularColor", [0.0; 3]);
                        let sh = f32v(mn, "shininess", 0.2);
                        let amb = f32v(mn, "ambientIntensity", 0.2);
                        transparency = f32v(mn, "transparency", 0.0);
                        m.base_color = [dc[0], dc[1], dc[2], 1.0 - transparency];
                        m.metallic = 0.0;
                        m.roughness = (1.0 - sh).clamp(0.0, 1.0);
                        m.emissive_factor = ec;
                        m.normal_scale = f32v(mn, "normalScale", 1.0);
                        m.occlusion_strength = f32v(mn, "occlusionStrength", 1.0);
                        extras_mat.insert("diffuseColor".into(), json!(dc));
                        extras_mat.insert("specularColor".into(), json!(sc));
                        extras_mat.insert("shininess".into(), json!(sh));
                        extras_mat.insert("ambientIntensity".into(), json!(amb));
                        m.base_color_texture =
                            self.tex_ref(doc, scene, mn, "diffuseTexture", warnings);
                        m.emissive_texture =
                            self.tex_ref(doc, scene, mn, "emissiveTexture", warnings);
                        m.normal_texture = self.tex_ref(doc, scene, mn, "normalTexture", warnings);
                        m.occlusion_texture =
                            self.tex_ref(doc, scene, mn, "occlusionTexture", warnings);
                        if m.base_color_texture.is_none() {
                            if let Some(t) = app_tex {
                                m.base_color_texture =
                                    self.texture(doc, scene, t, warnings).map(TextureRef::new);
                            }
                        }
                        if other == "TwoSidedMaterial" {
                            m.double_sided = true;
                            if f32v(mn, "backTransparency", 0.0) != 0.0
                                || mn.value("separateBackColor").and_then(|v| v.as_bool())
                                    == Some(true)
                            {
                                extras_mat.insert(
                                    "back".into(),
                                    json!({
                                        "diffuseColor": color(mn, "backDiffuseColor", [0.8; 3]),
                                        "emissiveColor": color(mn, "backEmissiveColor", [0.0; 3]),
                                        "specularColor": color(mn, "backSpecularColor", [0.0; 3]),
                                        "shininess": f32v(mn, "backShininess", 0.2),
                                        "transparency": f32v(mn, "backTransparency", 0.0),
                                    }),
                                );
                            }
                        }
                        if key.has_colors {
                            m.base_color[0] = 1.0;
                            m.base_color[1] = 1.0;
                            m.base_color[2] = 1.0;
                        }
                        if key.unlit_geometry {
                            // Lines / points are unlit, coloured by
                            // emissiveColor (ISO/IEC 19775-1 17.2.2.3).
                            m.ext.unlit = true;
                            if !key.has_colors {
                                m.base_color = [ec[0], ec[1], ec[2], m.base_color[3]];
                            }
                        }
                    }
                }
            }
        }
        if let Some(bm) = app
            .children_of("backMaterial")
            .first()
            .and_then(|&i| doc.node(i))
        {
            extras_mat.insert("backMaterialType".into(), json!(bm.type_name));
        }
        let alpha_mode = app
            .value("alphaMode")
            .and_then(|v| v.as_str().map(str::to_ascii_uppercase))
            .unwrap_or_else(|| "AUTO".into());
        let cutoff = f32v(app, "alphaCutoff", 0.5);
        m.alpha_mode = match alpha_mode.as_str() {
            "OPAQUE" => AlphaMode::Opaque,
            "MASK" => AlphaMode::Mask { cutoff },
            "BLEND" => AlphaMode::Blend,
            _ => {
                if transparency > 0.0 {
                    AlphaMode::Blend
                } else {
                    AlphaMode::Opaque
                }
            }
        };
        if alpha_mode != "AUTO" {
            extras_mat.insert("alphaMode".into(), json!(alpha_mode));
        }
        if let Some(t) = app_tex.and_then(|t| doc.node(t)) {
            if t.type_name == "MultiTexture" {
                extras_mat.insert("multiTexture".into(), json!(t.children_of("texture").len()));
            }
        }
        if let Some(tt) = app
            .children_of("textureTransform")
            .first()
            .and_then(|&i| doc.node(i))
        {
            extras_mat.insert("textureTransformBaked".into(), json!(tt.type_name));
        }
        m.extras
            .insert("x3d:material".into(), Value::Object(extras_mat));
        m
    }
}

/// Decode a `data:` URI (RFC 2397). Returns `(mime, bytes)`.
pub fn decode_data_uri(uri: &str) -> Option<(String, Vec<u8>)> {
    let rest = uri.strip_prefix("data:")?;
    let comma = rest.find(',')?;
    let (meta, data) = (&rest[..comma], &rest[comma + 1..]);
    let is_b64 = meta.ends_with(";base64");
    let mime = meta
        .trim_end_matches(";base64")
        .split(';')
        .next()
        .filter(|m| !m.is_empty())
        .unwrap_or("text/plain")
        .to_string();
    let bytes = if is_b64 {
        base64_decode(data)?
    } else {
        percent_decode(data)
    };
    Some((mime, bytes))
}

fn percent_decode(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    let hex = |c: u8| (c as char).to_digit(16);
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2])) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    out
}

/// Standard base64 (RFC 4648 §4) decoder; whitespace ignored.
pub fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut acc = 0u32;
    let mut bits = 0u32;
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => break,
            b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => return None,
        } as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

/// Standard base64 encoder (with padding).
pub fn base64_encode(b: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(b.len().div_ceil(3) * 4);
    for c in b.chunks(3) {
        let n = (c[0] as u32) << 16
            | (*c.get(1).unwrap_or(&0) as u32) << 8
            | *c.get(2).unwrap_or(&0) as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Encode 8-bit RGBA rows (top to bottom) as a PNG file.
pub fn encode_png_rgba(width: u32, height: u32, rgba: &[u8]) -> Option<Vec<u8>> {
    let mut raw = Vec::with_capacity((width as usize * 4 + 1) * height as usize);
    for row in rgba.chunks_exact(width as usize * 4).take(height as usize) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let z = compcol::vec::compress_to_vec::<compcol::zlib::Zlib>(&raw).ok()?;
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut chunk = |ty: &[u8; 4], data: &[u8]| {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut c = Vec::with_capacity(4 + data.len());
        c.extend_from_slice(ty);
        c.extend_from_slice(data);
        out.extend_from_slice(&c);
        out.extend_from_slice(&crc32(&c).to_be_bytes());
    };
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(b"IHDR", &ihdr);
    chunk(b"IDAT", &z);
    chunk(b"IEND", &[]);
    Some(out)
}

fn mime_for(uri: &str) -> Option<String> {
    let lower = uri.to_ascii_lowercase();
    let ext = lower.rsplit('.').next()?;
    Some(
        match ext {
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "gif" => "image/gif",
            "webp" => "image/webp",
            "ktx2" => "image/ktx2",
            "bmp" => "image/bmp",
            _ => return None,
        }
        .to_string(),
    )
}

fn sampler_for(doc: &X3dDocument, n: &X3dNode) -> Sampler {
    let mut s = Sampler::default_sampler();
    let rs = n.value("repeatS").and_then(|v| v.as_bool()).unwrap_or(true);
    let rt = n.value("repeatT").and_then(|v| v.as_bool()).unwrap_or(true);
    s.wrap_s = if rs {
        WrapMode::Repeat
    } else {
        WrapMode::ClampToEdge
    };
    s.wrap_t = if rt {
        WrapMode::Repeat
    } else {
        WrapMode::ClampToEdge
    };
    if let Some(tp) = n
        .children_of("textureProperties")
        .first()
        .and_then(|&i| doc.node(i))
    {
        let mode = |name: &str, cur: WrapMode| -> WrapMode {
            match tp
                .value(name)
                .and_then(|v| v.as_str().map(str::to_ascii_uppercase))
                .as_deref()
            {
                Some("REPEAT") => WrapMode::Repeat,
                Some("MIRRORED_REPEAT") => WrapMode::MirroredRepeat,
                Some("CLAMP" | "CLAMP_TO_EDGE" | "CLAMP_TO_BOUNDARY") => WrapMode::ClampToEdge,
                _ => cur,
            }
        };
        s.wrap_s = mode("boundaryModeS", s.wrap_s);
        s.wrap_t = mode("boundaryModeT", s.wrap_t);
        let mag = tp
            .value("magnificationFilter")
            .and_then(|v| v.as_str().map(str::to_ascii_uppercase));
        s.mag_filter = match mag.as_deref() {
            Some("NEAREST_PIXEL" | "FASTEST") => Some(MagFilter::Nearest),
            Some("AVG_PIXEL" | "NICEST") => Some(MagFilter::Linear),
            _ => None,
        };
        let min = tp
            .value("minificationFilter")
            .and_then(|v| v.as_str().map(str::to_ascii_uppercase));
        s.min_filter = match min.as_deref() {
            Some("NEAREST_PIXEL" | "FASTEST") => Some(MinFilter::Nearest),
            Some("AVG_PIXEL") => Some(MinFilter::Linear),
            Some("NEAREST_PIXEL_NEAREST_MIPMAP") => Some(MinFilter::NearestMipNearest),
            Some("AVG_PIXEL_NEAREST_MIPMAP") => Some(MinFilter::LinearMipNearest),
            Some("NEAREST_PIXEL_AVG_MIPMAP") => Some(MinFilter::NearestMipLinear),
            Some("AVG_PIXEL_AVG_MIPMAP" | "NICEST") => Some(MinFilter::LinearMipLinear),
            _ => None,
        };
    }
    s
}

fn build_texture(doc: &X3dDocument, idx: NodeIdx, warnings: &mut Vec<String>) -> Option<Texture> {
    let n = doc.node(idx)?;
    match n.type_name.as_str() {
        "ImageTexture" => {
            let urls = n
                .value("url")
                .map(|v| v.as_strings().to_vec())
                .unwrap_or_default();
            let url = urls
                .iter()
                .find(|u| !u.trim().is_empty())?
                .trim()
                .to_string();
            let mut tex = if url.starts_with("data:") {
                match decode_data_uri(&url) {
                    Some((mime, bytes)) => Texture::from_encoded(mime, bytes),
                    None => {
                        warnings.push("ImageTexture: undecodable data: URI".into());
                        return None;
                    }
                }
            } else {
                let mut t = Texture::from_uri(url.clone());
                if let oxideav_mesh3d::ImageData::External { mime, .. } = &mut t.image {
                    *mime = mime_for(&url);
                }
                t
            };
            tex.name = n.def.clone();
            tex.sampler = sampler_for(doc, n);
            Some(tex)
        }
        "PixelTexture" => {
            let img = n.value("image")?.as_images().first()?.clone();
            if img.width == 0 || img.height == 0 {
                return None;
            }
            let rgba = img.to_rgba8_top_down();
            let png = encode_png_rgba(img.width, img.height, &rgba)?;
            let mut tex = Texture::from_source(Arc::new(InMemoryAsset::new(
                Some("image/png".to_string()),
                png,
            )));
            tex.name = n.def.clone();
            tex.sampler = sampler_for(doc, n);
            Some(tex)
        }
        "MultiTexture" => {
            let first = *n.children_of("texture").first()?;
            build_texture(doc, first, warnings)
        }
        other => {
            warnings.push(format!("texture node {other} not converted"));
            None
        }
    }
}

/// Whether a geometry node type renders as lines or points.
pub fn is_unlit_geometry(type_name: &str) -> bool {
    matches!(
        type_name,
        "IndexedLineSet"
            | "LineSet"
            | "PointSet"
            | "Circle2D"
            | "Arc2D"
            | "Polyline2D"
            | "Polypoint2D"
    )
}

/// Texture-coordinate generator mode of a geometry, if any.
pub fn texgen_mode(doc: &X3dDocument, geom: &X3dNode) -> Option<String> {
    let tc = doc.node(*geom.children_of("texCoord").first()?)?;
    (tc.type_name == "TextureCoordinateGenerator")
        .then(|| string(tc, "mode").unwrap_or_else(|| "SPHERE".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_roundtrip() {
        for s in [
            &b""[..],
            b"a",
            b"ab",
            b"abc",
            b"abcd",
            &[0u8, 255, 128, 7, 9],
        ] {
            assert_eq!(base64_decode(&base64_encode(s)).unwrap(), s);
        }
        let (m, b) = decode_data_uri("data:image/png;base64,iVBORw==").unwrap();
        assert_eq!(m, "image/png");
        assert_eq!(&b[..4], &[0x89, b'P', b'N', b'G']);
        assert_eq!(decode_data_uri("data:,a%20b").unwrap().1, b"a b");
    }

    #[test]
    fn png_header() {
        let p = encode_png_rgba(1, 1, &[1, 2, 3, 4]).unwrap();
        assert_eq!(&p[..8], &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        assert_eq!(crc32(b"IEND"), 0xae42_6082);
    }
}
