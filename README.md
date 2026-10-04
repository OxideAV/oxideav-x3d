# oxideav-x3d

Pure-Rust, clean-room **X3D** reader and writer — ISO/IEC 19775-1
(edition 4, X3D 4.0; X3D 3.x content reads too) — implementing
[`oxideav-mesh3d`](https://github.com/OxideAV/oxideav-mesh3d)'s
`Mesh3DDecoder` / `Mesh3DEncoder` traits on its glTF-2.0-aligned
`Scene3D` model.

| Encoding | Extension | Read | Write |
|---|---|---|---|
| XML (ISO/IEC 19776-1) | `.x3d` | yes | yes (X3D 4.0, validates against the 4.0 XML Schema) |
| XML, gzip | `.x3dz` | yes | yes |
| ClassicVRML (ISO/IEC 19776-2) | `.x3dv` | yes | yes |
| ClassicVRML, gzip | `.x3dvz` | yes | yes |
| JSON (ISO/IEC 19776-5 draft) | `.x3dj` | yes | no |

Encoding and compression are sniffed from the bytes.

## Status

| Area | Supported |
|---|---|
| Document layer | Typed X3D document model (`X3dDocument`) for all 260 X3D 4.0 node types — node table generated from the Web3D X3D Unified Object Model; all 42 SF/MF field types (MFString quoting rules, SFImage, doubles, matrices); DEF/USE sharing; `ProtoDeclare` / `ExternProtoDeclare` / `ProtoInstance` + `fieldValue` + `IS`/`connect`; `ROUTE`; `IMPORT`/`EXPORT`; Script / shader field declarations and source; `head` (profile, version, component, unit, meta); unknown nodes and fields preserved; XML and ClassicVRML writers |
| Grouping | Transform (exact TRS incl. `center`; pivot node for animated centred transforms; matrix for non-uniform `scaleOrientation`), Group, StaticGroup, Collision, Anchor, Billboard, CAD*, Layer*, Switch (active choice), LOD (highest detail), Inline (extras; optional resolver splices inlined scenes) |
| Geometry | IndexedFaceSet (per-vertex/per-face colour and normal indices, concave faces via ear clipping, `creaseAngle` normal generation, bounding-box default texture coordinates), IndexedTriangleSet / TriangleSet / (Indexed)TriangleStripSet / (Indexed)TriangleFanSet / (Indexed)QuadSet, ElevationGrid, Extrusion (spine-aligned cross-section planes, caps, default UVs), IndexedLineSet / LineSet, PointSet, Box / Sphere / Cone / Cylinder with the spec's texture layouts, Geometry2D (Rectangle2D, Disk2D, Circle2D, Arc2D, ArcClose2D, Polyline2D, Polypoint2D, TriangleSet2D); Coordinate / CoordinateDouble, Normal, Color / ColorRGBA, TextureCoordinate (2D/3D/4D), MultiTextureCoordinate, TextureCoordinateGenerator (extras) |
| Appearance | PhysicalMaterial (1:1 metallic-roughness), Material (Phong → dielectric, originals in extras), UnlitMaterial, TwoSidedMaterial, `alphaMode` / `alphaCutoff`, ImageTexture (URLs, `data:` URIs), PixelTexture (→ embedded PNG), MultiTexture (first), TextureProperties → sampler, TextureTransform (baked into UVs) |
| Cameras / lights | Viewpoint, OrthoViewpoint; DirectionalLight, PointLight, SpotLight |
| Animation | TimeSensor → Position / Orientation interpolator ROUTEs on transforms, viewpoints and lights; CoordinateInterpolator → morph targets with one-hot weight animation |
| H-Anim | HAnimHumanoid skin binding → skeleton + skin (rest-pose inverse binds, top-4 weights) |
| Units / metadata | `unit` statements (known length units → `Scene3D::unit`, other factors → scaling root; angle units applied), Metadata* nodes, WorldInfo / NavigationInfo / Background / Fog → extras |
| Encoder | Scene3D → X3D 4.0: Transforms, IndexedTriangleSet / IndexedLineSet / PointSet with Normal / TextureCoordinate / Color, DEF/USE sharing of shapes / appearances / textures, PhysicalMaterial / UnlitMaterial / restored Phong Material, embedded textures as `data:` URIs, viewpoints, lights, animations (TimeSensor + interpolators + ROUTEs; step emulated, cubic sampled; morph weights → CoordinateInterpolator), units, metadata, required `component` statements |
| Robustness | Iterative XML tokenizer without entity expansion; caps on input size, depth, elements, attributes, nodes, prototype nesting and vertices (`Limits`); stack-safe at the maximum depth; cargo-fuzz harnesses `decode` and `roundtrip` |

Not (yet) converted to Scene3D: Text / FontStyle, NURBS, ParticleSystem,
sensors and scripts (kept in the document, routes listed in extras),
shaders, sound, geospatial coordinates, volume rendering. The encoder
does not write skins (meshes go out in bind pose), KHR texture
transforms or KHR material extensions beyond unlit.

## Usage

```rust,no_run
use oxideav_mesh3d::{Mesh3DDecoder, Mesh3DEncoder};
use oxideav_x3d::{X3dDecoder, X3dEncoder};

let bytes = std::fs::read("scene.x3d").unwrap();
let scene = X3dDecoder::new().decode(&bytes).unwrap();
let classic = X3dEncoder::new().with_classic(true).encode(&scene).unwrap();
std::fs::write("scene.x3dv", classic).unwrap();
```

Document-level access (no scene conversion):

```rust,no_run
let doc = oxideav_x3d::parse_document(&std::fs::read("scene.x3d").unwrap()).unwrap();
for n in &doc.nodes {
    println!("{} {:?}", n.type_name, n.def);
}
let xml = oxideav_x3d::write_xml(&doc);
let x3dv = oxideav_x3d::write_classic(&doc);
```

With the default `registry` feature, `oxideav_x3d::register` adds the
decoder (`x3d`, `x3dz`, `x3dv`, `x3dvz`, `x3dj`) and encoders (`x3d`,
`x3dz`, `x3dv`, `x3dvz`) to a `Mesh3DRegistry`. Build with
`default-features = false` for a standalone crate without
`oxideav-core`.

## ClassicVRML and oxideav-vrml

The ClassicVRML encoding reuses the public VRML-syntax layer of
[`oxideav-vrml`](https://github.com/OxideAV/oxideav-vrml) (its
`Dialect::X3dClassic` lexer, parser and writer), fed with this crate's
X3D node table through a `NodeCatalog`; prototype expansion and scene
conversion stay shared with the XML path.

## Clean-room note

Implemented from the ISO/IEC 19775-1 / 19776-1 / 19776-2 texts, the
19776-5 draft, the X3D XML Schema and the X3D Unified Object Model as
published by the Web3D Consortium (mirrored with provenance in the
OxideAV docs repository under `3d/x3d/`). No source code of other X3D
or VRML implementations was consulted. `src/nodes/generated.rs` is
generated from `X3dUnifiedObjectModel-4.0.xml` by an out-of-tree
script.

## License

MIT — see `LICENSE`.
