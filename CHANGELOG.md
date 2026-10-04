# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.0.1](https://github.com/OxideAV/oxideav-x3d/compare/v0.0.0...v0.0.1) - 2026-10-04

### Other

- Scene3D → X3D 4.0 XML (Mesh3DEncoder) + round-trip tests
- Scene conversion: X3dDocument → Scene3D + Mesh3DDecoder
- Document layer: XML tokenizer, typed field values, X3DUOM node table, XML reader/writer

### Added

- Document layer: hand-written hostile-input-hardened XML reader, typed
  values for all 42 SF/MF field types, node table for the 260 X3D 4.0
  node types generated from the X3D Unified Object Model, `X3dDocument`
  arena model with DEF/USE, prototypes (`ProtoDeclare`,
  `ExternProtoDeclare`, `ProtoInstance`, `IS`), ROUTE, IMPORT/EXPORT,
  Script/shader declarations, header statements; XML writer.
- gzip `.x3dz` / `.x3dvz` input and output.
- ClassicVRML `.x3dv` reading and writing through `oxideav-vrml`'s
  syntax layer (`Dialect::X3dClassic`) with an X3D `NodeCatalog`.
- JSON encoding (19776-5 draft) reader.
- Scene conversion to `Scene3D`: prototype expansion, grouping nodes,
  all polygonal / line / point / analytic / Geometry2D geometry,
  creaseAngle normals, default texture coordinates, materials
  (PhysicalMaterial, Material, UnlitMaterial, TwoSidedMaterial),
  textures (ImageTexture, PixelTexture, data: URIs, TextureProperties,
  TextureTransform), viewpoints, lights, interpolator animations,
  CoordinateInterpolator morph targets, H-Anim skinning, units,
  metadata, optional Inline resolver.
- `X3dDecoder` / `X3dEncoder` (`Mesh3DDecoder` / `Mesh3DEncoder`),
  feature-gated `register()`.
- Encoder: Scene3D → X3D 4.0 XML or ClassicVRML with DEF/USE sharing,
  PhysicalMaterial, viewpoints, lights and animations.
- cargo-fuzz harnesses (`decode`, `roundtrip`) and daily fuzz workflow.
