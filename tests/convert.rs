//! Scene conversion tests (X3D XML → Scene3D).

use oxideav_mesh3d::{Camera, Light, Mesh3DDecoder, Primitive, Scene3D, Topology, Transform};
use oxideav_x3d::X3dDecoder;

fn decode(name: &str) -> Scene3D {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(path).unwrap();
    X3dDecoder::new().decode(&bytes).unwrap()
}

fn node_named<'a>(s: &'a Scene3D, name: &str) -> &'a oxideav_mesh3d::Node {
    s.nodes
        .iter()
        .find(|n| n.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("no node {name}"))
}

fn prim_under<'a>(s: &'a Scene3D, name: &str) -> &'a Primitive {
    let n = node_named(s, name);
    let child = s.node(n.children[0]).unwrap();
    &s.mesh(child.mesh.unwrap()).unwrap().primitives[0]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Every non-degenerate triangle's winding agrees with its vertex
/// normals.
fn assert_winding_consistent(p: &Primitive, what: &str) {
    let Some(normals) = &p.normals else { return };
    let tris = p.triangle_indices();
    assert!(!tris.is_empty(), "{what}: no triangles");
    for t in tris {
        let [a, b, c] = t.map(|i| p.positions[i as usize]);
        let g = cross(sub(b, a), sub(c, a));
        let l = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
        if l < 1e-6 {
            continue;
        }
        for &i in &t {
            let n = normals[i as usize];
            let d = (g[0] * n[0] + g[1] * n[1] + g[2] * n[2]) / l;
            assert!(d > 0.0, "{what}: normal {n:?} opposes winding {g:?}");
        }
    }
}

#[test]
fn every_geometry_converts_with_consistent_winding() {
    let s = decode("geometry_all.x3d");
    s.validate().unwrap();
    let names = [
        "BoxT",
        "SphereT",
        "ConeT",
        "CylT",
        "IfsT",
        "ConcaveT",
        "CcwFalseT",
        "ItsT",
        "TsT",
        "StripT",
        "IStripT",
        "FanT",
        "IFanT",
        "QuadT",
        "IQuadT",
        "EgT",
        "ExtT",
        "ExtBentT",
    ];
    for n in names {
        let p = prim_under(&s, n);
        assert_eq!(p.topology, Topology::Triangles, "{n}");
        assert_winding_consistent(p, n);
    }
    assert_eq!(prim_under(&s, "LinesT").topology, Topology::Lines);
    assert_eq!(prim_under(&s, "PointsT").topology, Topology::Points);
    // Unknown types: none expected.
    assert!(!s.extras.contains_key("x3d:unconverted"), "{:?}", s.extras);
}

#[test]
fn geometry_details() {
    let s = decode("geometry_all.x3d");
    // Box extents.
    let b = prim_under(&s, "BoxT");
    let max_z = b.positions.iter().map(|p| p[2]).fold(f32::MIN, f32::max);
    assert_eq!(max_z, 3.0);
    assert_eq!(b.triangle_indices().len(), 12);
    // Cube IFS: flat shading (creaseAngle 0) → 24 vertices, 12 tris.
    let c = prim_under(&s, "IfsT");
    assert_eq!(c.positions.len(), 24);
    assert_eq!(c.triangle_indices().len(), 12);
    // Default bbox texture coordinates present, v flipped.
    assert_eq!(c.uvs.len(), 1);
    // Concave polygon: 4 triangles, per-face colour, double-sided.
    let cc = prim_under(&s, "ConcaveT");
    assert_eq!(cc.triangle_indices().len(), 4);
    assert!(cc.colors[0].iter().all(|c| *c == [0.0, 1.0, 0.0, 1.0]));
    let m = &s.materials[cc.material.unwrap().0 as usize];
    assert!(m.double_sided);
    // ITS keeps supplied attributes; v = 1 - t.
    let its = prim_under(&s, "ItsT");
    assert_eq!(its.positions.len(), 4);
    assert_eq!(its.colors[0][3], [1.0, 1.0, 1.0, 0.5]);
    let i0 = its
        .positions
        .iter()
        .position(|p| *p == [0.0, 0.0, 0.0])
        .unwrap();
    assert_eq!(its.uvs[0][i0], [0.0, 1.0]);
    // Strips: 3 triangles.
    assert_eq!(prim_under(&s, "StripT").triangle_indices().len(), 3);
    assert_eq!(prim_under(&s, "IStripT").triangle_indices().len(), 3);
    assert_eq!(prim_under(&s, "FanT").triangle_indices().len(), 2);
    assert_eq!(prim_under(&s, "QuadT").triangle_indices().len(), 2);
    // ElevationGrid 3x3 → 8 triangles, upward.
    let eg = prim_under(&s, "EgT");
    assert_eq!(eg.triangle_indices().len(), 8);
    // Default extrusion: square tube with caps → 4 sides*2 + 2 caps*2.
    assert_eq!(prim_under(&s, "ExtT").triangle_indices().len(), 12);
    // Lines: 3 segments, unlit emissive yellow.
    let l = prim_under(&s, "LinesT");
    assert_eq!(l.indices.as_ref().unwrap().len(), 6);
    let lm = &s.materials[l.material.unwrap().0 as usize];
    assert!(lm.ext.unlit);
    assert_eq!(&lm.base_color[..3], &[1.0, 1.0, 0.0]);
    // USE'd Shape shares the mesh.
    let a = node_named(&s, "BoxT");
    let b2 = node_named(&s, "Shared1");
    assert_eq!(
        s.node(a.children[0]).unwrap().mesh,
        s.node(b2.children[0]).unwrap().mesh
    );
    // Geometry2D group converted (8 shapes).
    assert_eq!(node_named(&s, "TwoDT").children.len(), 8);
}

#[test]
fn transforms_cameras_lights() {
    let x = br#"<X3D version='4.0'><Scene>
      <Viewpoint DEF='VP' position='0 1 5' orientation='0 1 0 0.5' fieldOfView='0.9' description='Front'/>
      <OrthoViewpoint DEF='OV' fieldOfView='-2 -1 2 1'/>
      <DirectionalLight DEF='Sun' direction='1 0 0' intensity='0.5' color='1 0.9 0.8'/>
      <PointLight DEF='Bulb' location='1 2 3' radius='10'/>
      <SpotLight DEF='Spot' location='0 5 0' direction='0 -1 0' beamWidth='0.3' cutOffAngle='0.6'/>
      <Transform DEF='Centered' center='1 0 0' rotation='0 0 1 1.5708'/>
      <Transform DEF='Plain' translation='1 2 3' scale='2 2 2'/>
    </Scene></X3D>"#;
    let s = X3dDecoder::new().decode(x).unwrap();
    s.validate().unwrap();
    let vp = node_named(&s, "VP");
    match s.cameras[vp.camera.unwrap().0 as usize] {
        Camera::Perspective { yfov, .. } => assert!((yfov - 0.9).abs() < 1e-6),
        _ => panic!(),
    }
    match vp.transform {
        Transform::Trs { translation, .. } => assert_eq!(translation, [0.0, 1.0, 5.0]),
        _ => panic!(),
    }
    let ov = node_named(&s, "OV");
    match s.cameras[ov.camera.unwrap().0 as usize] {
        Camera::Orthographic { xmag, ymag, .. } => assert_eq!((xmag, ymag), (2.0, 1.0)),
        _ => panic!(),
    }
    let sun = node_named(&s, "Sun");
    assert!(
        matches!(s.lights[sun.light.unwrap().0 as usize], Light::Directional { intensity, .. } if intensity == 0.5)
    );
    // The sun's -Z axis points along +X.
    let m = sun.transform.to_matrix();
    let fwd = [-m[0][2], -m[1][2], -m[2][2]];
    assert!((fwd[0] - 1.0).abs() < 1e-5, "{fwd:?}");
    let bulb = node_named(&s, "Bulb");
    assert!(
        matches!(s.lights[bulb.light.unwrap().0 as usize], Light::Point { range: Some(r), .. } if r == 10.0)
    );
    let spot = node_named(&s, "Spot");
    match s.lights[spot.light.unwrap().0 as usize] {
        Light::Spot {
            inner_cone_angle,
            outer_cone_angle,
            ..
        } => {
            assert!((inner_cone_angle - 0.3).abs() < 1e-6 && (outer_cone_angle - 0.6).abs() < 1e-6)
        }
        _ => panic!(),
    }
    let c = node_named(&s, "Centered").transform.to_matrix();
    // Rotation about (1,0,0) by 90°: origin maps to (1,-1,0).
    assert!(
        (c[0][3] - 1.0).abs() < 1e-4 && (c[1][3] + 1.0).abs() < 1e-4,
        "{c:?}"
    );
    assert!(
        matches!(node_named(&s, "Plain").transform, Transform::Trs { scale, .. } if scale == [2.0; 3])
    );
}

fn material_under<'a>(s: &'a Scene3D, name: &str) -> &'a oxideav_mesh3d::Material {
    let p = prim_under(s, name);
    &s.materials[p.material.unwrap().0 as usize]
}

#[test]
fn materials_and_textures() {
    use oxideav_mesh3d::{AlphaMode, ImageData, MagFilter, MinFilter, WrapMode};
    let s = decode("materials.x3d");
    s.validate().unwrap();
    let pbr = material_under(&s, "PBR");
    assert_eq!(pbr.base_color, [0.5, 0.6, 0.7, 0.8]);
    assert_eq!((pbr.metallic, pbr.roughness), (0.25, 0.75));
    assert_eq!(pbr.emissive_factor, [0.1, 0.0, 0.0]);
    assert_eq!(pbr.normal_scale, 0.5);
    assert_eq!(pbr.alpha_mode, AlphaMode::Mask { cutoff: 0.3 });
    let base = &s.textures[pbr.base_color_texture.unwrap().texture.0 as usize];
    assert!(
        matches!(&base.image, ImageData::External { uri, mime } if uri == "textures/base.png" && mime.as_deref() == Some("image/png"))
    );
    assert_eq!(base.sampler.wrap_s, WrapMode::ClampToEdge);
    let normal = &s.textures[pbr.normal_texture.unwrap().texture.0 as usize];
    assert_eq!(normal.sampler.wrap_s, WrapMode::MirroredRepeat);
    assert_eq!(normal.sampler.mag_filter, Some(MagFilter::Nearest));
    assert_eq!(normal.sampler.min_filter, Some(MinFilter::LinearMipLinear));
    assert!(pbr.metallic_roughness_texture.is_some());
    assert!(pbr.occlusion_texture.is_some());
    assert!(pbr.emissive_texture.is_some());
    // Shared appearance → shared material.
    let reuse = prim_under(&s, "Reuse").material;
    assert_eq!(reuse, prim_under(&s, "PBR").material);

    let phong = material_under(&s, "Phong");
    assert_eq!(phong.metallic, 0.0);
    assert!((phong.roughness - 0.2).abs() < 1e-6);
    assert_eq!(phong.base_color[3], 0.5);
    assert_eq!(phong.alpha_mode, AlphaMode::Blend);
    let px = &s.textures[phong.base_color_texture.unwrap().texture.0 as usize];
    match &px.image {
        ImageData::Source(src) => {
            assert_eq!(src.mime(), Some("image/png"));
            let mut r = src.open().unwrap();
            let mut b = Vec::new();
            std::io::Read::read_to_end(&mut r, &mut b).unwrap();
            assert_eq!(&b[1..4], b"PNG");
        }
        _ => panic!("PixelTexture should embed a PNG"),
    }
    // Texture transform baked: scale 2 → uv (2, 1-2).
    let pp = prim_under(&s, "Phong");
    let i = pp
        .positions
        .iter()
        .position(|p| *p == [1.0, 1.0, 0.0])
        .unwrap();
    assert_eq!(pp.uvs[0][i], [2.0, -1.0]);

    let unlit = material_under(&s, "Unlit");
    assert!(unlit.ext.unlit);
    assert_eq!(&unlit.base_color[..3], &[0.0, 1.0, 0.0]);
    let tex = &s.textures[unlit.base_color_texture.unwrap().texture.0 as usize];
    assert!(matches!(&tex.image, ImageData::Source(_)));

    let none = material_under(&s, "NoApp");
    assert!(none.ext.unlit);
    assert_eq!(none.base_color, [1.0; 4]);

    let two = material_under(&s, "TwoSided");
    assert!(two.double_sided);
    assert_eq!(&two.base_color[..3], &[1.0, 0.0, 1.0]);
}

#[test]
fn animations_from_routes() {
    use oxideav_mesh3d::{AnimationProperty, AnimationValues};
    let s = decode("animation.x3d");
    s.validate().unwrap();
    assert_eq!(s.animations.len(), 1);
    let a = &s.animations[0];
    assert_eq!(a.name.as_deref(), Some("Clock"));
    let mover = s
        .nodes
        .iter()
        .position(|n| n.name.as_deref() == Some("Mover"))
        .unwrap() as u32;
    let tr = a
        .channels
        .iter()
        .find(|c| c.target.node.0 == mover && c.target.property == AnimationProperty::Translation)
        .unwrap();
    assert_eq!(tr.sampler.keyframes, vec![0.0, 2.0, 4.0]);
    assert!(matches!(&tr.sampler.values, AnimationValues::Vec3(v) if v[1] == [0.0, 2.0, 0.0]));
    assert!(a
        .channels
        .iter()
        .any(|c| c.target.node.0 == mover && c.target.property == AnimationProperty::Scale));
    let rot = a
        .channels
        .iter()
        .find(|c| c.target.property == AnimationProperty::Rotation)
        .unwrap();
    match &rot.sampler.values {
        AnimationValues::Quat(q) => {
            // Hemisphere-continuous: consecutive dot products >= 0.
            for w in q.windows(2) {
                let d: f32 = (0..4).map(|k| w[0][k] * w[1][k]).sum();
                assert!(d >= 0.0);
            }
        }
        _ => panic!(),
    }
    // Morph targets from the CoordinateInterpolator.
    let morph = a
        .channels
        .iter()
        .find(|c| c.target.property == AnimationProperty::MorphWeights)
        .unwrap();
    let node = s.node(morph.target.node).unwrap();
    let mesh = s.mesh(node.mesh.unwrap()).unwrap();
    assert_eq!(mesh.primitives[0].targets.len(), 2);
    let t1 = mesh.primitives[0].targets[1].position.as_ref().unwrap();
    assert!(t1.iter().all(|d| d[2] == 1.0));
    // The TouchSensor route is reported as unmapped.
    assert!(s.extras["x3d:routes"].to_string().contains("touchTime"));
}

#[test]
fn hanim_skinning() {
    use oxideav_mesh3d::AnimationProperty;
    let s = decode("hanim.x3d");
    s.validate().unwrap();
    assert_eq!(s.skeletons.len(), 1);
    let sk = &s.skeletons[0];
    // Two joints + the humanoid root slot.
    assert_eq!(sk.joints.len(), 3);
    // Arm joint rest transform translates by +1 Y → IBM translates by -1.
    assert!((sk.inverse_bind_matrices[1][1][3] + 1.0).abs() < 1e-5);
    let skinned = s.nodes.iter().find(|n| n.skin.is_some()).unwrap();
    let p = &s.mesh(skinned.mesh.unwrap()).unwrap().primitives[0];
    let w = p.weights.as_ref().unwrap();
    let j = p.joints.as_ref().unwrap();
    for (k, pos) in p.positions.iter().enumerate() {
        let sum: f32 = w[k].iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        if *pos == [5.0, 5.0, 5.0] {
            // Unbound vertex rides the humanoid slot.
            assert_eq!(j[k][0], 2);
        }
        if *pos == [1.0, 2.0, 0.0] {
            // root weight 1 + arm weight 0.5 → 2/3, 1/3.
            assert!((w[k][0] - 2.0 / 3.0).abs() < 1e-5);
        }
    }
    assert_eq!(s.animations.len(), 1);
    assert_eq!(
        s.animations[0].channels[0].target.property,
        AnimationProperty::Rotation
    );
}

#[test]
fn prototypes_expand() {
    let path = format!(
        "{}/tests/fixtures/document_features.x3d",
        env!("CARGO_MANIFEST_DIR")
    );
    let s = X3dDecoder::new()
        .decode(&std::fs::read(path).unwrap())
        .unwrap();
    s.validate().unwrap();
    // The ColoredBox instance becomes a Shape with a blue material
    // and the interface-default 2×2×2 box.
    let cb = node_named(&s, "CB");
    let mesh = s.mesh(cb.mesh.unwrap()).unwrap();
    let mat = &s.materials[mesh.primitives[0].material.unwrap().0 as usize];
    assert_eq!(&mat.base_color[..3], &[0.0, 0.0, 1.0]);
    let max_x = mesh.primitives[0]
        .positions
        .iter()
        .map(|p| p[0])
        .fold(f32::MIN, f32::max);
    assert_eq!(max_x, 1.0);
    // Centimetre units statement.
    assert_eq!(s.unit, oxideav_mesh3d::Unit::Centimetres);
    // Unknown node + Script reported.
    let un = s.extras["x3d:unconverted"].to_string();
    assert!(un.contains("MysteryNode") && un.contains("Script"), "{un}");
    // Translation animation reached T.
    assert_eq!(s.animations.len(), 1);
}

#[test]
fn recursive_proto_terminates() {
    let x = br#"<X3D><Scene>
      <ProtoDeclare name='R'><ProtoInterface/><ProtoBody>
        <Group><ProtoInstance name='R'/><Shape><Box/></Shape></Group>
      </ProtoBody></ProtoDeclare>
      <ProtoInstance name='R'/>
    </Scene></X3D>"#;
    let s = X3dDecoder::new().decode(x).unwrap();
    assert!(s.nodes.len() < 200);
}

#[test]
fn metadata_and_environment_extras() {
    let x = br#"<X3D><Scene>
      <WorldInfo title='Hello' info='"a" "b"'/>
      <NavigationInfo type='"EXAMINE" "ANY"'/>
      <Transform DEF='M'>
        <MetadataSet containerField='metadata' name='info'>
          <MetadataString containerField='value' name='author' value='"me"'/>
          <MetadataFloat containerField='value' name='weights' value='1 2.5'/>
        </MetadataSet>
      </Transform>
      <Inline DEF='Inl' url='"part.x3d"'/>
    </Scene></X3D>"#;
    let s = X3dDecoder::new().decode(x).unwrap();
    let env = s.extras["x3d:environment"].to_string();
    assert!(env.contains("Hello") && env.contains("EXAMINE"));
    let m = node_named(&s, "M");
    let md = m.extras["x3d:metadata"].to_string();
    assert!(md.contains("author") && md.contains("2.5"), "{md}");
    assert!(node_named(&s, "Inl").extras["x3d:inline"]
        .to_string()
        .contains("part.x3d"));
}

#[test]
fn inline_resolver_splices() {
    use std::sync::Arc;
    let outer = br#"<X3D><Scene><Inline DEF='Inl' url='"child.x3d"'/></Scene></X3D>"#;
    let child =
        br#"<X3D><Scene><Transform DEF='Kid'><Shape><Box/></Shape></Transform></Scene></X3D>"#
            .to_vec();
    let s = X3dDecoder::new()
        .with_inline_resolver(Arc::new(move |u: &str| {
            (u == "child.x3d").then(|| child.clone())
        }))
        .decode(outer)
        .unwrap();
    s.validate().unwrap();
    assert_eq!(s.roots.len(), 1);
    let inl = node_named(&s, "Inl");
    assert_eq!(
        s.node(inl.children[0]).unwrap().name.as_deref(),
        Some("Kid")
    );
    // Self-inlining terminates.
    let selfref = br#"<X3D><Scene><Inline url='"me.x3d"'/></Scene></X3D>"#.to_vec();
    let again = selfref.clone();
    let s = X3dDecoder::new()
        .with_inline_resolver(Arc::new(move |_u: &str| Some(again.clone())))
        .decode(&selfref)
        .unwrap();
    assert!(s.nodes.len() < 20);
}

#[test]
fn arbitrary_unit_wraps_roots() {
    let x =
        br#"<X3D><head><unit category='length' name='furlong' conversionFactor='201.168'/></head>
      <Scene><Shape><Box/></Shape></Scene></X3D>"#;
    let s = X3dDecoder::new().decode(x).unwrap();
    assert_eq!(s.roots.len(), 1);
    assert_eq!(
        s.node(s.roots[0]).unwrap().name.as_deref(),
        Some("x3d:units")
    );
}
