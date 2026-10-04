//! Encoder tests: Scene3D → X3D XML → Scene3D round trips.

use std::sync::Arc;

use oxideav_mesh3d::{
    AlphaMode, Animation, AnimationChannel, AnimationProperty, AnimationSampler, AnimationValues,
    Camera, ImageData, InMemoryAsset, Indices, Interpolation, Light, Material, Mesh, Mesh3DDecoder,
    Mesh3DEncoder, MorphTarget, Node, Primitive, Scene3D, Texture, TextureRef, Topology, Transform,
    Unit, WrapMode,
};
use oxideav_x3d::{X3dDecoder, X3dEncoder};

fn roundtrip(s: &Scene3D) -> (String, Scene3D) {
    let bytes = X3dEncoder::new().encode(s).unwrap();
    let text = String::from_utf8(bytes.clone()).unwrap();
    let back = X3dDecoder::new()
        .decode(&bytes)
        .unwrap_or_else(|e| panic!("{e}\n{text}"));
    back.validate().unwrap_or_else(|e| panic!("{e:?}\n{text}"));
    (text, back)
}

fn named<'a>(s: &'a Scene3D, n: &str) -> &'a Node {
    s.nodes
        .iter()
        .find(|x| x.name.as_deref() == Some(n))
        .unwrap_or_else(|| panic!("no node {n}"))
}

fn quad_prim() -> Primitive {
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    p.normals = Some(vec![[0.0, 0.0, 1.0]; 4]);
    p.uvs = vec![vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]];
    p.colors = vec![vec![
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.5],
    ]];
    p.indices = Some(Indices::U16(vec![0, 1, 2, 0, 2, 3]));
    p
}

fn build_scene() -> Scene3D {
    let mut s = Scene3D::new();
    s.unit = Unit::Millimetres;
    let tex = {
        let mut t = Texture::from_encoded("image/png", vec![0x89, b'P', b'N', b'G', 1, 2, 3]);
        t.sampler.wrap_s = WrapMode::ClampToEdge;
        t.name = Some("Tex 1".into());
        s.add_texture(t)
    };
    let ext = s.add_texture(Texture::from_uri("images/wood.jpg"));
    let mut m = Material::new().with_name("Gold");
    m.base_color = [1.0, 0.8, 0.2, 0.75];
    m.metallic = 0.9;
    m.roughness = 0.3;
    m.emissive_factor = [0.1, 0.1, 0.0];
    m.alpha_mode = AlphaMode::Blend;
    m.double_sided = true;
    m.base_color_texture = Some(TextureRef::new(tex));
    m.normal_texture = Some(TextureRef::new(ext));
    m.normal_scale = 0.5;
    let mat = s.add_material(m);
    let mut p = quad_prim();
    p.material = Some(mat);
    let mut morph = MorphTarget::default();
    morph.position = Some(vec![[0.0, 0.0, 1.0]; 4]);
    p.targets.push(morph);
    let mut mesh = Mesh::new(Some("Quad".to_string()));
    mesh.primitives.push(p);
    mesh.weights = vec![0.0];
    let mesh = s.add_mesh(mesh);
    // Lines + points.
    let mut lines = Primitive::new(Topology::LineStrip);
    lines.positions = vec![[0.0; 3], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]];
    let mut pts = Primitive::new(Topology::Points);
    pts.positions = vec![[0.0; 3], [2.0, 0.0, 0.0]];
    let mut lm = Mesh::new(Some("Wire".to_string()));
    lm.primitives.push(lines);
    lm.primitives.push(pts);
    let lmesh = s.add_mesh(lm);

    let a = s.add_node(
        Node::new()
            .with_name("A")
            .with_mesh(mesh)
            .with_transform(Transform::Trs {
                translation: [1.0, 2.0, 3.0],
                rotation: [
                    0.0,
                    std::f32::consts::FRAC_1_SQRT_2,
                    0.0,
                    std::f32::consts::FRAC_1_SQRT_2,
                ],
                scale: [2.0, 2.0, 2.0],
            }),
    );
    let b = s.add_node(Node::new().with_name("B").with_mesh(mesh));
    let w = s.add_node(Node::new().with_name("W").with_mesh(lmesh));
    let cam = s.add_camera(Camera::Perspective {
        aspect_ratio: None,
        yfov: 0.8,
        znear: 0.05,
        zfar: Some(500.0),
    });
    let mut cn = Node::new().with_name("Cam");
    cn.camera = Some(cam);
    cn.transform = Transform::Trs {
        translation: [0.0, 1.0, 10.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0; 3],
    };
    let cn = s.add_node(cn);
    let ortho = s.add_camera(Camera::orthographic(3.0, 2.0, 0.1, 50.0));
    let mut on = Node::new().with_name("Ortho");
    on.camera = Some(ortho);
    let on = s.add_node(on);
    let mut lights = Vec::new();
    for (i, l) in [
        Light::Directional {
            color: [1.0, 1.0, 0.9],
            intensity: 2.0,
        },
        Light::Point {
            color: [1.0; 3],
            intensity: 5.0,
            range: Some(20.0),
        },
        Light::Spot {
            color: [0.5; 3],
            intensity: 1.0,
            range: None,
            inner_cone_angle: 0.2,
            outer_cone_angle: 0.5,
        },
    ]
    .into_iter()
    .enumerate()
    {
        let li = s.add_light(l);
        let mut n = Node::new().with_name(format!("Light{i}"));
        n.light = Some(li);
        lights.push(s.add_node(n));
    }
    let mut root = Node::new().with_name("Root");
    root.children = vec![a, b, w, cn, on];
    root.children.extend(lights);
    let root = s.add_node(root);
    s.add_root(root);

    let mut anim = Animation::new(Some("Spin".to_string()));
    anim.channels.push(AnimationChannel::new(
        a,
        AnimationProperty::Translation,
        AnimationSampler {
            keyframes: vec![0.0, 1.0, 2.0],
            values: AnimationValues::Vec3(vec![[0.0; 3], [0.0, 1.0, 0.0], [0.0; 3]]),
            interpolation: Interpolation::Linear,
        },
    ));
    anim.channels.push(AnimationChannel::new(
        a,
        AnimationProperty::Rotation,
        AnimationSampler {
            keyframes: vec![0.0, 2.0],
            values: AnimationValues::Quat(vec![[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]]),
            interpolation: Interpolation::Step,
        },
    ));
    anim.channels.push(AnimationChannel::new(
        b,
        AnimationProperty::MorphWeights,
        AnimationSampler::morph_weights(
            vec![0.0, 2.0],
            vec![vec![0.0], vec![1.0]],
            Interpolation::Linear,
        )
        .unwrap(),
    ));
    s.add_animation(anim);
    s.validate().unwrap();
    s
}

#[test]
fn programmatic_scene_round_trip() {
    let s = build_scene();
    let (text, back) = roundtrip(&s);
    // Optional: dump for external schema validation (xmllint --schema).
    if let Ok(dir) = std::env::var("OXIDEAV_X3D_DUMP_DIR") {
        std::fs::write(format!("{dir}/programmatic.x3d"), &text).unwrap();
    }
    assert!(text.contains("PhysicalMaterial"), "{text}");
    assert!(text.contains("IndexedTriangleSet"));
    assert_eq!(back.unit, Unit::Millimetres);
    // Shared mesh → USE'd Shape → shared decoded mesh.
    let a = named(&back, "A");
    let b = named(&back, "B");
    let shape_a = back.node(a.children[0]).unwrap();
    let shape_b = back.node(b.children[0]).unwrap();
    assert_eq!(shape_a.mesh, shape_b.mesh);
    match a.transform {
        Transform::Trs {
            translation,
            rotation,
            scale,
        } => {
            assert_eq!(translation, [1.0, 2.0, 3.0]);
            assert_eq!(scale, [2.0; 3]);
            assert!((rotation[1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5);
        }
        _ => panic!(),
    }
    let prim = &back.mesh(shape_a.mesh.unwrap()).unwrap().primitives[0];
    assert_eq!(prim.positions.len(), 4);
    assert_eq!(prim.triangle_indices().len(), 2);
    // UVs survive the v flip both ways.
    let i0 = prim
        .positions
        .iter()
        .position(|p| *p == [0.0, 0.0, 0.0])
        .unwrap();
    assert_eq!(prim.uvs[0][i0], [0.0, 1.0]);
    assert_eq!(
        prim.colors[0][prim
            .positions
            .iter()
            .position(|p| *p == [0.0, 1.0, 0.0])
            .unwrap()][3],
        0.5
    );
    let m = &back.materials[prim.material.unwrap().0 as usize];
    assert_eq!(
        m.base_color,
        [1.0, 1.0, 1.0, 0.75],
        "colour node replaces baseColor"
    );
    assert_eq!((m.metallic, m.roughness), (0.9, 0.3));
    assert!(m.double_sided);
    assert_eq!(m.alpha_mode, AlphaMode::Blend);
    assert_eq!(m.normal_scale, 0.5);
    let bt = &back.textures[m.base_color_texture.unwrap().texture.0 as usize];
    assert_eq!(bt.sampler.wrap_s, WrapMode::ClampToEdge);
    match &bt.image {
        ImageData::Source(src) => {
            let mut r = src.open().unwrap();
            let mut v = Vec::new();
            std::io::Read::read_to_end(&mut r, &mut v).unwrap();
            assert_eq!(v, vec![0x89, b'P', b'N', b'G', 1, 2, 3]);
        }
        _ => panic!("embedded texture should round-trip through a data: URI"),
    }
    let nt = &back.textures[m.normal_texture.unwrap().texture.0 as usize];
    assert!(matches!(&nt.image, ImageData::External { uri, .. } if uri == "images/wood.jpg"));
    // Lines and points.
    let w = named(&back, "W");
    assert_eq!(w.children.len(), 2);
    // Cameras and lights.
    let cam = named(&back, "Cam");
    let vp = back.node(cam.children[0]).unwrap();
    match back.cameras[vp.camera.unwrap().0 as usize] {
        Camera::Perspective {
            yfov, znear, zfar, ..
        } => {
            assert_eq!((yfov, znear, zfar), (0.8, 0.05, Some(500.0)))
        }
        _ => panic!(),
    }
    let ortho = named(&back, "Ortho");
    let ov = back.node(ortho.children[0]).unwrap();
    assert!(
        matches!(back.cameras[ov.camera.unwrap().0 as usize], Camera::Orthographic { xmag, ymag, .. } if xmag == 3.0 && ymag == 2.0)
    );
    assert_eq!(back.lights.len(), 3);
    assert!(back.lights.iter().any(|l| matches!(l, Light::Spot { inner_cone_angle, outer_cone_angle, .. } if *inner_cone_angle == 0.2 && *outer_cone_angle == 0.5)));
    // Animations.
    assert_eq!(back.animations.len(), 1);
    let an = &back.animations[0];
    assert_eq!(an.name.as_deref(), Some("Spin"));
    let tr = an
        .channels
        .iter()
        .find(|c| c.target.property == AnimationProperty::Translation)
        .unwrap();
    assert_eq!(tr.sampler.keyframes, vec![0.0, 1.0, 2.0]);
    let rot = an
        .channels
        .iter()
        .find(|c| c.target.property == AnimationProperty::Rotation)
        .unwrap();
    // Step emulated with a duplicated key (nudged to stay increasing).
    assert_eq!(rot.sampler.keyframes.len(), 3);
    assert!(an
        .channels
        .iter()
        .any(|c| c.target.property == AnimationProperty::MorphWeights));
}

#[test]
fn fixtures_round_trip_structurally() {
    for name in [
        "geometry_all.x3d",
        "materials.x3d",
        "animation.x3d",
        "document_features.x3d",
    ] {
        let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
        let first = X3dDecoder::new()
            .decode(&std::fs::read(path).unwrap())
            .unwrap();
        let (text, back) = roundtrip(&first);
        assert_eq!(
            first.triangle_count(),
            back.triangle_count(),
            "{name}\n{text}"
        );
        assert_eq!(first.cameras.len(), back.cameras.len(), "{name}");
        assert_eq!(first.lights.len(), back.lights.len(), "{name}");
        assert_eq!(first.animations.len(), back.animations.len(), "{name}");
        let ch = |s: &Scene3D| s.animations.iter().map(|a| a.channels.len()).sum::<usize>();
        assert_eq!(ch(&first), ch(&back), "{name}");
        // Second generation is textually stable.
        let (text2, _) = roundtrip(&back);
        let (text3, _) = roundtrip(&X3dDecoder::new().decode(text2.as_bytes()).unwrap());
        assert_eq!(text2, text3, "{name}");
    }
}

#[test]
fn gzip_output() {
    let s = build_scene();
    let gz = X3dEncoder::new().with_gzip(true).encode(&s).unwrap();
    assert_eq!(&gz[..2], &[0x1f, 0x8b]);
    let back = X3dDecoder::new().decode(&gz).unwrap();
    assert_eq!(back.triangle_count(), s.triangle_count());
}

#[test]
fn embedded_asset_arc() {
    // InMemoryAsset with no MIME still encodes.
    let mut s = Scene3D::new();
    let t = s.add_texture(Texture::from_source(Arc::new(InMemoryAsset::new(
        None,
        vec![1, 2, 3],
    ))));
    let mut m = Material::new();
    m.base_color_texture = Some(TextureRef::new(t));
    let mat = s.add_material(m);
    let mut p = quad_prim();
    p.material = Some(mat);
    let mut mesh = Mesh::new(None);
    mesh.primitives.push(p);
    let mesh = s.add_mesh(mesh);
    let n = s.add_node(Node::new().with_mesh(mesh));
    s.add_root(n);
    let (text, _) = roundtrip(&s);
    assert!(text.contains("data:application/octet-stream;base64,AQID"));
}
