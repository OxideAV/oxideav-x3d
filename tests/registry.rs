//! Registry wiring (only with the default `registry` feature).
#![cfg(feature = "registry")]

use oxideav_mesh3d::Mesh3DRegistry;

#[test]
fn registers_decoder_and_encoders() {
    let mut r = Mesh3DRegistry::new();
    oxideav_x3d::register(&mut r);
    for ext in ["x3d", "X3DZ", "x3dv", "x3dvz"] {
        assert!(r.decoder_for_extension(ext).is_some(), "{ext}");
    }
    let mut enc = r.encoder_for_extension("x3dz").unwrap();
    let mut s = oxideav_mesh3d::Scene3D::new();
    let n = s.add_node(oxideav_mesh3d::Node::new().with_name("Empty"));
    s.add_root(n);
    let bytes = enc.encode(&s).unwrap();
    let back = r.decoder_for_format("x3d").unwrap().decode(&bytes).unwrap();
    assert_eq!(back.nodes[0].name.as_deref(), Some("Empty"));
    let classic = r.encoder_for_extension("x3dv").unwrap().encode(&s).unwrap();
    assert!(classic.starts_with(b"#X3D V4.0 utf8"));
    let back = r
        .decoder_for_extension("x3dv")
        .unwrap()
        .decode(&classic)
        .unwrap();
    assert_eq!(back.nodes[0].name.as_deref(), Some("Empty"));
}
