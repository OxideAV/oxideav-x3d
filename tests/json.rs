//! JSON encoding (ISO/IEC 19776-5 draft) reader tests.

use oxideav_mesh3d::Mesh3DDecoder;
use oxideav_x3d::document::Encoding;
use oxideav_x3d::{parse_document, X3dDecoder};

fn fixture() -> Vec<u8> {
    std::fs::read(format!(
        "{}/tests/fixtures/json_features.x3dj",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

#[test]
fn reads_json_document() {
    let d = parse_document(&fixture()).unwrap();
    assert_eq!(d.encoding, Encoding::Json);
    assert_eq!(d.profile, "Immersive");
    assert_eq!(d.meta[0].1, "json_features.x3dj");
    assert_eq!(d.components[0].level, 1);
    assert_eq!(d.scene.routes.len(), 2);
    let wi = d.nodes.iter().find(|n| n.type_name == "WorldInfo").unwrap();
    assert_eq!(
        wi.get("info").unwrap().as_strings(),
        &["first", "say \"hi\""]
    );
    let s = d.node(d.find_def("S").unwrap()).unwrap();
    assert!(s.source_text.as_deref().unwrap().contains("initialize"));
    assert!(d.warnings.is_empty(), "{:?}", d.warnings);
}

#[test]
fn json_scene_conversion() {
    let s = X3dDecoder::new().decode(&fixture()).unwrap();
    s.validate().unwrap();
    let t = s
        .nodes
        .iter()
        .find(|n| n.name.as_deref() == Some("T"))
        .unwrap();
    assert_eq!(t.children.len(), 2);
    let ti = s
        .nodes
        .iter()
        .find(|n| n.name.as_deref() == Some("TI"))
        .unwrap();
    let m = &s.materials[s.mesh(ti.mesh.unwrap()).unwrap().primitives[0]
        .material
        .unwrap()
        .0 as usize];
    assert_eq!(&m.base_color[..3], &[0.0, 1.0, 0.0]);
    assert_eq!(s.animations.len(), 1);
}

#[test]
fn hostile_json() {
    for c in [
        &b"{}"[..],
        b"{\"X3D\": 3}",
        b"{\"X3D\": {\"Scene\": {\"-children\": [1, \"x\", {\"Box\": 2}]}}}",
        b"{\"X3D\": {\"Scene\": {\"-children\": [{\"Coordinate\": {\"@point\": [\"a\", null]}}]}}}",
    ] {
        let _ = X3dDecoder::new().decode(c);
    }
    let deep = format!(
        "{{\"X3D\":{{\"Scene\":{}{}}}}}",
        "{\"-children\":[{\"Group\":".repeat(200),
        "}]}".repeat(200)
    );
    assert!(X3dDecoder::new().decode(deep.as_bytes()).is_err());
}
