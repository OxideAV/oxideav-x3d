//! ClassicVRML (.x3dv) tests: reading via the shared oxideav-vrml
//! syntax layer, XML ↔ ClassicVRML conversion, scene round trips.

use oxideav_mesh3d::{Mesh3DDecoder, Mesh3DEncoder};
use oxideav_x3d::{parse_document, write_classic, write_xml, X3dDecoder, X3dEncoder};

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/tests/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

#[test]
fn reads_classic_document() {
    let d = parse_document(&fixture("classic_features.x3dv")).unwrap();
    assert_eq!(d.version, "4.0");
    assert_eq!(d.profile, "Interchange");
    assert_eq!(d.components[0].name, "HAnim");
    assert_eq!(d.units[0].conversion_factor, 0.01);
    assert_eq!(d.meta[0].1, "classic_features.x3dv");
    assert_eq!(d.scene.routes.len(), 2);
    assert_eq!(d.protos.len(), 1);
    let wi = d.nodes.iter().find(|n| n.type_name == "WorldInfo").unwrap();
    assert_eq!(wi.get("info").unwrap().as_strings()[1], "b \"quoted\"");
    assert!(d.warnings.is_empty(), "{:?}", d.warnings);
}

#[test]
fn classic_scene_matches_xml_scene() {
    let a = X3dDecoder::new()
        .decode(&fixture("classic_features.x3dv"))
        .unwrap();
    a.validate().unwrap();
    // Same document re-expressed as XML decodes to the same scene.
    let d = parse_document(&fixture("classic_features.x3dv")).unwrap();
    let xml = write_xml(&d);
    let b = X3dDecoder::new().decode(xml.as_bytes()).unwrap();
    assert_eq!(a.nodes.len(), b.nodes.len());
    assert_eq!(a.triangle_count(), b.triangle_count());
    assert_eq!(a.animations.len(), 1);
    assert_eq!(b.animations.len(), 1);
    assert_eq!(a.unit, oxideav_mesh3d::Unit::Centimetres);
    // Proto instance expanded with the overridden colour.
    let cb = a
        .nodes
        .iter()
        .find(|n| n.name.as_deref() == Some("CB"))
        .unwrap();
    let m = &a.materials[a.mesh(cb.mesh.unwrap()).unwrap().primitives[0]
        .material
        .unwrap()
        .0 as usize];
    assert_eq!(&m.base_color[..3], &[0.0, 0.0, 1.0]);
}

#[test]
fn xml_to_classic_to_xml_is_stable() {
    for name in [
        "document_features.x3d",
        "geometry_all.x3d",
        "materials.x3d",
        "hanim.x3d",
    ] {
        let d = parse_document(&fixture(name)).unwrap();
        let classic = write_classic(&d);
        assert!(classic.starts_with("#X3D V4.0 utf8"), "{classic}");
        let d2 =
            parse_document(classic.as_bytes()).unwrap_or_else(|e| panic!("{name}: {e}\n{classic}"));
        let classic2 = write_classic(&d2);
        assert_eq!(classic, classic2, "{name}");
        // Scene-level equivalence.
        let s1 = X3dDecoder::new().decode(&fixture(name)).unwrap();
        let s2 = X3dDecoder::new().decode(classic.as_bytes()).unwrap();
        assert_eq!(s1.triangle_count(), s2.triangle_count(), "{name}");
        assert_eq!(s1.animations.len(), s2.animations.len(), "{name}");
        assert_eq!(s1.skins.len(), s2.skins.len(), "{name}");
    }
}

#[test]
fn classic_encoder_round_trip() {
    let s = X3dDecoder::new().decode(&fixture("materials.x3d")).unwrap();
    let bytes = X3dEncoder::new().with_classic(true).encode(&s).unwrap();
    let text = String::from_utf8(bytes.clone()).unwrap();
    assert!(text.contains("PhysicalMaterial"), "{text}");
    let back = X3dDecoder::new().decode(&bytes).unwrap();
    back.validate().unwrap();
    assert_eq!(back.triangle_count(), s.triangle_count());
    let gz = X3dEncoder::new()
        .with_classic(true)
        .with_gzip(true)
        .encode(&s)
        .unwrap();
    assert_eq!(
        X3dDecoder::new().decode(&gz).unwrap().triangle_count(),
        s.triangle_count()
    );
}

#[test]
fn hostile_classic_input() {
    for c in [
        "#X3D V4.0 utf8\nShape { geometry USE Nope }",
        "#X3D V4.0 utf8\nTransform { children [ Transform { children [",
        "#X3D V4.0 utf8\nPROTO P [] { P {} } P {}",
        "#VRML V2.0 utf8\nShape {}",
        "#X3D V4.0 utf8\nCoordinate { point [ 1 2 x ] }",
    ] {
        let _ = X3dDecoder::new().decode(c.as_bytes());
    }
    let deep = format!("#X3D V4.0 utf8\n{}", "Group { children [ ".repeat(5000));
    assert!(X3dDecoder::new().decode(deep.as_bytes()).is_err());
}
