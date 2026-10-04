//! Document-layer tests: XML reading, typed values, DEF/USE, protos,
//! routes, unknown-node preservation and XML write → read round trips.

use oxideav_x3d::document::NodeKind;
use oxideav_x3d::{parse_document, write_xml, FieldType, X3dDocument};

fn load(name: &str) -> X3dDocument {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(path).unwrap();
    parse_document(&bytes).unwrap()
}

fn summarise(doc: &X3dDocument) -> String {
    // Re-serialise: equal output implies an equal reachable graph.
    write_xml(doc)
}

#[test]
fn header_and_statements() {
    let d = load("document_features.x3d");
    assert_eq!(d.version, "4.0");
    assert_eq!(d.profile, "Immersive");
    assert_eq!(d.components[0].name, "Scripting");
    assert_eq!(d.units[0].conversion_factor, 0.01);
    assert_eq!(d.meta.len(), 2);
    assert_eq!(d.protos.len(), 2);
    assert!(d.protos[1].body.is_none());
    assert_eq!(d.protos[1].url.len(), 2);
    assert_eq!(d.scene.routes.len(), 2);
    assert!(d.warnings.is_empty(), "{:?}", d.warnings);
}

#[test]
fn typed_values_and_sharing() {
    let d = load("document_features.x3d");
    let t = d.node(d.find_def("T").unwrap()).unwrap();
    assert_eq!(
        t.get("translation").unwrap().as_tuple::<3>(),
        Some([1.0, 2.0, 3.0])
    );
    assert_eq!(t.get("translation").unwrap().ty, FieldType::SFVec3f);
    // Default from the node table.
    assert_eq!(t.value("scale").unwrap().as_tuple::<3>(), Some([1.0; 3]));
    let wi = d.nodes.iter().find(|n| n.type_name == "WorldInfo").unwrap();
    assert_eq!(
        wi.get("info").unwrap().as_strings(),
        &["line one", "say \"hi\""]
    );
    // USE shares the arena slot.
    let s = d.find_def("S").unwrap();
    let refs = d
        .nodes
        .iter()
        .filter(|n| n.all_children().any(|c| c == s))
        .count();
    assert_eq!(refs, 2);
    // Proto instance resolved, with typed fieldValue.
    let cb = d.node(d.find_def("CB").unwrap()).unwrap();
    assert_eq!(cb.kind, NodeKind::ProtoInstance { proto: Some(0) });
    assert_eq!(cb.get("boxColor").unwrap().ty, FieldType::SFColor);
    // Unknown node preserved with raw attributes and child container.
    let m = d
        .nodes
        .iter()
        .find(|n| n.type_name == "MysteryNode")
        .unwrap();
    assert_eq!(m.kind, NodeKind::Unknown);
    assert_eq!(m.get("foo").unwrap().as_str(), Some("bar"));
    assert_eq!(m.children_of("thing").len(), 1);
    // Script decls + CDATA.
    let sc = d.node(d.find_def("SC").unwrap()).unwrap();
    assert_eq!(sc.decls.len(), 2);
    assert!(sc.source_text.as_deref().unwrap().contains("count++"));
    assert_eq!(sc.value("count").unwrap().as_i32(), Some(3));
}

#[test]
fn xml_round_trip_is_stable() {
    let d = load("document_features.x3d");
    let a = summarise(&d);
    let d2 = parse_document(a.as_bytes()).unwrap();
    assert!(d2.warnings.is_empty(), "{:?}\n{a}", d2.warnings);
    let b = summarise(&d2);
    assert_eq!(a, b);
    assert!(
        a.contains("<Shape USE='S'/>") || a.contains("<Shape USE=\"S\"/>"),
        "{a}"
    );
    assert!(a.contains("containerField=\"thing\""));
}

#[test]
fn gzip_input() {
    let path = format!(
        "{}/tests/fixtures/document_features.x3d",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(path).unwrap();
    let gz = oxideav_x3d::gzip(&bytes).unwrap();
    let d = parse_document(&gz).unwrap();
    assert_eq!(d.scene.routes.len(), 2);
}

#[test]
fn hostile_inputs_do_not_panic() {
    let cases: &[&[u8]] = &[
        b"",
        b"<X3D>",
        b"<X3D><Scene><Shape USE='nope'/></Scene></X3D>",
        b"<X3D><Scene><Transform DEF='a'><Transform USE='a'/></Transform></Scene></X3D>",
        b"<X3D><Scene><Coordinate point='1 2 x'/></Scene></X3D>",
        b"<X3D><Scene><PixelTexture image='100000 100000 4'/></Scene></X3D>",
        b"<NotX3D/>",
        b"\x1f\x8b\x08garbage",
        &[0xff, 0xfe, 0x3c, 0x00],
    ];
    for c in cases {
        let _ = parse_document(c);
    }
    // Cyclic USE survives a write.
    let d = parse_document(
        b"<X3D><Scene><Transform DEF='a'><Transform USE='a'/></Transform></Scene></X3D>",
    )
    .unwrap();
    let out = write_xml(&d);
    assert!(out.contains("USE"));
}
