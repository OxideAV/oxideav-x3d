//! Whatever decodes must re-encode (XML and ClassicVRML) and decode
//! again without error.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_x3d::{Limits, X3dDecoder, X3dEncoder};

fuzz_target!(|data: &[u8]| {
    let limits = Limits {
        max_input_bytes: 1 << 20,
        max_depth: 48,
        max_elements: 1 << 15,
        max_attributes: 256,
        max_nodes: 1 << 15,
        max_proto_depth: 6,
        max_vertices: 1 << 18,
    };
    let dec = X3dDecoder::new().with_limits(limits).with_segments(6);
    let Ok(scene) = dec.decode_scene(data) else {
        return;
    };
    let relaxed = X3dDecoder::new().with_segments(6);
    for classic in [false, true] {
        let bytes = X3dEncoder::new()
            .with_classic(classic)
            .encode_scene(&scene)
            .expect("encoding a decoded scene must succeed");
        relaxed
            .decode_scene(&bytes)
            .expect("encoder output must decode");
    }
});
