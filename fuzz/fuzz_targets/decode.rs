//! Decode arbitrary bytes (XML / ClassicVRML / gzip are sniffed) all
//! the way to a Scene3D: parsing, prototype expansion, geometry
//! tessellation, animation / H-Anim binding must never panic, loop or
//! allocate without bound on hostile input.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_x3d::{Limits, X3dDecoder};

fuzz_target!(|data: &[u8]| {
    let limits = Limits {
        max_input_bytes: 1 << 20,
        max_depth: 64,
        max_elements: 1 << 16,
        max_attributes: 256,
        max_nodes: 1 << 16,
        max_proto_depth: 8,
        max_vertices: 1 << 20,
    };
    let _ = X3dDecoder::new()
        .with_limits(limits)
        .with_segments(8)
        .decode_scene(data);
});
