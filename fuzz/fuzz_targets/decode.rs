//! Decode arbitrary bytes (XML / ClassicVRML / gzip are sniffed) all
//! the way to a Scene3D: parsing, prototype expansion, geometry
//! tessellation, animation / H-Anim binding must never panic, loop or
//! allocate without bound on hostile input.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_x3d::{Limits, X3dDecoder};

fuzz_target!(|data: &[u8]| {
    let limits = Limits::default()
        .with_max_input_bytes(1 << 20)
        .with_max_depth(64)
        .with_max_elements(1 << 16)
        .with_max_attributes(256)
        .with_max_nodes(1 << 16)
        .with_max_proto_depth(8)
        .with_max_vertices(1 << 20);
    let _ = X3dDecoder::new()
        .with_limits(limits)
        .with_segments(8)
        .decode_scene(data);
});
