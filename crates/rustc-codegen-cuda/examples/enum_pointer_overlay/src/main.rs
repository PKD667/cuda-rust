/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! A pointer word and a whole pointer-sized integer word sharing enum bytes.
//!
//! `PointerOrBits` lays `Bits(u64)` and `Pointer(*const u8)` over the same
//! eight bytes, so at most one is alive at a time. The enum slot map used to
//! refuse the shape ("refusing to erase LLVM pointer provenance"), which is
//! what stood in the way of the NV backend's `Result<_, Failure>` types.
//!
//! The overlay stores the shared bytes as a real `ptr` slot. The `Pointer`
//! variant is inserted/extracted as a pointer (provenance intact); the `Bits`
//! variant converts through `inttoptr` on construction and `ptrtoint` on
//! extraction (all 64 bits exact). No byte store ever reinterprets a pointer as
//! an integer or the other way around.
//!
//! The two kernels cross the host/device ABI in both directions, which is the
//! strongest check that rustc's bytes and the device's slot agree:
//!
//! | Kernel | Direction      | Exercises                                        |
//! |--------|----------------|--------------------------------------------------|
//! | decode | host -> device | match plus payload reads over host-written bytes |
//! | encode | device -> host | construct plus whole-enum stores, host reads     |
//!
//! The pointer variant only ever moves its bits; nothing dereferences the
//! address, so the test is about layout and provenance-preserving movement.
//!
//! Run with:
//!   cargo oxide run enum_pointer_overlay

use cuda_core::{CudaContext, DeviceBuffer, DeviceCopy, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[repr(C)]
#[derive(Clone, Copy)]
pub enum PointerOrBits {
    Bits(u64),
    Pointer(*const u8),
}

// Sound because the device slot map is byte-identical to rustc's layout (which
// is exactly what this example proves end-to-end) and the pointer variant never
// crosses into a dereference on the device.
unsafe impl DeviceCopy for PointerOrBits {}

#[cuda_module]
mod kernels {
    use super::*;

    /// host -> device: decode each host-written enum's payload word.
    #[kernel]
    pub fn decode(input: &[PointerOrBits], mut out: DisjointSlice<u64>) {
        let index = thread::index_1d();
        if let (Some(value), Some(out_elem)) = (input.get(index.get()), out.get_mut(index)) {
            *out_elem = match *value {
                PointerOrBits::Bits(bits) => bits,
                PointerOrBits::Pointer(pointer) => pointer as u64,
            };
        }
    }

    /// device -> host: construct each variant from an integer and store the
    /// whole enum for the host to read.
    #[kernel]
    pub fn encode(src: &[u64], mut out: DisjointSlice<PointerOrBits>) {
        let index = thread::index_1d();
        if let (Some(&value), Some(out_elem)) = (src.get(index.get()), out.get_mut(index)) {
            *out_elem = if value & 1 == 0 {
                PointerOrBits::Bits(value)
            } else {
                PointerOrBits::Pointer(value as *const u8)
            };
        }
    }
}

/// Host-side mirror of `decode`.
fn expected_decode(value: &PointerOrBits) -> u64 {
    match *value {
        PointerOrBits::Bits(bits) => bits,
        PointerOrBits::Pointer(pointer) => pointer as u64,
    }
}

fn main() {
    println!("=== enum_pointer_overlay ===");

    // The premise: rustc overlays the two payloads.
    assert_eq!(
        std::mem::size_of::<PointerOrBits>(),
        std::mem::size_of::<u64>() + std::mem::size_of::<u64>(),
        "rustc layout: tag plus one overlaid pointer-sized word"
    );

    let ctx = CudaContext::new(0).expect("context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load embedded module");
    let cfg = LaunchConfig::for_num_elems(64);

    let mut failures = 0usize;

    // ---- host -> device: the kernel decodes host-written enum bytes ----
    let host_input: Vec<PointerOrBits> = (0..64u64)
        .map(|i| {
            if i % 2 == 0 {
                PointerOrBits::Bits(0xA5A5_0000_0000_0000 | i)
            } else {
                PointerOrBits::Pointer((0x1000_0000_0000 | i) as *const u8)
            }
        })
        .collect();
    let device_input = DeviceBuffer::from_host(&stream, &host_input).unwrap();
    let mut device_decoded = DeviceBuffer::<u64>::zeroed(&stream, 64).unwrap();
    // SAFETY: the launch is one-dimensional and both buffers hold one element
    // per launched thread.
    unsafe { module.decode(&stream, cfg, &device_input, &mut device_decoded) }
        .expect("decode launch failed");
    let decoded = device_decoded.to_host_vec(&stream).unwrap();
    for (i, (&got, want)) in decoded
        .iter()
        .zip(host_input.iter().map(expected_decode))
        .enumerate()
    {
        if got != want {
            println!("FAIL decode i={i}: got {got:#x}, want {want:#x}");
            failures += 1;
        }
    }

    // ---- device -> host: the host decodes device-constructed enums ----
    let host_src: Vec<u64> = (0..64u64)
        .map(|i| if i % 2 == 0 { i } else { 0x2000_0000_0000 | i })
        .collect();
    let device_src = DeviceBuffer::from_host(&stream, &host_src).unwrap();
    let mut device_encoded = DeviceBuffer::<PointerOrBits>::zeroed(&stream, 64).unwrap();
    // SAFETY: as the decode launch above.
    unsafe { module.encode(&stream, cfg, &device_src, &mut device_encoded) }
        .expect("encode launch failed");
    let encoded = device_encoded.to_host_vec(&stream).unwrap();
    for (i, value) in encoded.iter().enumerate() {
        let source = host_src[i];
        let ok = match (source % 2, value) {
            (0, PointerOrBits::Bits(bits)) => *bits == source,
            (1, PointerOrBits::Pointer(pointer)) => *pointer as u64 == source,
            _ => false,
        };
        if !ok {
            println!("FAIL encode i={i}: variant/payload mismatch for {source:#x}");
            failures += 1;
        }
    }

    if failures == 0 {
        println!("PASS: enum_pointer_overlay ({cfg:?}; both ABI directions bit-exact)");
    } else {
        println!("enum_pointer_overlay: {failures} FAILURES");
        std::process::exit(1);
    }
}
