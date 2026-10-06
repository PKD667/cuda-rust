/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: a pointer sharing bytes with a SMALLER integer stays refused.
//!
//! A pointer word and a whole pointer-sized integer word that share bytes can
//! now be overlaid (see the `enum_pointer_overlay` example): the shared bytes
//! are stored as a real `ptr`, and the integer side crosses the boundary with
//! `inttoptr`/`ptrtoint`. That representation only works when the two sides
//! cover exactly the same bytes. Here `Bits(u32)` covers only the low half of
//! `Pointer(*const u8)`, so whichever way the slot is typed, one variant loses
//! either pointer provenance or four integer bits. Lowering therefore still
//! fails closed.

use cuda_device::kernel;

#[repr(C)]
#[derive(Clone, Copy)]
pub enum PointerOrPartialBits {
    Bits(u32),
    Pointer(*const u8),
}

#[inline(never)]
fn value_bits(value: PointerOrPartialBits) -> u64 {
    match value {
        PointerOrPartialBits::Bits(bits) => bits as u64,
        PointerOrPartialBits::Pointer(pointer) => pointer as u64,
    }
}

/// # Safety
///
/// `input` must point to one initialized `PointerOrPartialBits`, and `out` must
/// point to one writable `u64`. Neither location may be raced by another thread.
#[kernel]
pub unsafe fn enum_pointer_overlap(input: *const PointerOrPartialBits, out: *mut u64) {
    let value = unsafe { input.read() };
    unsafe {
        out.write(value_bits(value));
    }
}

fn main() {
    // Keep both variants reachable in the host crate as well as device MIR.
    let values = [
        PointerOrPartialBits::Bits(7),
        PointerOrPartialBits::Pointer(core::ptr::null()),
    ];
    println!(
        "This negative example should fail during device compilation ({} variants).",
        values.len()
    );
}
