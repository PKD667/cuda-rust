/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Byte comparison: the one memory intrinsic with no NVVM counterpart.

/// Compare `n` bytes at `a` and `b` as `memcmp` does: zero when they are equal, otherwise the sign
/// of the first pair that differs.
///
/// `core::intrinsics::compare_bytes` has no device lowering of its own. `llvm.memcmp` is not enough
/// for it: that intrinsic only promises zero or non-zero, while Rust's `[u8]::cmp` and `str::cmp`
/// read the sign of the result, so lowering to it would silently answer an ordering question
/// wrongly. The loop is therefore the lowering, and it is reached by name: the collector enqueues
/// this function where device code calls the intrinsic, and the importer calls it under the fixed
/// symbol the collector exported it as.
///
/// # Safety
///
/// Both pointers must be readable for `n` bytes, and the regions must not overlap the destination of
/// anything the caller writes while this runs.
#[cuda_macros::device]
pub unsafe fn memcmp(a: *const u8, b: *const u8, n: usize) -> i32 {
    let mut i = 0usize;
    while i < n {
        let (left, right) = unsafe { (*a.add(i), *b.add(i)) };
        if left != right {
            return i32::from(left) - i32::from(right);
        }
        i += 1;
    }
    0
}
