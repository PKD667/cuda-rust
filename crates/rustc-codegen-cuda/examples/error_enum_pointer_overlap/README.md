# `error_enum_pointer_overlap`

This negative test protects LLVM pointer provenance when a pointer word shares
bytes with a *smaller* integer.

```rust
#[repr(C)]
enum PointerOrPartialBits {
    Bits(u32),
    Pointer(*const u8),
}
```

Rust lays the `u32` over the low half of the pointer word. A whole-word
pointer/integer overlay is representable (the shared bytes become a real `ptr`
and the integer side converts through `inttoptr`/`ptrtoint`; see the
`enum_pointer_overlay` example), but a *partial* overlap is not: whichever type
the shared bytes take, one variant loses either pointer provenance or integer
bits. cuda-oxide therefore fails closed.

Run it with:

```bash
cargo oxide build error_enum_pointer_overlap
```

The build must fail with a diagnostic containing:

```text
overlapping pointer and non-identical storage
```
