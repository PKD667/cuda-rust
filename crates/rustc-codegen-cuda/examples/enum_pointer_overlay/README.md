# `enum_pointer_overlay`

A `#[repr(C)]` enum whose variants overlay a pointer word and a pointer-sized
integer word:

```rust
#[repr(C)]
enum PointerOrBits {
    Bits(u64),
    Pointer(*const u8),
}
```

Rust lays both payloads over the same eight bytes, so at most one is alive at a
time. The enum slot map stores that word as a real `ptr` slot. The `Pointer`
variant is inserted and extracted as a pointer, so its provenance survives; the
`Bits` variant converts through `inttoptr` on construction and `ptrtoint` on
extraction, so all 64 bits round-trip and no byte store ever reinterprets a
pointer as an integer or the other way around.

This is the shape the NV backend's `Result<_, Failure>` types need. Only an
exact whole-word overlay is admitted; a pointer sharing bytes with a smaller
integer stays refused (see the `error_enum_pointer_overlap` example).

The two kernels cross the host/device ABI in both directions and the host
checks every value, so a reordered field, a lost bit, or a provenance-erased
pointer would be caught at runtime.

Run it with:

```bash
cargo oxide run enum_pointer_overlay
```
