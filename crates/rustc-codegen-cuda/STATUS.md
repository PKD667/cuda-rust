# rustc-codegen-cuda: error example status

Examples named `error*` fall into two kinds:

- **diagnostics-fixture**: intentional negative test. The compiler is supposed
  to reject this. It is not a gap.
- **support-gap**: a real Rust feature that the compiler does not yet handle.
  Kept as an expected-failure regression test until it is implemented.

When adding a new `error*` example, update this table and the
`ERROR_EXAMPLES` array in `scripts/smoketest.sh` in the same commit.
Run `scripts/check-error-example-status.sh` to verify both are in sync.

| Example                               | Kind                | Fails at                            |
| :------------------------------------ | :------------------ | :---------------------------------- |
| `error`                               | diagnostics-fixture | `core::fmt` reachable from device   |
| `error_enum_bool_payload_addr`        | diagnostics-fixture | `&mut` to a canonical-byte payload  |
| `error_enum_pointer_overlap`          | support-gap         | Partial pointer/integer word overlap|
| `error_enum_shared_pointer_layout`    | support-gap         | AS3 pointer arrays/vectors in enums |
| `error_generated_intrinsic_abi`       | diagnostics-fixture | Unsupported raw intrinsic ABI       |
| `error_generated_intrinsic_callable`  | diagnostics-fixture | Raw intrinsic passed through `Fn`   |
| `error_generated_intrinsic_fn_pointer`| diagnostics-fixture | Raw intrinsic made into `fn` pointer|
| `error_generated_intrinsic_unknown_id`| diagnostics-fixture | Unknown ID in a supported ABI       |
| `error_heap_alloc`                    | diagnostics-fixture | `__rust_alloc` reachable (#108)     |
| `error_missing_device_attr`           | diagnostics-fixture | `thread::index_*` stub (#76)        |
| `error_set_discriminant_uninhabited`  | diagnostics-fixture | Invalid enum variant selection      |

Drops whose monomorphized glue is provably a no-op (e.g. the
`core::array::IntoIter` behind `for x in arr` with Copy elements) lower
to a plain branch since issue #138. When the no-op proof fails, the drop
lowers to a device-side call to the monomorphized `drop_in_place::<T>`
shim, which the collector gathers and the pipeline compiles like any
other device function; see the `drop_glue` example. This covers direct
`impl Drop` types reachable from kernels. The shim and every
`Drop::drop` body it reaches go through the same device pipeline, so
drop glue whose MIR uses constructs the pipeline cannot yet translate
(e.g. slice/`Vec` element drops, panic formatting) still fails
compilation with a diagnostic. Destructors are never silently skipped:
a drop is either proven unobservable or its call is emitted.

`error_heap_alloc` is the diagnostic for a kernel that allocates when the program registers no
device allocator. A program registers one with `#[device_allocator] static HEAP: A = A;`
(`cuda_device::device_allocator`, `A: GlobalAlloc`; the using crate enables the nightly feature
`ptr_alignment_type`). The collector then serves `__rust_alloc`, `__rust_alloc_zeroed`,
`__rust_dealloc` and `__rust_realloc` from the four `#[device]` functions the attribute generates,
`handle_alloc_error` traps, and `Vec`, `Box` and `vec![]` compile. The attribute is not
`#[global_allocator]`: that is the allocator of the host program too. Lowering `Vec` needed
`ptr::write_bytes` (`mir.memset`) and `size_of_val`/`align_of_val` for sized types. A `&str` or `&[u8]` constant lowers too: the pliron type is a slice, so the constant is
materialized as a slice of a promoted immutable global whose bytes come from the allocation its
provenance names, with the length read from the fat pointer. A slice constant whose element is wider
than a byte is refused by name. `compare_bytes` is served by a loop in `cuda_device::mem`, which
the collector enqueues where device code calls the intrinsic: `llvm.memcmp` only promises zero or
non-zero and Rust reads the sign, so an intrinsic is not a lowering for it. A function's address is real. `let f: fn(u32) -> u32 = inc;` used to hash the function's mangled
name into an integer and cast that to a pointer: the value compared and stored, but it was not an
address, and calling through it was refused because no definition existed under any name the call
could reach. `mir.func_addr` now names the function, under the symbol a direct call to it names, so
the two reach the same definition; `mir.call_indirect` calls through a value, with the callee as
operand 0 and the arguments after it, since a function pointer carries no signature. The collector
collects a function whose address is taken, which it did not before, because nothing called it by
name. A call whose callee is a *constant* function pointer is emitted directly: rustc const-folds a
struct of fn items, and the constant's provenance is `GlobalAlloc::Function`, which names the
function it points at. `llvm-export` already implemented `CallOpCallable::Indirect` and
`AddressOfOp`, so nothing below the MIR dialect changed.

`cuda_device::warp::{run_worker, dispatch}` provides a cooperative-entry prerequisite: one
controller keeps 31 helper lanes resident instead of letting them exit before future work exists.
The controller publishes a callback and frame through block-shared storage, helpers release their
completion after calling it, and the controller acquires every helper before reusing the frame.
Only entry and exit use a converged full-warp barrier; the divergent service loop uses block-scope
acquire/release. All callback frames must already be warp-accessible: this is not promotion of a
private local pointer, a generic result-transfer implementation, an allocator service, or a nested
invocation scheduler. The two unsafe primitives make those missing obligations explicit rather
than supplying a CPU model or a scalar fallback. An isolated sm_75 probe passed on rome4gpu GPU1
(RTX 2080 SUPER, driver 550.163.01): two independent warps, two complete worker lifetimes,
lengths 0/1/30/31/32/96, each item once, all 31 physical helpers, and two callback addresses.
This is entry/dispatch evidence, not support for trame's full public `parallel` signature.

`core::hint::spin_loop` lowers to nothing: core selects its body with `cfg_select!` on
`target_arch` and core is precompiled for x86_64, so the device build reaches
`core::arch::x86_64::_mm_pause`, whose declaration names `llvm.x86.sse2.pause`. There is no pause
instruction on the device, and the standard library's own rule for a platform that does not support
the hint is to do nothing. `size_of_val` and `align_of_val` of a slice lower too:
`align_of_val` is the element alignment, still a constant, and `size_of_val` is the element count
times the element size, the count being field 1 of the `{ ptr, len }` pair the slice lowers to.
`sort_unstable` reaches this through `sift_down`. Still refused:
`size_of_val` of any other unsized type, anything reaching `core::fmt` or a `dyn` vtable — which
includes `Result::unwrap` and `Result::expect`, whose panic path takes `&dyn Debug` — and a constant
whose bytes carry a relocation, which is what a slice compared with a literal is.

The remaining enum storage fixtures deliberately fail closed. A pointer word
and a *whole* pointer-sized integer word may share bytes: the shared bytes are
stored as a real `ptr` and the integer side crosses the boundary with
`inttoptr`/`ptrtoint`, so the pointer variant keeps its provenance and the
integer variant keeps all 64 bits (the `enum_pointer_overlay` example). A
pointer-free payload that shares bytes with another variant's pointer is also
representable: a pointer leaf may split a pointer-free byte-faithful aggregate
it sits in or reaches past (the `Result<Route, Failure>` / `Result<Fabric,
BackendFault>` shapes), and a pointer-free field that reaches past a foreign
pointer word is stored slotlessly with the pointer bytes it does not cover
zeroed (`Result<Launched, BackendFault>`). A pointer only *partly* covered by
an integer strictly inside it stays refused, because no single slot type is
both provenance-carrying and integer-exact. Shared-memory pointers are 64 bits
in PTX and legacy NVVM output but 32 bits in modern NVVM output. Direct fields
and pointer leaves nested through structs/tuples use target-stable generic
physical storage, with recursive reconstruction at enum construction and
extraction boundaries. Arrays and vectors containing shared pointers remain
rejected: arrays need an explicit expansion bound and code-shape contract,
while pointer vectors need separate ABI and address-space-cast semantics.
(Bool leaves use canonical i8 physical bytes and are converted recursively at
the same boundary.)
