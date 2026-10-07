/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MIR call operations.
//!
//! This module defines function call operations for the MIR dialect.

use pliron::{
    builtin::attributes::StringAttr,
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    r#type::Typed,
    result::Error,
    verify_err,
};
use pliron_derive::pliron_op;

// ============================================================================
// MirCallOp
// ============================================================================

/// MIR call operation.
///
/// Represents a function call.
///
/// # Attributes
///
/// ```text
/// | Name     | Type       | Description                |
/// |----------|------------|----------------------------|
/// | `callee` | StringAttr | Name of the called function |
/// ```
///
/// # Operands
///
/// Variadic operands matching the callee's argument types.
///
/// # Results
///
/// Variadic results matching the callee's return types.
///
/// # Verification
///
/// - Must have `callee` attribute.
/// - Argument type verification is deferred to later passes that have
///   access to the symbol table.
#[pliron_op(
    name = "mir.call",
    format,
    attributes = (callee: StringAttr)
)]
pub struct MirCallOp;

impl MirCallOp {
    /// Create a new MirCallOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirCallOp { op }
    }
}

impl Verify for MirCallOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        // Check callee attribute exists
        if self.get_attr_callee(ctx).is_none() {
            return verify_err!(op.loc(), "MirCallOp must have a callee attribute");
        }

        // Note: We cannot easily verify argument types against the callee signature
        // without a symbol table lookup which might be expensive or complex here.
        // For now, we strictly verify the structure.

        Ok(())
    }
}

// ============================================================================
// MirFuncAddrOp
// ============================================================================

/// MIR function-address operation: a function as a value.
///
/// `let f: fn(u32) -> u32 = inc;` reifies the fn item `inc`. The symbol is the one the function is
/// defined under, which is the same symbol a direct call to it names, so a call through the
/// reified pointer reaches the same definition.
///
/// # Attributes
///
/// ```text
/// | Name     | Type       | Description                                |
/// |----------|------------|--------------------------------------------|
/// | `symbol` | StringAttr | Name of the function whose address this is |
/// ```
///
/// # Operands
///
/// None.
///
/// # Results
///
/// One, the function pointer.
///
/// # Verification
///
/// - Must have a `symbol` attribute.
/// - Must have no operands and exactly one result.
#[pliron_op(
    name = "mir.func_addr",
    format,
    attributes = (symbol: StringAttr)
)]
pub struct MirFuncAddrOp;

impl MirFuncAddrOp {
    /// Create a new MirFuncAddrOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirFuncAddrOp { op }
    }
}

impl Verify for MirFuncAddrOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        if self.get_attr_symbol(ctx).is_none() {
            return verify_err!(op.loc(), "MirFuncAddrOp must have a symbol attribute");
        }
        if op.get_num_operands() != 0 {
            return verify_err!(op.loc(), "MirFuncAddrOp takes no operands");
        }
        if op.get_num_results() != 1 {
            return verify_err!(op.loc(), "MirFuncAddrOp must have exactly one result");
        }

        Ok(())
    }
}

// ============================================================================
// MirCallIndirectOp
// ============================================================================

/// MIR indirect call operation: a call through a function pointer that is a value.
///
/// The callee is not a symbol, so nothing about the signature can be read from the call site
/// alone; the arguments' own types are the signature.
///
/// # Operands
///
/// ```text
/// | Index | Name    | Description                  |
/// |-------|---------|------------------------------|
/// | 0     | callee  | The function pointer to call |
/// | 1..   | args    | The call's arguments         |
/// ```
///
/// # Results
///
/// None for a unit return, one otherwise, as for `mir.call`.
///
/// # Verification
///
/// - Must have at least the callee operand.
/// - The callee must be a pointer.
/// - Must have at most one result.
#[pliron_op(name = "mir.call_indirect", format)]
pub struct MirCallIndirectOp;

impl MirCallIndirectOp {
    /// Create a new MirCallIndirectOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirCallIndirectOp { op }
    }
}

impl Verify for MirCallIndirectOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        if op.get_num_operands() == 0 {
            return verify_err!(op.loc(), "MirCallIndirectOp must have a callee operand");
        }
        if op.get_num_results() > 1 {
            return verify_err!(op.loc(), "MirCallIndirectOp has at most one result");
        }

        // The callee is operand 0. A function pointer is a `MirPtrType` here, and a callee that is
        // not one means the importer emitted the wrong operand order.
        let callee_ty = op.get_operand(0).get_type(ctx);
        if callee_ty.deref(ctx).downcast_ref::<crate::types::MirPtrType>().is_none() {
            return verify_err!(
                op.loc(),
                "MirCallIndirectOp callee must be a pointer, found {}",
                callee_ty.deref(ctx).disp(ctx)
            );
        }

        Ok(())
    }
}

/// Register call operations into the given context.
pub fn register(ctx: &mut Context) {
    MirCallOp::register(ctx);
    MirFuncAddrOp::register(ctx);
    MirCallIndirectOp::register(ctx);
}
