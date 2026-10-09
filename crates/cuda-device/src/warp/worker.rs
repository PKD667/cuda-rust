// SPDX-License-Identifier: Apache-2.0

//! Resident helpers for one scalar controller, not another CUDA launch. Callback frames belong
//! to the caller and must already be accessible to the whole warp; private local pointers are
//! not promoted by this protocol.

use core::cell::UnsafeCell;

use crate::atomic::{AtomicOrdering, BlockAtomicU32};
use crate::shared::SharedArray;

const HELPERS: usize = 31;
// CUDA admits at most 1024 threads, hence 32 warps, in one block.
const WARPS: usize = 32;

#[derive(Clone, Copy)]
struct Job {
    callback: unsafe fn(*mut u8, u32),
    frame: *mut u8,
    count: u32,
}

struct Team {
    epoch: BlockAtomicU32,
    stopped: BlockAtomicU32,
    job: UnsafeCell<Option<Job>>,
    completed: [BlockAtomicU32; HELPERS],
}

// SAFETY: only the controller writes `job`, before releasing an epoch. It cannot overwrite the
// job until all helpers release completion, which the controller acquires. Each helper owns one
// completion word; all state words are atomic while the team is live.
unsafe impl Sync for Team {}

static mut TEAMS: SharedArray<Team, WARPS> = SharedArray::UNINIT;

#[inline(always)]
unsafe fn team() -> &'static Team {
    let warp = crate::thread::threadIdx_x() / 32;
    unsafe { &*(&*(&raw const TEAMS)).as_ptr().add(warp as usize) }
}

/// Run scalar worker code while the other 31 lanes remain available for dispatch.
///
/// # Safety
///
/// All 32 lanes of each participating warp must enter this call together on sm_70 or newer.
/// Blocks are one-dimensional with a multiple of 32 threads. Calls cannot overlap or nest within
/// one warp. `body` runs only on lane 0 and must finish every dispatch before returning. Closure
/// construction and the unused helper closures' drops must not perform worker effects: construct
/// worker-owned state inside `body`. A trap ends the launch; it does not promise helper cleanup.
#[inline(never)]
pub unsafe fn run_worker(body: impl FnOnce()) {
    assert!(crate::thread::blockDim_y() == 1 && crate::thread::blockDim_z() == 1);
    assert!(crate::thread::blockDim_x() % 32 == 0);
    let lane = super::lane_id();
    if lane == 0 {
        let warp = crate::thread::threadIdx_x() / 32;
        let slot = unsafe { (&*(&raw const TEAMS)).as_ptr().add(warp as usize) as *mut Team };
        unsafe {
            slot.write(Team {
                epoch: BlockAtomicU32::new(0),
                stopped: BlockAtomicU32::new(0),
                job: UnsafeCell::new(None),
                completed: [const { BlockAtomicU32::new(0) }; HELPERS],
            });
        }
    }
    super::sync_mask(u32::MAX);
    let team = unsafe { team() };
    if lane == 0 {
        body();
        team.stopped.store(1, AtomicOrdering::Release);
    } else {
        unsafe { serve(team, lane - 1) };
    }
    // Both branches reconverge at this same call before shared state may be reused.
    super::sync_mask(u32::MAX);
}

#[inline(never)]
unsafe fn serve(team: &Team, slot: u32) {
    let mut seen = 0;
    while team.stopped.load(AtomicOrdering::Acquire) == 0 {
        let epoch = team.epoch.load(AtomicOrdering::Acquire);
        if epoch != seen {
            let job = match unsafe { *team.job.get() } {
                Some(job) => job,
                None => crate::debug::trap(),
            };
            if slot < job.count {
                unsafe { (job.callback)(job.frame, slot) };
            }
            team.completed[slot as usize].store(epoch, AtomicOrdering::Release);
            seen = epoch;
        }
    }
}

/// Dispatch one batch and join it while servicing controller-owned work.
///
/// # Safety
///
/// Called only by lane 0 inside `run_worker`, with at most 31 callbacks. `frame` and everything
/// the callback accesses must be warp-accessible and live until return. The callback owns its
/// slot's mutable state, must finish, and cannot recursively call this controller primitive.
/// `service` runs on the controller while helpers are live and must not mutate their borrowed
/// state. It is where a higher layer can service allocation requests without a same-warp lock.
/// The callback retains responsibility for initializing and transferring typed results.
#[inline(never)]
pub unsafe fn dispatch(
    frame: *mut u8,
    callback: unsafe fn(*mut u8, u32),
    count: u32,
    mut service: impl FnMut(),
) {
    assert!(super::lane_id() == 0 && count <= HELPERS as u32);
    let team = unsafe { team() };
    let epoch = team.epoch.load(AtomicOrdering::Relaxed).wrapping_add(1);
    unsafe { *team.job.get() = Some(Job { callback, frame, count }) };
    team.epoch.store(epoch, AtomicOrdering::Release);
    let mut pending = 0;
    while pending < HELPERS {
        service();
        if team.completed[pending].load(AtomicOrdering::Acquire) == epoch {
            pending += 1;
        }
    }
}
