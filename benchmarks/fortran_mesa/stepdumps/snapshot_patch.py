# SPDX-License-Identifier: GPL-3.0-or-later
"""Inserts opt-in snapshot and per-step wall-clock instrumentation into
MESA 26.4.1 ``star/job/run_star_support.f90`` for ``#684`` W1.

The patch adds read-only timing logs and optional ``star_write_model``
calls at listed ``(stage, model_number)`` targets; no arithmetic, solver
state, or control flow of the star run changes. Nothing runs unless the
environment variable ``FLEXMESA_SNAPSHOT_LOG`` names an output path.

Environment variables read once per process when ``FLEXMESA_SNAPSHOT_LOG``
is set and non-empty:

* ``FLEXMESA_SNAPSHOT_LOG``: tab-delimited log path appended across ``./rn``
  stages (records ``STAGE_START``, ``STEP``, ``SNAP``, ``STAGE_END``).
* ``FLEXMESA_SNAPSHOT_SPEC``: optional path listing ``<stage> <model>``
  pairs (whitespace-delimited) to dump via ``star_write_model``.
* ``FLEXMESA_SNAPSHOT_DIR``: directory where ``<case>_<stage>_<model>.mod``
  snapshots are written.
* ``FLEXMESA_SNAPSHOT_CASE``: case name prefix used in snapshot filenames.

Usage::

    python snapshot_patch.py <overlay>/star/job/run_star_support.f90
"""

from __future__ import annotations

import pathlib
import sys

MARKER = "FLEXMESA_SNAPSHOT_LOG"

EDITS: list[tuple[str, str, str]] = [
    (
        # Right after star_job actions (including relax_*) and extras_startup.
        "         if (dbg) write(*,*) 'call extras_startup'\n"
        "         call s% extras_startup(id, restart, ierr)\n"
        "         if (failed('extras_startup',ierr)) return\n",
        "         call flexmesa_snap_hook(id, 1, ierr)\n",
        "after",
    ),
    (
        # Inside do_saves right after MESA's own save_model_number write.
        "         if (s% model_number == s% job% save_model_number) then\n"
        "            call star_write_model(id, s% job% save_model_filename, "
        "ierr)\n"
        "            if (failed('star_write_model',ierr)) return\n"
        "            write(*, *) 'model saved to ' // "
        "trim(s% job% save_model_filename)\n"
        "         end if\n",
        "         call flexmesa_snap_hook(id, 2, ierr)\n"
        "         if (failed('flexmesa_snap_hook',ierr)) return\n",
        "after",
    ),
    (
        # At stage exit inside after_evolve_loop.
        "         if (s% termination_code > 0 .and. "
        "s% termination_code <= num_termination_codes) then\n"
        "            write(*, '(a)') 'termination code: ' // &\n"
        "               trim(termination_code_str(s% termination_code))\n"
        "         end if\n",
        "         call flexmesa_snap_hook(id, 3, ierr)\n",
        "after",
    ),
]

HELPER = f"""
      subroutine flexmesa_snap_hook(id, mode, ierr)
         use, intrinsic :: iso_fortran_env, only: int64
         integer, intent(in) :: id, mode
         integer, intent(out) :: ierr
         integer, parameter :: max_ids = 16, max_req = 256
         logical, save :: init_done = .false.
         integer, save :: snap_u = 0, stages_seen = 0, n_req = 0
         integer, save :: id_stage(max_ids) = 0
         logical, save :: in_evolve(max_ids) = .false.
         integer(int64), save :: t_prev(max_ids) = 0_int64, t_rate = 0_int64
         integer, save :: req_stage(max_req) = 0, req_model(max_req) = 0
         logical, save :: req_done(max_req) = .false.
         character(len=1024), save :: snap_dir = '', snap_case = ''
         character(len=1024) :: log_path, spec_path, line, snap_path
         character(len=128) :: tcode, rcode
         type(star_info), pointer :: s
         integer :: st, cu, k, rs, rm, io, lierr
         integer(int64) :: t_now
         real(dp) :: dt_wall
         logical :: prev_flag

         ierr = 0
         if (.not. init_done) then
            init_done = .true.
            call get_environment_variable('{MARKER}', log_path, status=st)
            if (st == 0 .and. len_trim(log_path) > 0) then
               open(newunit=cu, file=trim(log_path), status='old', &
                  action='read', iostat=st)
               if (st == 0) then
                  do
                     read(cu, '(A)', iostat=st) line
                     if (st /= 0) exit
                     if (len_trim(line) >= 12) then
                        if (line(1:12) == 'STAGE_START' // achar(9)) then
                           stages_seen = stages_seen + 1
                        end if
                     end if
                  end do
                  close(cu)
               end if
               open(newunit=snap_u, file=trim(log_path), status='unknown', &
                  position='append', action='write')
               call get_environment_variable('FLEXMESA_SNAPSHOT_DIR', &
                  snap_dir, status=st)
               if (st /= 0) snap_dir = ''
               call get_environment_variable('FLEXMESA_SNAPSHOT_CASE', &
                  snap_case, status=st)
               if (st /= 0) snap_case = 'star'
               call get_environment_variable('FLEXMESA_SNAPSHOT_SPEC', &
                  spec_path, status=st)
               if (st == 0 .and. len_trim(spec_path) > 0) then
                  open(newunit=cu, file=trim(spec_path), status='old', &
                     action='read', iostat=st)
                  if (st == 0) then
                     do
                        read(cu, *, iostat=io) rs, rm
                        if (io /= 0) exit
                        if (n_req < max_req) then
                           n_req = n_req + 1
                           req_stage(n_req) = rs
                           req_model(n_req) = rm
                        end if
                     end do
                     close(cu)
                  end if
               end if
               call system_clock(count_rate=t_rate)
            end if
         end if

         if (snap_u == 0 .or. id < 1 .or. id > max_ids) return
         lierr = 0
         call star_ptr(id, s, lierr)
         if (lierr /= 0) return

         if (mode == 1) then
            stages_seen = stages_seen + 1
            id_stage(id) = stages_seen
            in_evolve(id) = .true.
            write(snap_u, '(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A,L1,A,A,A,A)') &
               'STAGE_START' // achar(9), id_stage(id), achar(9), id, &
               achar(9), s% model_number, achar(9), s% nz, achar(9), &
               s% nz_old, achar(9), s% generations, achar(9), &
               s% job% load_saved_model, achar(9), &
               trim(s% job% load_model_filename), achar(9), &
               trim(s% inlist_fname)
            flush(snap_u)
            call system_clock(t_prev(id))
         else if (mode == 2) then
            if (.not. in_evolve(id)) return
            call system_clock(t_now)
            if (t_rate > 0_int64 .and. t_now >= t_prev(id)) then
               dt_wall = dble(t_now - t_prev(id)) / dble(t_rate)
            else
               dt_wall = 0.0_dp
            end if
            prev_flag = (s% nz_old == s% nz .and. s% generations > 1)
            write(snap_u, &
               '(A,I0,A,I0,A,I0,A,ES23.16E3,A,I0,A,I0,A,I0,A,L1)') &
               'STEP' // achar(9), id_stage(id), achar(9), id, achar(9), &
               s% model_number, achar(9), dt_wall, achar(9), s% nz, &
               achar(9), s% nz_old, achar(9), s% generations, achar(9), &
               prev_flag
            if (len_trim(snap_dir) > 0) then
               do k = 1, n_req
                  if (req_done(k)) cycle
                  if (req_stage(k) == id_stage(id) .and. &
                        req_model(k) == s% model_number) then
                     write(snap_path, '(A,A,A,A,I0,A,I0,A)') &
                        trim(snap_dir), '/', trim(snap_case), '_', &
                        id_stage(id), '_', s% model_number, '.mod'
                     call star_write_model(id, trim(snap_path), ierr)
                     if (ierr /= 0) return
                     req_done(k) = .true.
                     write(snap_u, '(A,I0,A,I0,A,L1,A,A)') &
                        'SNAP' // achar(9), id_stage(id), achar(9), &
                        s% model_number, achar(9), prev_flag, achar(9), &
                        trim(snap_path)
                  end if
               end do
            end if
            flush(snap_u)
            call system_clock(t_prev(id))
         else if (mode == 3) then
            if (.not. in_evolve(id)) return
            tcode = 'none'
            if (s% termination_code > 0 .and. &
                  s% termination_code <= num_termination_codes) then
               tcode = termination_code_str(s% termination_code)
            end if
            rcode = 'normal'
            if (s% result_reason /= result_reason_normal .and. &
                  s% result_reason > 0 .and. &
                  s% result_reason <= size(result_reason_str)) then
               rcode = result_reason_str(s% result_reason)
            end if
            write(snap_u, '(A,I0,A,I0,A,I0,A,A,A,A)') &
               'STAGE_END' // achar(9), id_stage(id), achar(9), id, &
               achar(9), s% model_number, achar(9), trim(tcode), achar(9), &
               trim(rcode)
            flush(snap_u)
            in_evolve(id) = .false.
         end if
      end subroutine flexmesa_snap_hook

"""

END_ANCHOR = "      end module run_star_support\n"


def patch_text(text: str) -> str:
    """Return ``text`` with the opt-in snapshot/clock instrumentation added.

    Args:
        text: Contents of upstream ``star/job/run_star_support.f90``.

    Returns:
        The patched Fortran source text.

    Raises:
        ValueError: If an anchor is missing/non-unique or already patched.
    """
    if MARKER in text:
        raise ValueError("run_star_support.f90 is already patched")
    for anchor, insert, where in EDITS:
        count = text.count(anchor)
        if count != 1:
            raise ValueError(f"anchor found {count} times:\n{anchor}")
        new = anchor + insert if where == "after" else insert + anchor
        text = text.replace(anchor, new)
    if text.count(END_ANCHOR) != 1:
        raise ValueError("end module anchor not unique")
    return text.replace(END_ANCHOR, HELPER + END_ANCHOR)


def main() -> None:
    """CLI entry point."""
    path = pathlib.Path(sys.argv[1])
    path.write_text(
        patch_text(path.read_text(encoding="utf-8")), encoding="utf-8"
    )
    print(f"patched {path}")


if __name__ == "__main__":
    main()
