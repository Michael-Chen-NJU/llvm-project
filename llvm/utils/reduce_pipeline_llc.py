#!/usr/bin/env python3

"""
LLVM Backend Pass Pipeline Reducer (reduce_pipeline_llc.py)

Reduces an llc backend pass pipeline to the minimal set of passes needed
to reproduce a crash, verification error, or correctness bug.

Three modes of operation:
  1. Full-pipeline mode: Input is .ll, full pipeline crashes.
     Tool uses --stop-before binary search + DDMIN.
  2. Run-pass mode: Input is .mir with a known crashing --run-pass sequence.
     Tool applies DDMIN directly to reduce the pass list.
  3. Test-script mode: User provides a script that decides "interesting" or not.
     Enables reduction for correctness bugs (miscompiles), not just crashes.

Usage:
  # Mode 1: full pipeline crash (.ll input)
  reduce_pipeline_llc.py --llc-binary=./build/bin/llc --input=test.ll

  # Mode 2: known crashing pass sequence (.mir input)
  reduce_pipeline_llc.py --llc-binary=./build/bin/llc --input=test.mir \\
      --run-pass=livevars,phi-node-elimination,twoaddressinstruction,...

  # Mode 3: test-script for correctness bugs
  reduce_pipeline_llc.py --llc-binary=./build/bin/llc --input=test.ll \\
      --run-pass=PASSES --test-script=./check_miscompile.sh

  # With error pattern to distinguish target crash from noise:
  reduce_pipeline_llc.py --llc-binary=./build/bin/llc --input=test.mir \\
      --run-pass=PASSES --error-pattern="Wrong value out of predecessor"

Test-script interface:
  The script is called with no arguments. It receives environment variables:
    $LLC_BINARY   - path to llc
    $INPUT_FILE   - path to input .ll or .mir file
    $PASSES       - comma-separated pass list (current candidate)
    $EXTRA_ARGS   - space-separated extra llc arguments
    $OUTPUT_DIR   - temporary directory the script may use for scratch files
  Exit code 0 = interesting (bug reproduces), non-zero = not interesting.

  Example test script for a miscompile:
    #!/bin/bash
    $LLC_BINARY --run-pass=$PASSES -o $OUTPUT_DIR/out.mir $INPUT_FILE $EXTRA_ARGS || exit 1
    $LLC_BINARY -o $OUTPUT_DIR/out.s $OUTPUT_DIR/out.mir $EXTRA_ARGS || exit 1
    grep -q "wrong_instruction" $OUTPUT_DIR/out.s
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from enum import Enum
from math import ceil


class Result(Enum):
    INTERESTING = 0
    UNINTERESTING = 1
    NOISE = 2


FILTER_PASSES = frozenset({
    'targetlibinfo', 'targetpassconfig', 'machinemoduleinfo',
    'machine-branch-prob', 'regalloc-evict', 'regalloc-priority',
    'machineverifier', 'verify',
})


def run_llc(llc_binary, input_file, extra_args=None, run_pass=None,
            stop_before=None, output=None, verify=False, timeout_sec=60):
    """Run llc and return (returncode, stdout, stderr).

    Returns negative signal number if killed by signal.
    Returns -9 on timeout.
    """
    cmd = [llc_binary]
    if extra_args:
        cmd.extend(extra_args)
    if run_pass:
        cmd.append('--run-pass=' + ','.join(run_pass))
    if stop_before:
        cmd.append(f'--stop-before={stop_before}')
    if verify:
        cmd.append('-verify-machineinstrs')
    if output:
        cmd.extend(['-o', output])
    else:
        cmd.extend(['-o', '/dev/null'])
    cmd.append(input_file)

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=timeout_sec)
        return result.returncode, result.stdout.decode('utf-8', errors='replace'), \
               result.stderr.decode('utf-8', errors='replace')
    except subprocess.TimeoutExpired:
        return -9, '', 'TIMEOUT'
    except OSError as e:
        return -1, '', f'OSError: {e}'


def classify(rc, stderr, error_pattern=None):
    """Classify a test result.

    - INTERESTING: the target bug reproduces
    - UNINTERESTING: no crash (rc==0)
    - NOISE: a crash, but not the target crash
    """
    if rc == 0:
        return Result.UNINTERESTING
    # If no error_pattern specified, any crash is interesting
    if not error_pattern:
        return Result.INTERESTING
    # Check if the error pattern appears in stderr
    if error_pattern in stderr:
        return Result.INTERESTING
    # Pattern not found in stderr — this is a different crash (NOISE)
    return Result.NOISE


def run_test_script(test_script, llc_binary, input_file, extra_args, passes,
                    output_dir, timeout_sec=120):
    """Run the user's test script and return Result based on exit code.

    Exit 0 = INTERESTING, non-zero = UNINTERESTING.
    """
    env = os.environ.copy()
    env['LLC_BINARY'] = llc_binary
    env['INPUT_FILE'] = input_file
    env['PASSES'] = ','.join(passes) if passes else ''
    env['EXTRA_ARGS'] = ' '.join(extra_args) if extra_args else ''
    env['OUTPUT_DIR'] = output_dir

    try:
        result = subprocess.run([test_script], env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=timeout_sec)
        if result.returncode == 0:
            return Result.INTERESTING
        return Result.UNINTERESTING
    except subprocess.TimeoutExpired:
        return Result.UNINTERESTING
    except OSError as e:
        print(f"  WARNING: test script error: {e}")
        return Result.UNINTERESTING


def extract_pass_pipeline(llc_binary, input_file, extra_args=None):
    """Extract the pass pipeline from llc -debug-pass=Arguments."""
    cmd = [llc_binary]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(['-debug-pass=Arguments', '-o', '/dev/null', input_file])
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=60)
    except (subprocess.TimeoutExpired, OSError):
        return []
    stderr = result.stderr.decode('utf-8', errors='replace')

    passes = []
    for line in stderr.split('\n'):
        if 'Pass Arguments:' in line:
            tokens = line.split('Pass Arguments:')[1].strip().split()
            for t in tokens:
                name = t.lstrip('-')
                if name and name not in FILTER_PASSES:
                    passes.append(name)
    return passes


def find_machine_pass_start(passes):
    """Find the index where machine passes begin (after ISel)."""
    isel_markers = ('finalize-isel', 'riscv-isel', 'aarch64-isel', 'x86-isel',
                    'amdgpu-isel', 'riscv-vector-peephole')
    for i, p in enumerate(passes):
        if p in isel_markers:
            return i
    for i, p in enumerate(passes):
        if 'isel' in p.lower():
            return i
    return 0


def get_crash_signature(llc_binary, input_file, extra_args, run_pass=None):
    """Run llc and extract a crash signature from stderr."""
    rc, _, stderr = run_llc(llc_binary, input_file, extra_args,
                            run_pass=run_pass, verify=True, timeout_sec=60)
    if rc == 0:
        return None, None

    # Try to extract assertion text
    for line in stderr.split('\n'):
        if 'Assertion' in line and 'failed' in line:
            m = re.search(r"Assertion [`'](.+?)[`'] failed", line)
            if m:
                return rc, m.group(1)
            idx = line.find('Assertion')
            if idx >= 0:
                return rc, line[idx:].strip()[:120]
            return rc, line.strip()[:120]
    # Try LLVM ERROR
    if 'LLVM ERROR:' in stderr:
        msg = stderr.split('LLVM ERROR:')[1].split('\n')[0].strip()
        return rc, msg[:80]
    # Try UNREACHABLE
    if 'UNREACHABLE' in stderr:
        lines = stderr.split('\n')
        for i, line in enumerate(lines):
            if 'UNREACHABLE' in line:
                # Message is usually the line before UNREACHABLE
                if i > 0 and lines[i-1].strip() and 'Stack dump' not in lines[i-1]:
                    return rc, lines[i-1].strip()[:80]
                # Fallback: extract file:line from UNREACHABLE line
                m = re.search(r'UNREACHABLE executed at .*/([^/]+:\d+)', line)
                if m:
                    return rc, f'UNREACHABLE at {m.group(1)}'
                return rc, line.strip()[:80]
    # Generic: just note the rc
    return rc, None


def phase0_verify_and_extract(llc_binary, input_file, extra_args, run_pass_list,
                               error_pattern):
    """Phase 0: Verify crash reproduces and extract pass pipeline."""
    print("=" * 70)
    print("Phase 0: Verify and Extract")
    print("=" * 70)

    if run_pass_list:
        print(f"Mode: --run-pass with {len(run_pass_list)} passes")
        rc, _, stderr = run_llc(llc_binary, input_file, extra_args,
                                run_pass=run_pass_list, verify=True)
        if rc == 0:
            print(f"ERROR: Given --run-pass sequence does not crash (rc=0)")
            return None, None

        # Auto-detect error pattern if not provided
        if not error_pattern:
            _, sig = get_crash_signature(llc_binary, input_file, extra_args,
                                         run_pass=run_pass_list)
            if sig:
                error_pattern = sig
                print(f"Auto-detected error: '{error_pattern}'")

        # Verify with the pattern
        result = classify(rc, stderr, error_pattern)
        if result != Result.INTERESTING:
            print(f"WARNING: Crash doesn't match error pattern (rc={rc})")
            print(f"  Pattern: '{error_pattern}'")
            print(f"  Proceeding without pattern (any crash = interesting)")
            error_pattern = None

        print(f"Confirmed: --run-pass sequence crashes (rc={rc})")
        return run_pass_list, error_pattern

    # Full pipeline mode
    print("Mode: Full pipeline")
    rc, _, stderr = run_llc(llc_binary, input_file, extra_args, verify=True)
    if rc == 0:
        print(f"ERROR: Full pipeline does not crash (rc=0)")
        print("Hint: use --run-pass=A,B,C to specify a known-crashing sequence")
        return None, None

    print(f"Confirmed: full pipeline crashes (rc={rc})")

    # Auto-detect error pattern
    if not error_pattern:
        _, sig = get_crash_signature(llc_binary, input_file, extra_args)
        if sig:
            error_pattern = sig
            print(f"Error signature: '{error_pattern}'")

    # Extract pass pipeline
    print("\nExtracting pass pipeline...")
    passes = extract_pass_pipeline(llc_binary, input_file, extra_args)
    if not passes:
        print("ERROR: Could not extract pass pipeline")
        return None, None
    print(f"Extracted {len(passes)} passes")

    machine_start = find_machine_pass_start(passes)
    machine_passes = passes[machine_start:]
    print(f"Machine passes: {len(machine_passes)} (from {passes[machine_start]})")

    return machine_passes, error_pattern


def phase1_binary_search_stop_before(llc_binary, input_file, extra_args, passes,
                                      error_pattern):
    """Phase 1: Binary search to find which pass triggers the crash."""
    print("\n" + "=" * 70)
    print("Phase 1: Binary Search (--stop-before)")
    print("=" * 70)

    lo, hi = 0, len(passes) - 1
    last_interesting = len(passes) - 1

    while lo < hi:
        mid = (lo + hi) // 2
        pass_name = passes[mid]
        rc, _, stderr = run_llc(llc_binary, input_file, extra_args,
                                stop_before=pass_name, verify=True)
        # Classify: crash before this pass = INTERESTING (pass not needed)
        # No crash = UNINTERESTING (this pass or later is required)
        result = classify(rc, stderr, error_pattern)
        if result == Result.INTERESTING:
            hi = mid
            last_interesting = mid
            print(f"  [{mid:3d}] --stop-before={pass_name:<35} crashes")
        else:
            lo = mid + 1
            print(f"  [{mid:3d}] --stop-before={pass_name:<35} OK")

    trigger_idx = last_interesting
    # The trigger pass is the one BEFORE trigger_idx:
    # --stop-before=passes[trigger_idx] crashes → passes[trigger_idx-1] caused it
    if trigger_idx > 0:
        actual_trigger = trigger_idx - 1
    else:
        actual_trigger = 0
    print(f"\nTrigger pass: {passes[actual_trigger]} (index {actual_trigger})")
    print(f"  (--stop-before={passes[actual_trigger]} OK; --stop-before={passes[trigger_idx]} crashes)")
    return actual_trigger


def phase2_freeze_mir(llc_binary, input_file, extra_args, passes, trigger_idx,
                       output_mir, error_pattern):
    """Phase 2: Generate frozen MIR at the trigger point."""
    print("\n" + "=" * 70)
    print("Phase 2: Freeze MIR")
    print("=" * 70)

    trigger_pass = passes[trigger_idx]
    print(f"Generating MIR before: {trigger_pass}")

    rc, _, _ = run_llc(llc_binary, input_file, extra_args,
                       stop_before=trigger_pass, output=output_mir, verify=False)
    if rc != 0:
        print(f"ERROR: Failed to generate MIR (rc={rc})")
        return None

    # Verify bug reproduces from frozen MIR
    candidate_passes = passes[trigger_idx:]
    print(f"Candidate passes for DDMIN: {len(candidate_passes)}")
    print("Verifying MIR round-trip...")

    rc, _, stderr = run_llc(llc_binary, output_mir, extra_args,
                            run_pass=candidate_passes, verify=True)
    result = classify(rc, stderr, error_pattern)
    if result != Result.INTERESTING:
        print("WARNING: MIR round-trip lost the bug.")
        print("Falling back to Phase 1 result only.")
        return None

    print("MIR round-trip OK.")
    return candidate_passes


def ddmin(test_fn, working, verbose=True):
    """Standard DDMIN algorithm. test_fn(candidate) -> Result."""
    test_count = 0
    initial_len = len(working)
    n = 2

    while len(working) >= 2:
        chunk_size = ceil(len(working) / n)
        reduced = False

        for i in range(n):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(working))
            complement = working[:start] + working[end:]

            if not complement:
                continue

            test_count += 1
            result = test_fn(complement)
            if result == Result.INTERESTING:
                removed = working[start:end]
                working = complement
                n = max(n - 1, 2)
                reduced = True
                if verbose:
                    label = removed[0] if len(removed) == 1 else f"{removed[0]}...+{len(removed)-1}"
                    print(f"  Removed ({len(removed)}): {label}")
                break

        if not reduced:
            if n >= len(working):
                break
            n = min(n * 2, len(working))

    if verbose:
        print(f"  DDMIN: {test_count} tests, {initial_len} -> {len(working)} passes")
    return working, test_count


def phase3_reduce(llc_binary, input_file, extra_args, passes, error_pattern,
                  use_verify):
    """Phase 3: DDMIN to find minimal pass set."""
    print("\n" + "=" * 70)
    print("Phase 3: DDMIN Reduction")
    print("=" * 70)
    print(f"Starting with {len(passes)} passes, verify={use_verify}")

    def test_fn(candidate):
        if not candidate:
            return Result.UNINTERESTING
        rc, _, stderr = run_llc(llc_binary, input_file, extra_args,
                                run_pass=candidate, verify=use_verify)
        return classify(rc, stderr, error_pattern)

    result, test_count = ddmin(test_fn, list(passes))
    return result


def phase4_single_pass_sweep(llc_binary, input_file, extra_args, passes,
                              error_pattern, use_verify):
    """Phase 4: Try removing each pass individually."""
    print("\n" + "=" * 70)
    print("Phase 4: Single-Pass Sweep")
    print("=" * 70)

    changed = True
    while changed:
        changed = False
        for i in range(len(passes)):
            candidate = passes[:i] + passes[i+1:]
            if not candidate:
                continue
            rc, _, stderr = run_llc(llc_binary, input_file, extra_args,
                                    run_pass=candidate, verify=use_verify)
            result = classify(rc, stderr, error_pattern)
            if result == Result.INTERESTING:
                print(f"  Removed: {passes[i]}")
                passes = candidate
                changed = True
                break

    print(f"  Final: {len(passes)} passes")
    return passes


def _run_test_script_mode(args, extra_args, run_pass_list, use_verify):
    """Run in test-script mode: user-provided oracle for interestingness."""
    print("=" * 70)
    print("Test-Script Mode")
    print("=" * 70)
    print(f"Script: {args.test_script}")

    tmpdir = tempfile.mkdtemp(prefix='reduce_pipeline_')
    print(f"Scratch dir: {tmpdir}")

    # Determine pass list
    is_mir = args.input.endswith('.mir') or run_pass_list is not None

    if run_pass_list:
        passes = run_pass_list
        print(f"Starting passes: {len(passes)} (from --run-pass)")
    else:
        print("Extracting pass pipeline...")
        passes = extract_pass_pipeline(args.llc_binary, args.input, extra_args)
        if not passes:
            print("ERROR: Could not extract pass pipeline")
            sys.exit(1)
        machine_start = find_machine_pass_start(passes)
        passes = passes[machine_start:]
        print(f"Extracted {len(passes)} machine passes")

    # Verify the test script says "interesting" with the full pass set
    print("\nVerifying test script with full pass set...")
    if is_mir:
        # MIR input: test with all passes via --run-pass
        result = run_test_script(args.test_script, args.llc_binary, args.input,
                                 extra_args, passes, tmpdir)
    else:
        # .ll input: initial verification uses full pipeline (empty PASSES)
        result = run_test_script(args.test_script, args.llc_binary, args.input,
                                 extra_args, ['FULL_PIPELINE'], tmpdir)
    if result != Result.INTERESTING:
        print("ERROR: Test script returns 'not interesting' with full pass set.")
        print("  The test script must return exit code 0 when the bug is present.")
        sys.exit(1)
    print("Confirmed: test script reports bug with full pass set.")

    # For .ll inputs: try opt-bisect-limit binary search first
    mir_input = args.input

    if not is_mir:
        print("\n" + "=" * 70)
        print("Phase 1: opt-bisect-limit Binary Search")
        print("=" * 70)
        bisect_result = _bisect_limit_test_script(
            args.test_script, args.llc_binary, args.input, extra_args, tmpdir)

        if bisect_result:
            trigger_limit, trigger_pass, max_n = bisect_result
            print(f"\nTrigger: pass #{trigger_limit}/{max_n}: {trigger_pass}")
            print(f"\nReproduction:")
            ea = ' '.join(extra_args) if extra_args else ''
            print(f"  {args.llc_binary} -opt-bisect-limit={trigger_limit} "
                  f"-o /dev/null {args.input} {ea}")
            if args.output:
                import shutil
                shutil.copy(args.input, args.output)
                print(f"\nInput saved to: {args.output}")
            sys.exit(0)
        else:
            print("opt-bisect-limit search inconclusive (mandatory pass bug).")
            print("Attempting --stop-before + --run-pass DDMIN (may not reduce)...")

        # Fall through to --stop-before + DDMIN approach

    if not is_mir:
        print("\n" + "=" * 70)
        print("Phase 1: Binary Search (--stop-before) with test script")
        print("=" * 70)
        trigger_idx = _binary_search_test_script(
            args.test_script, args.llc_binary, args.input, extra_args,
            passes, tmpdir)

        if trigger_idx is not None:
            output_mir = args.output or os.path.join(tmpdir, 'frozen.mir')
            trigger_pass = passes[trigger_idx]
            print(f"\nFreezing MIR before: {trigger_pass}")
            rc, _, _ = run_llc(args.llc_binary, args.input, extra_args,
                               stop_before=trigger_pass, output=output_mir,
                               verify=False)
            if rc == 0:
                candidate_passes = passes[trigger_idx:]
                # Verify test script still passes with frozen MIR
                result = run_test_script(args.test_script, args.llc_binary,
                                         output_mir, extra_args,
                                         candidate_passes, tmpdir)
                if result == Result.INTERESTING:
                    mir_input = output_mir
                    passes = candidate_passes
                    print(f"MIR round-trip OK. Reduced to {len(passes)} candidate passes.")
                else:
                    print("WARNING: MIR round-trip lost the bug. Using full pass list.")
            else:
                print(f"WARNING: Failed to generate MIR (rc={rc}). Using full pass list.")
        else:
            print("Binary search inconclusive. Using full pass list.")

    # Phase 3: DDMIN with test script as oracle
    print("\n" + "=" * 70)
    print("Phase 3: DDMIN Reduction (test-script oracle)")
    print("=" * 70)
    print(f"Starting with {len(passes)} passes")

    def test_fn(candidate):
        if not candidate:
            return Result.UNINTERESTING
        return run_test_script(args.test_script, args.llc_binary, mir_input,
                               extra_args, candidate, tmpdir)

    reduced, _ = ddmin(test_fn, list(passes))

    # Phase 4: Single-pass sweep
    print("\n" + "=" * 70)
    print("Phase 4: Single-Pass Sweep (test-script oracle)")
    print("=" * 70)

    changed = True
    while changed:
        changed = False
        for i in range(len(reduced)):
            candidate = reduced[:i] + reduced[i+1:]
            if not candidate:
                continue
            result = test_fn(candidate)
            if result == Result.INTERESTING:
                print(f"  Removed: {reduced[i]}")
                reduced = candidate
                changed = True
                break

    print(f"  Final: {len(reduced)} passes")

    # Output
    print(f"\n{'='*70}")
    print("RESULT")
    print(f"{'='*70}")
    print(f"Minimal pass set ({len(reduced)} passes):")
    for i, p in enumerate(reduced):
        print(f"  {i}: {p}")
    pass_str = ','.join(reduced)
    ea = ' '.join(extra_args) if extra_args else ''
    print(f"\nReproduction:")
    print(f"  LLC_BINARY={args.llc_binary} INPUT_FILE={mir_input} "
          f"PASSES={pass_str} EXTRA_ARGS=\"{ea}\" {args.test_script}")

    if args.output and mir_input != args.output:
        import shutil
        shutil.copy(mir_input, args.output)
        print(f"\nMIR saved to: {args.output}")

    sys.exit(0)


def _bisect_limit_test_script(test_script, llc_binary, input_file, extra_args,
                               tmpdir):
    """Use opt-bisect-limit + test script to find the trigger pass.

    Returns (trigger_limit, trigger_pass_name, max_n) or None if the bug
    is in a mandatory pass (present even at limit=0).
    """
    # Determine max bisect count
    rc, _, stderr = run_llc(llc_binary, input_file, extra_args,
                            verify=False, timeout_sec=60)
    # Run with high limit to get BISECT lines
    cmd = [llc_binary]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(['-opt-bisect-limit=999999', '-o', '/dev/null', input_file])
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=60)
        bisect_stderr = result.stderr.decode('utf-8', errors='replace')
    except (subprocess.TimeoutExpired, OSError):
        print("  Could not determine max bisect count")
        return None

    bisect_lines = [l for l in bisect_stderr.split('\n') if 'BISECT:' in l]
    if not bisect_lines:
        print("  No BISECT output (llc may not support opt-bisect-limit)")
        return None
    max_n = len(bisect_lines)
    print(f"  Max bisect count: {max_n}")

    # Check if bug is present at limit=0 (mandatory pass bug)
    result_0 = run_test_script(test_script, llc_binary, input_file,
                                extra_args + ['-opt-bisect-limit=0'],
                                ['FULL_PIPELINE'], tmpdir)
    if result_0 == Result.INTERESTING:
        print("  Bug present at limit=0 — mandatory pass bug")
        return None

    # Binary search
    lo, hi = 0, max_n
    print(f"  Binary search over opt-bisect-limit [0, {max_n}]...")

    while lo < hi:
        mid = (lo + hi) // 2
        result = run_test_script(test_script, llc_binary, input_file,
                                  extra_args + [f'-opt-bisect-limit={mid}'],
                                  ['FULL_PIPELINE'], tmpdir)
        if result == Result.INTERESTING:
            hi = mid
            print(f"    limit={mid}: INTERESTING")
        else:
            lo = mid + 1
            print(f"    limit={mid}: not interesting")

    # Extract pass name at trigger limit
    trigger_pass = None
    if lo > 0 and lo <= max_n:
        cmd = [llc_binary]
        if extra_args:
            cmd.extend(extra_args)
        cmd.extend([f'-opt-bisect-limit={lo}', '-o', '/dev/null', input_file])
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, timeout=60)
            stderr = result.stderr.decode('utf-8', errors='replace')
            for line in stderr.split('\n'):
                if f'BISECT: running pass ({lo})' in line:
                    m = re.search(r'BISECT: running pass \(\d+\) (.+?) on', line)
                    if m:
                        trigger_pass = m.group(1).strip()
                    break
        except (subprocess.TimeoutExpired, OSError):
            pass

    if trigger_pass:
        return (lo, trigger_pass, max_n)

    # Fallback: try to get pass name from max limit run
    for line in bisect_lines:
        if f'({lo})' in line:
            m = re.search(r'BISECT: running pass \(\d+\) (.+?) on', line)
            if m:
                return (lo, m.group(1).strip(), max_n)

    return (lo, f'pass_#{lo}', max_n)


def _binary_search_test_script(test_script, llc_binary, input_file, extra_args,
                                passes, tmpdir):
    """Binary search using --stop-before + test script as oracle.

    For correctness bugs: stop-before a pass, produce MIR, then run
    the test script with passes[mid:] to see if the bug still manifests.
    Returns the index of the earliest pass where the bug first appears,
    or None if binary search is inconclusive.
    """
    lo, hi = 0, len(passes) - 1
    last_interesting = None

    while lo < hi:
        mid = (lo + hi) // 2
        pass_name = passes[mid]
        # Generate MIR at this point
        mir_path = os.path.join(tmpdir, f'bisect_{mid}.mir')
        rc, _, _ = run_llc(llc_binary, input_file, extra_args,
                           stop_before=pass_name, output=mir_path, verify=False)
        if rc != 0:
            lo = mid + 1
            print(f"  [{mid:3d}] --stop-before={pass_name:<35} llc failed, skip")
            continue

        # Test with passes from mid onward
        candidate = passes[mid:]
        result = run_test_script(test_script, llc_binary, mir_path,
                                 extra_args, candidate, tmpdir)
        if result == Result.INTERESTING:
            hi = mid
            last_interesting = mid
            print(f"  [{mid:3d}] --stop-before={pass_name:<35} INTERESTING")
        else:
            lo = mid + 1
            print(f"  [{mid:3d}] --stop-before={pass_name:<35} not interesting")

    if last_interesting is not None:
        print(f"\nTrigger region starts at: {passes[last_interesting]} (index {last_interesting})")
    return last_interesting


def main():
    parser = argparse.ArgumentParser(
        description="LLVM Backend Pass Pipeline Reducer",
        epilog="Extra arguments after the known flags are forwarded to llc.")
    parser.add_argument('--llc-binary', default='llc',
                        help='Path to llc binary (default: llc)')
    parser.add_argument('--input', required=True,
                        help='Input .ll or .mir file')
    parser.add_argument('--output',
                        help='Save reduced MIR to this path')
    parser.add_argument('--run-pass',
                        help='Known crashing pass sequence (comma-separated)')
    parser.add_argument('--error-pattern',
                        help='String to match in stderr for target crash')
    parser.add_argument('--no-verify', action='store_true',
                        help='Do not add -verify-machineinstrs during DDMIN')
    parser.add_argument('--test-script',
                        help='Path to interestingness test script (exit 0 = interesting). '
                             'Enables correctness bug reduction.')
    args, extra_args = parser.parse_known_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    if args.test_script and not os.path.isfile(args.test_script):
        print(f"ERROR: Test script not found: {args.test_script}")
        sys.exit(1)
    if args.test_script and not os.access(args.test_script, os.X_OK):
        print(f"ERROR: Test script not executable: {args.test_script}")
        sys.exit(1)

    run_pass_list = None
    if args.run_pass:
        run_pass_list = [p.strip() for p in args.run_pass.split(',') if p.strip()]

    use_verify = not args.no_verify

    # Test-script mode: user provides the interestingness oracle
    if args.test_script:
        _run_test_script_mode(args, extra_args, run_pass_list, use_verify)
        return

    # Phase 0: Verify and extract
    passes, error_pattern = phase0_verify_and_extract(
        args.llc_binary, args.input, extra_args, run_pass_list, args.error_pattern)
    if passes is None:
        sys.exit(1)

    mir_input = args.input
    is_mir = args.input.endswith('.mir') or run_pass_list is not None

    # Phase 1 + 2: Only for full-pipeline mode (.ll without --run-pass)
    if not is_mir:
        trigger_idx = phase1_binary_search_stop_before(
            args.llc_binary, args.input, extra_args, passes, error_pattern)

        # If trigger is the very first pass (index 0), the crash is likely
        # in a pre-ISel pass that runs before the machine pipeline
        if trigger_idx == 0:
            rc_check, _, _ = run_llc(args.llc_binary, args.input, extra_args,
                                     stop_before=passes[0], verify=False)
            if rc_check != 0:
                print(f"\n{'='*70}")
                print("RESULT (pre-ISel crash — trigger before machine pipeline)")
                print(f"{'='*70}")
                print(f"Earliest machine pass: {passes[0]}")
                print("The crash occurs in a pass that runs before the machine pipeline.")
                ea = ' '.join(extra_args)
                print(f"  {args.llc_binary} --stop-before={passes[0]} "
                      f"-o reduced.mir {args.input} {ea}")
                sys.exit(0)

        output_mir = args.output or tempfile.mktemp(suffix='.mir')
        candidate_passes = phase2_freeze_mir(
            args.llc_binary, args.input, extra_args, passes, trigger_idx,
            output_mir, error_pattern)

        if candidate_passes:
            mir_input = output_mir
            passes = candidate_passes
        else:
            print(f"\n{'='*70}")
            print("RESULT (Phase 1 only — MIR round-trip failed)")
            print(f"{'='*70}")
            print(f"Trigger pass: {passes[trigger_idx]}")
            ea = ' '.join(extra_args)
            print(f"  {args.llc_binary} --stop-before={passes[trigger_idx]} "
                  f"-o reduced.mir {args.input} {ea}")
            sys.exit(0)

    # Phase 3: DDMIN
    reduced = phase3_reduce(args.llc_binary, mir_input, extra_args, passes,
                            error_pattern, use_verify)

    # Phase 4: Single-pass sweep for final minimality
    final = phase4_single_pass_sweep(args.llc_binary, mir_input, extra_args,
                                     reduced, error_pattern, use_verify)

    # Output
    print(f"\n{'='*70}")
    print("RESULT")
    print(f"{'='*70}")
    print(f"Minimal pass set ({len(final)} passes):")
    for i, p in enumerate(final):
        print(f"  {i}: {p}")
    pass_str = ','.join(final)
    verify_flag = '-verify-machineinstrs ' if use_verify else ''
    ea = ' '.join(extra_args) if extra_args else ''
    print(f"\nReproduction command:")
    print(f"  {args.llc_binary} --run-pass={pass_str} {verify_flag}"
          f"-o /dev/null {mir_input} {ea}")

    if args.output and mir_input != args.output:
        import shutil
        shutil.copy(mir_input, args.output)
        print(f"\nMIR saved to: {args.output}")

    sys.exit(0)


if __name__ == '__main__':
    main()
