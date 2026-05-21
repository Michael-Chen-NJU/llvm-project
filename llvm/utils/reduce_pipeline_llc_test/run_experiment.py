#!/usr/bin/env python3
"""
Large-scale validation experiment for reduce_pipeline_llc.py.

Runs reduce_pipeline_llc.py on backend machine-pass bugs from the comp-bench
dataset, evaluating accuracy and robustness.

Usage:
    python run_experiment.py [--start-index=N] [--max-instances=M] [--dry-run]

Resume: just re-run the script; it reads checkpoint.json and skips completed.
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────

DATASET_PATH = "/data/chenziang3/project/comp-bench/data/llvm/llvm-2025.jsonl"
EXPERIMENT_DIR = Path(__file__).parent
REDUCE_TOOL = str(EXPERIMENT_DIR.parent / "reduce_pipeline_llc.py")

STABLE_WORKTREE = EXPERIMENT_DIR / "worktree_stable"
FALLBACK_WORKTREE = EXPERIMENT_DIR / "worktree_fallback"
TEST_FILES_DIR = EXPERIMENT_DIR / "test_files"
RESULTS_FILE = EXPERIMENT_DIR / "results.jsonl"
CHECKPOINT_FILE = EXPERIMENT_DIR / "checkpoint.json"
LOG_FILE = EXPERIMENT_DIR / "experiment.log"

STABLE_LLC = STABLE_WORKTREE / "build" / "bin" / "llc"
STABLE_FILECHECK = STABLE_WORKTREE / "build" / "bin" / "FileCheck"
FALLBACK_LLC = FALLBACK_WORKTREE / "build" / "bin" / "llc"

LLC_TIMEOUT = 60
REDUCE_TIMEOUT = 300
BUILD_TIMEOUT = 600

FILTER_PASSES = frozenset({
    'targetlibinfo', 'targetpassconfig', 'machinemoduleinfo',
    'machine-branch-prob', 'regalloc-evict', 'regalloc-priority',
})

FILE_TO_PASS = {
    "PeepholeOptimizer.cpp": "peephole-opt",
    "TailDuplicator.cpp": "early-tailduplication",
    "RegisterCoalescer.cpp": "register-coalescer",
    "PHIElimination.cpp": "phi-node-elimination",
    "TwoAddressInstructionPass.cpp": "twoaddressinstruction",
    "BranchFolding.cpp": "branch-folder",
    "MachineCSE.cpp": "machine-cse",
    "MachineSink.cpp": "machine-sink",
    "MachineLICM.cpp": "machinelicm",
    "BreakFalseDeps.cpp": "break-false-deps",
    "MachineScheduler.cpp": "machine-scheduler",
    "StackSlotColoring.cpp": "stack-slot-coloring",
    "VirtRegMap.cpp": "virtregrewriter",
    "RegAllocGreedy.cpp": "greedy",
    "RegAllocFast.cpp": "regallocfast",
    "PrologEpilogInserter.cpp": "prologepilog",
    "ProcessImplicitDefs.cpp": "processimpdefs",
    "XRayInstrumentation.cpp": "xray-instrumentation",
    "FixupStatepointCallerSaved.cpp": "fixup-statepoint-caller-saved",
    "MachineCycleAnalysis.cpp": "print-machine-cycles",
    "RemoveLoadsIntoFakeUses.cpp": "remove-loads-into-fake-uses",
    "X86FrameLowering.cpp": "prologepilog",
    "X86FixupInstTuning.cpp": "x86-fixup-inst-tuning",
    "X86IndirectBranchTracking.cpp": "x86-indirect-branch-tracking",
    "X86ExpandPseudo.cpp": "x86-pseudo",
    "X86FloatingPoint.cpp": "x86-fptransforms",
    "X86FixupVectorConstants.cpp": "x86-fixup-vector-constants",
    "X86WinEHUnwindV2.cpp": "x86-wineh-unwindv2",
    "X86WinEHState.cpp": "x86-wineh-state",
    "X86CompressEVEX.cpp": "x86-compress-evex",
}

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ─── Dataset Loading ─────────────────────────────────────────────────────────

def load_and_filter_dataset():
    """Load JSONL and filter to applicable backend machine-pass instances."""
    instances = []
    with open(DATASET_PATH) as f:
        for line in f:
            d = json.loads(line)
            test_cases = d.get('test_cases', [])
            buggy_files = d.get('buggy_files', [])
            test_patch = d.get('test_patch', '')

            codegen_tests = [tc for tc in test_cases
                            if tc.startswith('llvm/test/CodeGen/')
                            and (tc.endswith('.ll') or tc.endswith('.mir'))]
            if not codegen_tests:
                continue

            machine_buggy = [bf for bf in buggy_files
                            if (bf.endswith('.cpp') or bf.endswith('.h'))
                            and ('lib/CodeGen/' in bf or 'lib/Target/' in bf)
                            and not any(x in bf for x in [
                                'SelectionDAG', 'GlobalISel', 'Legalizer',
                                'InstructionSelector', '/MC/', 'MCInstLower',
                                'AsmPrinter', 'AsmParser'])]
            if not machine_buggy:
                continue

            target = None
            for tc in codegen_tests:
                parts = tc[len('llvm/test/CodeGen/'):].split('/')
                if parts[0][0].isupper():
                    target = parts[0]
                    break

            run_pass_matches = re.findall(r'run-pass[= ]+([^\s\\|\"]+)', test_patch)

            instances.append({
                'instance_id': d['instance_id'],
                'base_commit': d['base_commit'],
                'target': target,
                'test_cases': codegen_tests,
                'buggy_files': machine_buggy,
                'test_patch': test_patch,
                'patch': d.get('patch', ''),
                'problem_statement': d.get('problem_statement', ''),
                'run_pass_ground_truth': run_pass_matches,
            })

    log.info(f"Loaded {len(instances)} applicable instances")
    return instances


# ─── Test File Extraction ────────────────────────────────────────────────────

def extract_test_file(instance):
    """Extract test file content from git or diff.

    Returns path to the extracted file, or None on failure.
    """
    test_patch = instance['test_patch']
    test_cases = instance['test_cases']
    instance_id = instance['instance_id']

    target_file = None
    for tc in test_cases:
        if tc.endswith('.ll') or tc.endswith('.mir'):
            target_file = tc
            break
    if not target_file:
        return None

    filename = target_file.split('/')[-1]
    ext = '.mir' if filename.endswith('.mir') else '.ll'
    output_path = TEST_FILES_DIR / f"{instance_id}{ext}"

    if output_path.exists():
        return output_path

    fix_commit = re.match(r'From ([0-9a-f]+)', test_patch)
    if fix_commit:
        fix_hash = fix_commit.group(1)
        try:
            result = subprocess.run(
                ['git', 'show', f'{fix_hash}:{target_file}'],
                capture_output=True, text=True, timeout=10,
                cwd='/data/chenziang3/project/llvm-project')
            if result.returncode == 0 and result.stdout.strip():
                output_path.write_text(result.stdout)
                return output_path
        except (subprocess.TimeoutExpired, OSError):
            pass

    lines = test_patch.split('\n')
    in_target = False
    content_lines = []

    for line in lines:
        if line.startswith('+++ b/') and target_file in line:
            in_target = True
            continue
        if in_target:
            if line.startswith('diff --git'):
                break
            if line.startswith('@@'):
                continue
            if line.startswith('+'):
                content_lines.append(line[1:])
            elif line.startswith(' '):
                content_lines.append(line[1:])
            elif line.startswith('-'):
                continue

    if not content_lines:
        return None

    content = '\n'.join(content_lines)
    if not content.endswith('\n'):
        content += '\n'

    output_path.write_text(content)
    return output_path


def parse_run_line(test_content):
    """Parse RUN lines from test file to extract llc flags and FileCheck config.
    Returns list of (llc_flags, check_prefix, is_crash_test, run_pass_in_cmd) tuples."""
    raw_lines = re.findall(r'[;#] RUN:(.+)', test_content)
    if not raw_lines:
        return []

    joined = []
    current = ''
    for line in raw_lines:
        line = line.strip()
        if current:
            current += ' ' + line
        else:
            current = line
        if current.endswith('\\'):
            current = current[:-1]
        else:
            joined.append(current)
            current = ''
    if current:
        joined.append(current)

    results = []
    for run_line in joined:
        if 'llc' not in run_line:
            continue

        is_crash_test = 'not llc' in run_line or 'not --crash' in run_line

        run_pass_in_cmd = None
        rp_match = re.search(r'-run-pass[= ]+([^\s\\|]+)', run_line)
        if rp_match:
            run_pass_in_cmd = rp_match.group(1).split(',')

        llc_part = run_line.split('|')[0] if '|' in run_line else run_line
        llc_part = re.sub(r'^not\s+--crash\s+', '', llc_part)
        llc_part = re.sub(r'^not\s+', '', llc_part)
        llc_part = re.sub(r'llc\s*', '', llc_part, count=1)
        llc_part = re.sub(r'<\s*%s', '', llc_part)
        llc_part = re.sub(r'\s+%s\b', '', llc_part)
        llc_part = re.sub(r'-o\s+(%t\S*|-|/dev/null)', '', llc_part)
        llc_part = re.sub(r'2>&1', '', llc_part)
        llc_part = re.sub(r'2>\s*%t\S*', '', llc_part)

        llc_flags = []
        tokens = llc_part.split()
        for tok in tokens:
            tok = tok.strip()
            if tok and tok.startswith('-') and tok != '-':
                llc_flags.append(tok)

        check_prefix = None
        check_match = re.search(r'--check-prefix[= ]+(\S+)', run_line)
        if check_match:
            check_prefix = check_match.group(1)
        check_match = re.search(r'--check-prefixes[= ]+(\S+)', run_line)
        if check_match:
            check_prefix = check_match.group(1)

        results.append((llc_flags, check_prefix, is_crash_test, run_pass_in_cmd))

    return results


# ─── LLC Execution ───────────────────────────────────────────────────────────

def run_llc(llc_binary, input_file, flags, timeout=LLC_TIMEOUT):
    """Run llc and return (rc, stdout, stderr)."""
    cmd = [str(llc_binary)] + flags + ['-o', '/dev/null', str(input_file)]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        return result.returncode, \
               result.stdout.decode('utf-8', errors='replace'), \
               result.stderr.decode('utf-8', errors='replace')
    except subprocess.TimeoutExpired:
        return -9, '', 'TIMEOUT'
    except OSError as e:
        return -1, '', f'OSError: {e}'


def run_llc_to_stdout(llc_binary, input_file, flags, timeout=LLC_TIMEOUT):
    """Run llc outputting to stdout (for piping to FileCheck)."""
    cmd = [str(llc_binary)] + flags + ['-o', '-', str(input_file)]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        return result.returncode, \
               result.stdout.decode('utf-8', errors='replace'), \
               result.stderr.decode('utf-8', errors='replace')
    except subprocess.TimeoutExpired:
        return -9, '', 'TIMEOUT'
    except OSError as e:
        return -1, '', f'OSError: {e}'


def run_filecheck(filecheck_binary, check_file, input_text, check_prefix=None):
    """Run FileCheck on input_text against check_file. Returns rc."""
    cmd = [str(filecheck_binary), str(check_file)]
    if check_prefix:
        cmd.append(f'--check-prefix={check_prefix}')
    try:
        result = subprocess.run(cmd, input=input_text, capture_output=True,
                               text=True, timeout=30)
        return result.returncode
    except (subprocess.TimeoutExpired, OSError):
        return -1


def is_real_crash(rc, stderr):
    """Distinguish real crashes from error exits."""
    if rc == 0:
        return False
    if 'not registered' in stderr:
        return False
    if 'input language must be' in stderr:
        return False
    if rc >= 128 or rc < 0:
        return True
    if 'Assertion' in stderr and 'failed' in stderr:
        return True
    if 'LLVM ERROR' in stderr:
        return True
    if 'UNREACHABLE' in stderr:
        return True
    if 'Stack dump:' in stderr:
        return True
    return False


def is_isel_crash(stderr):
    """Check if crash occurs during instruction selection (pre-machine-pass)."""
    isel_indicators = ['ISelLowering', 'ISelDAGToDAG', 'SelectionDAGISel',
                       'SelectionDAGBuilder', 'GlobalISel', 'IRTranslator',
                       'InstructionSelector', 'LegalizeDAG', 'LegalizeTypes']
    for indicator in isel_indicators:
        if indicator in stderr:
            return True
    return False


# ─── Crash Bug Handling ──────────────────────────────────────────────────────

def run_crash_reduce(llc_binary, test_file, flags, error_pattern=None):
    """Run reduce_pipeline_llc.py on a crash bug. Returns dict with results."""
    cmd = [sys.executable, REDUCE_TOOL,
           f'--llc-binary={llc_binary}',
           f'--input={test_file}']
    if error_pattern:
        cmd.append(f'--error-pattern={error_pattern}')
    cmd.extend(flags)

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=REDUCE_TIMEOUT)
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        return {'error': 'reduce_timeout', 'time': REDUCE_TIMEOUT}
    except OSError as e:
        return {'error': f'OSError: {e}', 'time': 0}

    output = result.stdout + result.stderr

    passes = []
    if 'Minimal pass set' in output:
        in_result = False
        for line in output.split('\n'):
            if 'Minimal pass set' in line:
                in_result = True
                continue
            if in_result:
                m = re.match(r'\s*\d+:\s*(\S+)', line)
                if m:
                    passes.append(m.group(1))
                elif line.strip() and not line.startswith(' '):
                    break

    if not passes:
        m = re.search(r'--run-pass=([^\s]+)', output)
        if m:
            passes = m.group(1).split(',')

    if not passes:
        m = re.search(r'^Trigger pass:\s+(\S+)', output, re.MULTILINE)
        if m:
            passes = [m.group(1)]

    if not passes and 'pre-ISel crash' in output:
        m = re.search(r'Earliest machine pass:\s+(\S+)', output)
        if m:
            passes = [f'PRE_ISEL_CRASH(before {m.group(1)})']

    input_count = None
    m = re.search(r'Starting with (\d+) passes', output)
    if m:
        input_count = int(m.group(1))

    ddmin_tests = None
    m = re.search(r'DDMIN: (\d+) tests', output)
    if m:
        ddmin_tests = int(m.group(1))

    return {
        'passes': passes if passes else None,
        'input_pass_count': input_count,
        'ddmin_tests': ddmin_tests,
        'time': elapsed,
        'error': None if passes else 'no_result_parsed',
        'raw_output': output[-500:] if not passes else None,
    }


# ─── Correctness Bug Handling ────────────────────────────────────────────────

def run_bisect_correctness(llc_binary, filecheck_binary, test_file, flags,
                           check_prefix=None):
    """Use opt-bisect-limit to find the trigger pass for a correctness bug."""
    start = time.time()

    rc, stdout, stderr = run_llc_to_stdout(
        llc_binary, test_file, flags + ['-opt-bisect-limit=999999'])
    if rc != 0:
        return {'error': 'llc_crash_during_bisect', 'trigger_pass': None,
                'time': time.time() - start}

    bisect_lines = [l for l in stderr.split('\n') if 'BISECT:' in l]
    if not bisect_lines:
        return {'error': 'no_bisect_output', 'trigger_pass': None,
                'time': time.time() - start}
    max_n = len(bisect_lines)

    fc_rc = run_filecheck(filecheck_binary, test_file, stdout, check_prefix)
    if fc_rc == 0:
        return {'error': 'filecheck_passes_at_max', 'trigger_pass': None,
                'time': time.time() - start}

    rc0, stdout0, _ = run_llc_to_stdout(
        llc_binary, test_file, flags + ['-opt-bisect-limit=0'])
    if rc0 == 0:
        fc_rc0 = run_filecheck(filecheck_binary, test_file, stdout0, check_prefix)
        if fc_rc0 != 0:
            return {'error': 'mandatory_pass_bug', 'trigger_pass': None,
                    'max_bisect': max_n, 'time': time.time() - start}

    lo, hi = 0, max_n

    while lo < hi:
        mid = (lo + hi) // 2
        rc, stdout, stderr = run_llc_to_stdout(
            llc_binary, test_file, flags + [f'-opt-bisect-limit={mid}'])
        if rc != 0:
            hi = mid
            continue

        fc_rc = run_filecheck(filecheck_binary, test_file, stdout, check_prefix)
        if fc_rc != 0:
            hi = mid
        else:
            lo = mid + 1

    elapsed = time.time() - start

    trigger_pass = None
    if lo > 0 and lo <= max_n:
        rc, stdout, stderr = run_llc_to_stdout(
            llc_binary, test_file, flags + [f'-opt-bisect-limit={lo}'])
        for line in stderr.split('\n'):
            if f'BISECT: running pass ({lo})' in line:
                m = re.search(r'BISECT: running pass \(\d+\) (.+?) on', line)
                if m:
                    trigger_pass = m.group(1).strip()
                break
        if not trigger_pass:
            rc2, _, stderr2 = run_llc_to_stdout(
                llc_binary, test_file, flags + [f'-opt-bisect-limit={max_n}'])
            for line in stderr2.split('\n'):
                if f'({lo})' in line:
                    m = re.search(r'BISECT: running pass \(\d+\) (.+?) on', line)
                    if m:
                        trigger_pass = m.group(1).strip()
                    break

    return {
        'trigger_pass': trigger_pass,
        'bisect_limit': lo,
        'max_bisect': max_n,
        'time': elapsed,
        'error': None if trigger_pass else 'trigger_not_identified',
    }


def run_correctness_reduce(llc_binary, filecheck_binary, test_file, flags,
                           check_prefix=None):
    """Reduce a correctness bug using test-script mode."""
    tmpdir = tempfile.mkdtemp(prefix='correctness_reduce_')

    fc_cmd = f'{filecheck_binary} {test_file}'
    if check_prefix:
        fc_cmd += f' --check-prefix={check_prefix}'

    script_path = os.path.join(tmpdir, 'check.sh')
    script_content = f'''#!/bin/bash
if [ "$PASSES" = "FULL_PIPELINE" ] || [ -z "$PASSES" ]; then
    ASM=$("$LLC_BINARY" -o - "$INPUT_FILE" $EXTRA_ARGS 2>/dev/null)
else
    "$LLC_BINARY" --run-pass="$PASSES" -o "$OUTPUT_DIR/out.mir" "$INPUT_FILE" $EXTRA_ARGS 2>/dev/null || exit 1
    ASM=$("$LLC_BINARY" -o - "$OUTPUT_DIR/out.mir" $EXTRA_ARGS 2>/dev/null)
fi
[ $? -ne 0 ] && exit 1
echo "$ASM" | {fc_cmd}
if [ $? -ne 0 ]; then
    exit 0   # interesting: FileCheck fails = bug is present
else
    exit 1   # not interesting: FileCheck passes = bug gone
fi
'''
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

    cmd = [sys.executable, REDUCE_TOOL,
           f'--llc-binary={llc_binary}',
           f'--input={test_file}',
           f'--test-script={script_path}',
           '--no-verify']
    cmd.extend(flags)

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=REDUCE_TIMEOUT)
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return {'error': 'reduce_timeout', 'time': REDUCE_TIMEOUT}
    except OSError as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return {'error': f'OSError: {e}', 'time': 0}

    output = result.stdout + result.stderr

    passes = []
    method = 'test-script-ddmin'

    m = re.search(r'Trigger: pass #(\d+)/(\d+): (.+)', output)
    if m:
        passes = [m.group(3).strip()]
        method = 'opt-bisect-limit'

    if not passes and 'Minimal pass set' in output:
        in_result = False
        for line in output.split('\n'):
            if 'Minimal pass set' in line:
                in_result = True
                continue
            if in_result:
                m2 = re.match(r'\s*\d+:\s*(\S+)', line)
                if m2:
                    passes.append(m2.group(1))
                elif line.strip() and not line.startswith(' '):
                    break

    if not passes:
        m = re.search(r'PASSES=([^\s]+)', output)
        if m:
            passes = m.group(1).split(',')

    input_count = None
    m = re.search(r'Starting with (\d+) passes', output)
    if m:
        input_count = int(m.group(1))

    ddmin_tests = None
    m = re.search(r'DDMIN: (\d+) tests', output)
    if m:
        ddmin_tests = int(m.group(1))

    shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        'passes': passes if passes else None,
        'method': method,
        'input_pass_count': input_count,
        'ddmin_tests': ddmin_tests,
        'time': elapsed,
        'error': None if passes else 'no_result_parsed',
        'raw_output': output[-500:] if not passes else None,
    }


# ─── Ground Truth & Accuracy ────────────────────────────────────────────────

def map_buggy_file_to_pass(buggy_files):
    """Map buggy source files to expected pass names."""
    for bf in buggy_files:
        filename = bf.split('/')[-1]
        if filename in FILE_TO_PASS:
            pass_name = FILE_TO_PASS[filename]
            if pass_name:
                return pass_name
    return None


def classify_accuracy(our_passes, gt_passes_from_test, expected_pass_from_file):
    """Classify accuracy of our result vs ground truth."""
    if not our_passes:
        return "no_result"

    if gt_passes_from_test:
        gt_set = set(gt_passes_from_test)
        our_set = set(our_passes)
        if our_set == gt_set:
            return "exact"
        elif gt_set.issubset(our_set):
            return "superset"
        elif our_set.issubset(gt_set):
            return "subset"
        elif expected_pass_from_file and expected_pass_from_file in our_set:
            return "contains_target"
        else:
            return "wrong"

    if expected_pass_from_file:
        if expected_pass_from_file in our_passes:
            return "contains_target"
        if len(our_passes) == 1 and our_passes[0] == expected_pass_from_file:
            return "exact"
        return "produced"

    return "no_ground_truth"


# ─── Crash Signature Extraction ──────────────────────────────────────────────

def extract_crash_signature(stderr):
    """Extract crash type and signature from llc stderr."""
    crash_type = None
    signature = None

    if '*** Bad machine code:' in stderr:
        crash_type = 'verifier'
        for line in stderr.split('\n'):
            if '*** Bad machine code:' in line:
                signature = line.strip().strip('*').strip()[:200]
                break
    elif 'Assertion' in stderr and 'failed' in stderr:
        crash_type = 'assertion'
        for line in stderr.split('\n'):
            if 'Assertion' in line and 'failed' in line:
                m = re.search(r"Assertion [`'](.+?)[`'] failed", line)
                if m:
                    signature = m.group(1)[:200]
                else:
                    idx = line.find('Assertion')
                    signature = line[idx:].strip()[:200] if idx >= 0 else line.strip()[:200]
                break
    elif 'LLVM ERROR' in stderr:
        crash_type = 'llvm_error'
        signature = stderr.split('LLVM ERROR:')[1].split('\n')[0].strip()[:100]
    elif 'UNREACHABLE' in stderr:
        crash_type = 'unreachable'
        lines = stderr.split('\n')
        for i, line in enumerate(lines):
            if 'UNREACHABLE' in line:
                if i > 0 and lines[i-1].strip() and 'Stack dump' not in lines[i-1]:
                    signature = lines[i-1].strip()[:200]
                else:
                    m = re.search(r'UNREACHABLE executed at .*/([^/]+:\d+)', line)
                    if m:
                        signature = f'UNREACHABLE at {m.group(1)}'
                    else:
                        signature = line.strip()[:120]
                break
    else:
        crash_type = 'signal'

    return crash_type, signature


# ─── Instance Processing ─────────────────────────────────────────────────────

def try_instance(llc_binary, filecheck_binary, test_file, run_configs, instance):
    """Try to reproduce and reduce a bug instance."""
    result = {
        'instance_id': instance['instance_id'],
        'test_file': str(test_file),
        'base_commit': instance['base_commit'],
        'target': instance['target'],
        'bug_type': None,
        'status': None,
        'crash_info': {'type': None, 'signature': None},
        'reduction_result': {
            'method': None, 'passes': None, 'input_pass_count': None,
            'result_pass_count': None, 'ddmin_tests': None,
            'bisect_limit': None, 'trigger_pass': None,
        },
        'timing': {'build_seconds': 0.0, 'reduce_seconds': 0.0, 'total_seconds': 0.0},
        'error_detail': None,
    }

    if not run_configs:
        result['status'] = 'no_run_line'
        result['error_detail'] = 'no llc RUN line found in test file'
        return result

    start_total = time.time()

    for llc_flags, check_prefix, is_crash_test, run_pass_in_cmd in run_configs:
        rc, stdout_llc, stderr_llc = run_llc(llc_binary, test_file, llc_flags)

        if rc != 0 and is_crash_test and is_real_crash(rc, stderr_llc):
            result['bug_type'] = 'crash'
            crash_type, signature = extract_crash_signature(stderr_llc)
            result['crash_info']['type'] = crash_type
            result['crash_info']['signature'] = signature

            if run_pass_in_cmd:
                result['status'] = 'already_reduced'
                result['reduction_result']['method'] = 'run_pass_in_test'
                result['reduction_result']['passes'] = run_pass_in_cmd
                result['reduction_result']['result_pass_count'] = len(run_pass_in_cmd)
                break

            if is_isel_crash(stderr_llc):
                result['status'] = 'isel_crash'
                result['error_detail'] = 'crash in ISel, not reducible by pass pipeline tool'
                break

            error_pattern = result['crash_info']['signature']
            reduce_result = run_crash_reduce(str(llc_binary), str(test_file),
                                             llc_flags, error_pattern)
            result['timing']['reduce_seconds'] = reduce_result.get('time', 0)

            if reduce_result.get('passes'):
                result['status'] = 'reduced'
                result['reduction_result']['method'] = 'ddmin'
                result['reduction_result']['passes'] = reduce_result['passes']
                result['reduction_result']['input_pass_count'] = reduce_result.get('input_pass_count')
                result['reduction_result']['result_pass_count'] = len(reduce_result['passes'])
                result['reduction_result']['ddmin_tests'] = reduce_result.get('ddmin_tests')
            else:
                result['status'] = 'crash_reduce_failed'
                result['error_detail'] = reduce_result.get('error', 'unknown')
            break

        elif rc != 0 and not is_crash_test and is_real_crash(rc, stderr_llc):
            result['bug_type'] = 'crash'
            crash_type, signature = extract_crash_signature(stderr_llc)
            result['crash_info']['type'] = crash_type
            result['crash_info']['signature'] = signature

            if is_isel_crash(stderr_llc):
                result['status'] = 'isel_crash'
                result['error_detail'] = 'crash in ISel, not reducible by pass pipeline tool'
                break

            error_pattern = result['crash_info']['signature']
            reduce_result = run_crash_reduce(str(llc_binary), str(test_file),
                                             llc_flags, error_pattern)
            result['timing']['reduce_seconds'] = reduce_result.get('time', 0)

            if reduce_result.get('passes'):
                result['status'] = 'reduced'
                result['reduction_result']['method'] = 'ddmin'
                result['reduction_result']['passes'] = reduce_result['passes']
                result['reduction_result']['input_pass_count'] = reduce_result.get('input_pass_count')
                result['reduction_result']['result_pass_count'] = len(reduce_result['passes'])
                result['reduction_result']['ddmin_tests'] = reduce_result.get('ddmin_tests')
            else:
                result['status'] = 'crash_reduce_failed'
                result['error_detail'] = reduce_result.get('error', 'unknown')
            break

        elif rc == 0 and is_crash_test:
            continue

        elif rc != 0 and is_crash_test and not is_real_crash(rc, stderr_llc):
            if 'LLVM ERROR' in stderr_llc:
                result['bug_type'] = 'crash'
                result['crash_info']['type'] = 'llvm_error'
                result['crash_info']['signature'] = stderr_llc.split('LLVM ERROR:')[1].split('\n')[0].strip()[:100]
                if run_pass_in_cmd:
                    result['status'] = 'already_reduced'
                    result['reduction_result']['method'] = 'run_pass_in_test'
                    result['reduction_result']['passes'] = run_pass_in_cmd
                    result['reduction_result']['result_pass_count'] = len(run_pass_in_cmd)
                else:
                    result['status'] = 'already_reduced'
                    result['reduction_result']['method'] = 'error_exit'
                    result['reduction_result']['passes'] = ['full_pipeline']
                    result['reduction_result']['result_pass_count'] = 1
                break
            continue

        elif rc != 0 and not is_crash_test and not is_real_crash(rc, stderr_llc):
            continue

        elif rc == 0 and not is_crash_test:
            rc2, llc_stdout, _ = run_llc_to_stdout(llc_binary, test_file, llc_flags)
            if rc2 != 0:
                continue

            fc_rc = run_filecheck(filecheck_binary, test_file, llc_stdout, check_prefix)
            if fc_rc != 0:
                result['bug_type'] = 'correctness'

                if run_pass_in_cmd:
                    result['status'] = 'already_reduced'
                    result['reduction_result']['method'] = 'run_pass_in_test'
                    result['reduction_result']['passes'] = run_pass_in_cmd
                    result['reduction_result']['result_pass_count'] = len(run_pass_in_cmd)
                    break

                bisect_result = run_bisect_correctness(
                    llc_binary, filecheck_binary, test_file, llc_flags, check_prefix)
                result['timing']['reduce_seconds'] = bisect_result.get('time', 0)

                if bisect_result.get('trigger_pass'):
                    result['status'] = 'bisected'
                    result['reduction_result']['method'] = 'opt-bisect-limit'
                    result['reduction_result']['trigger_pass'] = bisect_result['trigger_pass']
                    result['reduction_result']['passes'] = [bisect_result['trigger_pass']]
                    result['reduction_result']['result_pass_count'] = 1
                    result['reduction_result']['bisect_limit'] = bisect_result.get('bisect_limit')
                elif bisect_result.get('error') == 'mandatory_pass_bug':
                    result['status'] = 'bisect_failed'
                    result['error_detail'] = 'mandatory_pass_bug'
                else:
                    reduce_result = run_correctness_reduce(
                        str(llc_binary), str(filecheck_binary), str(test_file),
                        llc_flags, check_prefix)
                    result['timing']['reduce_seconds'] += reduce_result.get('time', 0)

                    if reduce_result.get('passes'):
                        result['status'] = 'reduced'
                        result['reduction_result']['method'] = reduce_result.get('method', 'test-script-ddmin')
                        result['reduction_result']['passes'] = reduce_result['passes']
                        result['reduction_result']['input_pass_count'] = reduce_result.get('input_pass_count')
                        result['reduction_result']['result_pass_count'] = len(reduce_result['passes'])
                        result['reduction_result']['ddmin_tests'] = reduce_result.get('ddmin_tests')
                    else:
                        result['status'] = 'bisect_failed'
                        result['error_detail'] = bisect_result.get('error', 'unknown')
                        result['reduction_result']['fallback_error'] = reduce_result.get('error')
                break
            else:
                continue

    if result['status'] is None:
        result['status'] = 'no_bug_found'

    if result['status'] == 'no_bug_found' and run_configs:
        llc_flags, _, _, _ = run_configs[0]
        rc_v, _, stderr_v = run_llc(llc_binary, test_file,
                                     llc_flags + ['-verify-machineinstrs'])
        if rc_v != 0:
            result['bug_type'] = 'crash'
            result['crash_info']['type'] = 'verifier'
            for line in stderr_v.split('\n'):
                if 'Assertion' in line or 'error:' in line.lower():
                    result['crash_info']['signature'] = line.strip()[:200]
                    break
            reduce_result = run_crash_reduce(
                str(llc_binary), str(test_file),
                llc_flags + ['-verify-machineinstrs'],
                result['crash_info']['signature'])
            result['timing']['reduce_seconds'] = reduce_result.get('time', 0)
            if reduce_result.get('passes'):
                result['status'] = 'reduced'
                result['reduction_result']['method'] = 'ddmin'
                result['reduction_result']['passes'] = reduce_result['passes']
                result['reduction_result']['input_pass_count'] = reduce_result.get('input_pass_count')
                result['reduction_result']['result_pass_count'] = len(reduce_result['passes'])
                result['reduction_result']['ddmin_tests'] = reduce_result.get('ddmin_tests')
            else:
                result['status'] = 'crash_reduce_failed'
                result['error_detail'] = reduce_result.get('error', 'unknown')

    result['timing']['total_seconds'] = time.time() - start_total
    return result


# ─── Fallback Build Management ───────────────────────────────────────────────

def checkout_and_rebuild(commit):
    """Checkout a specific commit in fallback worktree and rebuild llc."""
    worktree_path = FALLBACK_WORKTREE
    try:
        subprocess.run(['git', 'checkout', commit],
                      cwd=worktree_path, capture_output=True, timeout=60)
        result = subprocess.run(['ninja', '-C', 'build', 'llc', 'FileCheck'],
                               cwd=worktree_path, capture_output=True,
                               text=True, timeout=BUILD_TIMEOUT)
        if result.returncode == 0:
            return True
        log.warning(f"  Incremental build failed, attempting clean build...")
        build_dir = Path(worktree_path) / 'build'
        subprocess.run(['ninja', '-C', str(build_dir), '-t', 'clean'],
                      capture_output=True, timeout=120)
        result = subprocess.run(['ninja', '-C', 'build', 'llc', 'FileCheck'],
                               cwd=worktree_path, capture_output=True,
                               text=True, timeout=BUILD_TIMEOUT * 3)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


# ─── Checkpoint Management ───────────────────────────────────────────────────

def load_checkpoint():
    """Load checkpoint or return default."""
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {"last_completed_index": -1}


def save_checkpoint(index):
    """Save checkpoint."""
    CHECKPOINT_FILE.write_text(json.dumps({"last_completed_index": index}))


def write_result(result):
    """Append result to JSONL file."""
    with open(RESULTS_FILE, 'a') as f:
        f.write(json.dumps(result) + '\n')


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Large-scale reduce_pipeline_llc validation")
    parser.add_argument('--start-index', type=int, default=None,
                        help='Override start index (ignores checkpoint)')
    parser.add_argument('--max-instances', type=int, default=None,
                        help='Max instances to process')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only extract tests and check crash, don\'t reduce')
    args = parser.parse_args()

    if not STABLE_LLC.exists():
        log.error(f"Stable LLC not found: {STABLE_LLC}")
        sys.exit(1)
    if not STABLE_FILECHECK.exists():
        log.error(f"Stable FileCheck not found: {STABLE_FILECHECK}")
        sys.exit(1)

    TEST_FILES_DIR.mkdir(parents=True, exist_ok=True)

    instances = load_and_filter_dataset()

    checkpoint = load_checkpoint()
    start_idx = args.start_index if args.start_index is not None else checkpoint['last_completed_index'] + 1

    log.info(f"Starting from index {start_idx}, {len(instances)} total instances")

    processed = 0
    for idx, instance in enumerate(instances):
        if idx < start_idx:
            continue
        if args.max_instances and processed >= args.max_instances:
            break

        log.info(f"[{idx}/{len(instances)}] Processing {instance['instance_id']}")

        test_file = extract_test_file(instance)
        if not test_file or not test_file.exists():
            log.warning(f"  Failed to extract test file, skipping")
            result = {
                'instance_id': instance['instance_id'],
                'status': 'extract_failed',
                'error_detail': 'could not extract test file from patch',
            }
            write_result(result)
            save_checkpoint(idx)
            processed += 1
            continue

        test_content = test_file.read_text()
        run_configs = parse_run_line(test_content)

        result = try_instance(STABLE_LLC, STABLE_FILECHECK, test_file,
                             run_configs, instance)
        result['build_used'] = 'stable'

        if result.get('status') == 'no_bug_found' and FALLBACK_LLC.exists():
            log.info(f"  No bug on stable, trying fallback at {instance['base_commit'][:12]}")
            if checkout_and_rebuild(instance['base_commit']):
                result = try_instance(FALLBACK_LLC, STABLE_FILECHECK, test_file,
                                     run_configs, instance)
                result['build_used'] = 'fallback'
            else:
                log.warning(f"  Fallback build failed")
                result['error_detail'] = 'fallback_build_failed'

        if result.get('status') == 'crash_reduce_failed' and FALLBACK_LLC.exists():
            log.info(f"  Crash reduce failed on stable, trying fallback at {instance['base_commit'][:12]}")
            if checkout_and_rebuild(instance['base_commit']):
                result = try_instance(FALLBACK_LLC, STABLE_FILECHECK, test_file,
                                     run_configs, instance)
                result['build_used'] = 'fallback'
            else:
                log.warning(f"  Fallback build failed")

        expected_pass = map_buggy_file_to_pass(instance['buggy_files'])
        gt_passes = None
        gt_source = 'default_pipeline'
        if instance['run_pass_ground_truth']:
            gt_passes = []
            for p in instance['run_pass_ground_truth']:
                gt_passes.extend(p.split(','))
            gt_source = 'run_pass_line'

        result['ground_truth'] = {
            'buggy_files': instance['buggy_files'],
            'expected_pass': expected_pass,
            'run_pass_from_test': gt_passes,
            'source': gt_source,
        }

        our_passes = result.get('reduction_result', {}).get('passes')
        result['accuracy'] = classify_accuracy(our_passes, gt_passes, expected_pass)

        status = result.get('status', '?')
        if status == 'reduced':
            passes = result['reduction_result']['passes']
            log.info(f"  REDUCED: {len(passes)} passes -> {passes}")
        elif status == 'bisected':
            tp = result['reduction_result']['trigger_pass']
            log.info(f"  BISECTED: trigger_pass={tp}")
        elif status == 'no_bug_found':
            log.info(f"  NO BUG FOUND on this build")
        else:
            log.info(f"  Status: {status}")

        write_result(result)
        save_checkpoint(idx)
        processed += 1

    log.info(f"Experiment complete. Processed {processed} instances.")
    log.info(f"Results written to: {RESULTS_FILE}")


if __name__ == '__main__':
    main()
