#!/usr/bin/env python3

"""
Simplified LLVM Backend Crash Point Finder with Custom Test Script

This script delegates ALL execution (including pass sequence extraction) to a
user-provided test script (interestingtest.sh).

Test script requirements:
- Must accept: <input_file> [--stop-before=PASS] [--run-pass=PASSES]
- When not -debug-pass: return 0 (correct), 1 (target crash), other (noise)
- stdout line 1 (when -debug-pass): space-separated list of passes
- Return codes:
    0: Correct behavior (no crash)
    1: Target crash/miscompile detected
    other: Neither target nor correct (noise/other error)

Algorithm:
1. Extract pass sequence by calling test script with -debug-pass=Arguments
2. Linear probe backward through passes with --stop-before
3. DDMIN reduction on the effective pass list
4. Output diagnostic commands
"""

import argparse
import os
import subprocess
import sys


# ============================================================================
# Pass Sequence Extraction via Test Script
# ============================================================================


def extract_pass_sequence(test_script, input_file):
    """Extract the full pass sequence by calling test script
    
    Args:
        test_script: Path to interestingtest.sh
        input_file: Input file (.ll or .mir)
    
    Returns:
        List of pass names (strings), or None if extraction fails
    """
    run_args = [test_script, input_file]
    
    result = subprocess.run(run_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_text = result.stdout.decode()
    
    # First line should contain space-separated pass list
    lines = stdout_text.split('\n')
    if lines:
        first_line = lines[0].strip()
        if first_line:
            # delete the first two because they are not passes but some debug info about the pass manager
            passes = [p.strip() for p in first_line.split() if p.strip()]
            return passes[2:] if len(passes) > 2 else []
    
    return None


def process_pass_sequence(passes, deduplicate=True):
    """Clean and process pass sequence."""
    # Strip dashes and filter out print passes
    cleaned = [p.lstrip('-') for p in passes 
               if 'print' not in p.lstrip('-').lower()]
    
    # Deduplicate while preserving order
    if deduplicate:
        seen = set()
        return [p for p in cleaned if not (p in seen or seen.add(p))]
    return cleaned


# ============================================================================
# Test Execution and Classification
# ============================================================================


class TestResult:
    """Represents the result of a single test."""
    
    # Classification constants
    TARGET = "TARGET"           # rc == 1 (target crash/miscompile)
    CORRECT = "CORRECT"         # rc == 0 (correct behavior)
    NOISE = "NOISE"             # other rc (neither target nor correct)
    
    def __init__(self, pass_idx, pass_name, rc, stderr_text, stdout_text):
        self.pass_idx = pass_idx
        self.pass_name = pass_name
        self.rc = rc
        self.stderr = stderr_text
        self.stdout = stdout_text
        self.classify()
    
    def classify(self):
        """Classify based on return code."""
        if self.rc == 1:
            self.classification = self.TARGET
        elif self.rc == 0:
            self.classification = self.CORRECT
        else:
            self.classification = self.NOISE
    
    def __str__(self):
        """Format result for display."""
        symbols = {
            self.TARGET: "🎯", 
            self.CORRECT: "✓ ",
            self.NOISE: "✗ ",
        }
        symbol = symbols.get(self.classification, "? ")
        return (f"[{self.pass_idx:3d}] {self.pass_name:<35} | "
                f"rc={self.rc:3d} | {self.classification:<10} {symbol}")


def run_test_script(test_script, input_file, output_file=None, stop_before=None, run_pass=None):
    """Run user-provided test script.
    
    Args:
        test_script: Path to interestingtest.sh
        input_file: Input file to test
        output_file: Output file (if any)
        stop_before: Pass name for --stop-before
        run_pass: Pass list for --run-pass
    
    Returns:
        (returncode, stdout_text, stderr_text) tuple
    """
    run_args = [test_script, input_file]
    if output_file:
        run_args.append(f"--output={output_file}")
    if stop_before:
        run_args.append(f"--stop-before={stop_before}")
    if run_pass:
        run_args.append(f"--run-pass={run_pass}")
    
    result = subprocess.run(run_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode, result.stdout.decode(), result.stderr.decode()


# ============================================================================
# Linear Probing: Find Earliest Trigger Pass
# ============================================================================


def find_earliest_trigger_pass(test_script, input_file, passes):
    """Find the earliest pass that still triggers the target crash.
    
    Scans backward through pass list using --stop-before.
    
    Args:
        test_script: Path to interestingtest.sh
        input_file: Input file (.ll or .mir)
        passes: List of pass names in execution order
    
    Returns:
        Tuple of (reduced_passes, results_list, first_correct_pass)
    """
    
    print("="*80)
    print("STEP #1: Find Earliest Trigger Pass")
    print("="*80)
    print()
    
    # Run full pipeline
    print("Running full pipeline to check if crash is reproducible...")
    rc, stdout, stderr = run_test_script(test_script, input_file)
    
    if rc != 1:
        print(f"ERROR: Full pipeline did not trigger target (returncode={rc})")
        print("Expected returncode=1 for target crash/miscompile")
        print("Aborting.")
        return (None, [], None)
    
    print(f"✓ Full pipeline triggers target crash (rc={rc})")
    print()
    
    # ========================================================================
    # Linear probing: scan backward through passes
    # ========================================================================
    print("Linear Probe Results (scanning backward through passes):")
    print("-"*80)
    print(f"{'Index':<5} {'Pass Name':<35} {'RC':<5} {'Classification':<10}")
    print("-"*80)
    
    results = []
    earliest_target_idx = -1
    first_correct_pass = None
    
    # Scan from last pass backward to first
    for idx in range(len(passes) - 1, -1, -1):
        pass_name = passes[idx]
        rc, stdout, stderr = run_test_script(test_script, input_file, stop_before=pass_name)
        result = TestResult(idx, pass_name, rc, stderr, stdout)
        results.append(result)
        print(result)
        
        # Track first CORRECT result (earliest recovery point)
        if result.classification == TestResult.CORRECT and first_correct_pass is None:
            earliest_target_idx = idx
            first_correct_pass = pass_name
    
    print()
    
    if earliest_target_idx < 0:
        print("ERROR: No pass triggers target crash with --stop-before")
        return (None, results, None)
    
    print(f"✓ Target crash first appears at: {first_correct_pass}")
    print(f"  (index {earliest_target_idx})")
    print()
    
    # Extract reduced pass list
    reduced_passes = passes[earliest_target_idx:]
    print(f"✓ Reduced pass list ({len(reduced_passes)} passes):")
    for i, p in enumerate(reduced_passes):
        print(f"  {i:3d}: {p}")
    print()
    
    return (reduced_passes, results, first_correct_pass)


# ============================================================================
# Backward Suffix Expansion (Phase 2.5)
# ============================================================================


def phase_backward_suffix_expansion(test_script, input_file, passes, target_pass):
    """Backward suffix expansion from target pass to find minimum prefix.
    
    Starting from the target pass, greedily expand backward to find the minimum
    set of preceding passes needed to reproduce the crash.
    
    Args:
        test_script: Path to interestingtest.sh
        input_file: Input file (.mir)
        passes: Full pass sequence
        target_pass: The pass to anchor on (first_correct_pass from Step #1)
    
    Returns:
        Minimized list of passes needed to reproduce crash, or None if failed
    """
    print("="*80)
    print("STEP #3: Backward Suffix Expansion (Finding Minimum Prefix)")
    print("="*80)
    print()
    print(f"Target pass: {target_pass}")
    print(f"Starting expansion from the end...\n")
    
    # Find target pass index
    if target_pass not in passes:
        print(f"ERROR: Target pass '{target_pass}' not found in pass sequence")
        return None
    
    target_idx = passes.index(target_pass)
    print(f"Target pass index: {target_idx}")
    print(f"Passes before target: {target_idx}")
    print()
    
    # Track forbidden passes (blacklist)
    forbidden_passes = set()
    effective_passes = None
    expansion_count = 0
    
    for expansion_size in range(1, target_idx + 2):
        # Calculate boundaries
        start_idx = max(0, target_idx - expansion_size + 1)
        candidate_full = passes[start_idx:target_idx + 1]
        
        # Filter out forbidden passes from candidate
        candidate = [p for p in candidate_full if p not in forbidden_passes]
        
        if not candidate:
            print(f"[Expansion #{expansion_size}] All candidates are forbidden, skip")
            continue
        
        expansion_count += 1
        pass_count = len(candidate)
        first_pass = candidate[0]
        last_pass = candidate[-1]
        
        print(f"[Expansion #{expansion_count}] Testing {pass_count} passes: {first_pass} ... {last_pass}")
        
        # Run test
        run_pass_str = ",".join(candidate)
        try:
            rc, stdout, stderr = run_test_script(test_script, input_file, run_pass=run_pass_str)
        except Exception as e:
            print(f"  ⚠️  ERROR - {e}")
            continue
        
        # Classify result
        if rc == 1:
            print(f"  ✓ REPRODUCED - Found minimum prefix\n")
            effective_passes = candidate
            break
        elif rc == 0:
            print(f"  ✓ CLEAN_RUN - Safe run, continue expanding\n")
            continue
        else:
            print(f"  ⚠️  NOISE_ERROR - Different error detected")
            # Mark the first (newly added) pass in candidate as forbidden
            forbidden_pass = candidate[0]
            forbidden_passes.add(forbidden_pass)
            print(f"     Added to blacklist: {forbidden_pass}\n")
            continue
    
    if effective_passes is None:
        print("ERROR: Could not find any pass sequence that reproduces the crash")
        if forbidden_passes:
            print(f"Forbidden passes (blacklist): {sorted(forbidden_passes)}")
        return None
    
    print(f"✓ Phase 2.5 result: {len(effective_passes)} passes needed")
    for i, p in enumerate(effective_passes):
        print(f"  {i:3d}: {p}")
    
    if forbidden_passes:
        print(f"\nForbidden passes encountered: {sorted(forbidden_passes)}")
    print()
    
    return effective_passes


# ============================================================================
# DDMIN Delta Debugging
# ============================================================================


def phase_ddmin_reduction(test_script, input_file, prefix_passes):
    """Apply DDMIN delta debugging to minimize pass list.
    
    Args:
        test_script: Path to interestingtest.sh
        input_file: Input file (.mir)
        prefix_passes: List of passes that reproduce the crash
    
    Returns:
        Minimized list of passes
    """
    print("="*80)
    print("STEP #4: DDMIN Delta Debugging Reduction")
    print("="*80)
    print()
    print(f"Starting with {len(prefix_passes)} passes")
    print(f"Goal: Remove unnecessary passes while maintaining crash reproduction\n")
    
    working_set = prefix_passes[:]  # Start with the effective prefix
    granularity = 2
    iteration = 0
    
    def test_pass_set(passes_list):
        """Test if a pass set reproduces the target using --run-pass."""
        run_pass_str = ",".join(passes_list)
        
        try:
            rc, stdout, stderr = run_test_script(test_script, input_file, run_pass=run_pass_str)
            return rc
        except Exception as e:
            print(f"ERROR: {e}")
            return 2
    
    while granularity <= len(working_set):
        iteration += 1
        print(f"Iteration #{iteration} (granularity={granularity}):")
        
        chunk_size = max(1, len(working_set) // granularity)
        removed_any = False
        
        for chunk_idx in range(granularity):
            # Define chunk boundaries
            chunk_start = chunk_idx * chunk_size
            if chunk_idx == granularity - 1:
                chunk_end = len(working_set)
            else:
                chunk_end = (chunk_idx + 1) * chunk_size
            
            # Test removing this chunk
            candidate = working_set[:chunk_start] + working_set[chunk_end:]
            
            if not candidate:
                print(f"  [Chunk {chunk_idx}] Empty candidate, skip")
                continue
            
            print(f"  [Chunk {chunk_idx}] Testing removal of [{chunk_start}:{chunk_end}] "
                  f"({chunk_end - chunk_start} passes)")
            
            rc = test_pass_set(candidate)
            
            if rc == 1:
                print(f"    → ✓ TARGET - Chunk can be removed")
                working_set = candidate
                removed_any = True
                break
            elif rc == 0:
                print(f"    → ✗ CORRECT - Chunk is necessary")
            else:
                print(f"    → ⚠️  NOISE - Chunk causes different error")
        
        if not removed_any:
            # Try finer granularity
            if granularity < len(working_set):
                granularity = min(granularity * 2, len(working_set))
                print(f"  → Increasing granularity to {granularity}\n")
            else:
                print(f"  → Cannot increase granularity further, done\n")
                break
        else:
            print(f"  → Restarting with smaller set ({len(working_set)} passes)\n")
            granularity = 2
    
    print()
    print(f"✓ Final minimized pass list ({len(working_set)} passes):")
    for i, p in enumerate(working_set):
        print(f"  {i:3d}: {p}")
    print()
    
    return working_set


# ============================================================================
# Main Function
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Simplified LLVM Backend Crash Point Finder. "
            "Uses a user-provided test script for ALL operations."
        ),
        epilog=(
            "The test script (interestingtest.sh) must support:\n"
            "  1. -debug-pass=Arguments: Extract and output pass list as first line\n"
            "  2. --stop-before=PASS: Stop before a specific pass\n"
            "  3. --run-pass=PASSES: Run only specific passes (comma-separated)\n"
            "\n"
            "Return codes from test script:\n"
            "  0: Correct behavior (no crash)\n"
            "  1: Target crash/miscompile detected\n"
            "  other: Neither (noise/other error)\n"
        )
    )
    parser.add_argument(
        "--test-script",
        action="store",
        dest="test_script",
        required=True,
        help="Path to interestingtest.sh (required)"
    )
    parser.add_argument(
        "--input",
        action="store",
        dest="input",
        required=True,
        help="Input file (.ll or .mir) (required)"
    )
    parser.add_argument(
        "--output",
        action="store",
        dest="output",
        help="Output file to write detailed diagnostic information"
    )
    parser.add_argument(
        "--output-mir",
        action="store",
        dest="output_mir",
        required=True,
        help="Output MIR file for intermediate IR at first recovery point (required)"
    )
    
    args, unknown = parser.parse_known_args()
    
    if unknown:
        print(f"WARNING: Unknown arguments ignored: {unknown}\n")
    
    # Check test script exists
    if not os.path.exists(args.test_script):
        print(f"ERROR: Test script not found: {args.test_script}")
        sys.exit(1)
    
    ll_input = args.input
    
    # ========================================================================
    # Step #0: Extract and clean pass sequence
    # ========================================================================
    print("="*80)
    print("STEP #0: Extracting Pass Sequence")
    print("="*80)
    print()
    
    print("Extracting full pass sequence via test script -debug-pass=Arguments...")
    passes = extract_pass_sequence(args.test_script, ll_input)
    
    if not passes:
        print("ERROR: Could not extract pass sequence.")
        print("Make sure test script outputs pass list on first line when -debug-pass=Arguments is used")
        sys.exit(1)
    
    print(f"✓ Extracted {len(passes)} passes")
    
    # Clean and deduplicate
    original_count = len(passes)
    passes = process_pass_sequence(passes, deduplicate=True)
    removed = original_count - len(passes)
    print(f"✓ Removed {removed} passes (analysis/duplicates)")
    print(f"✓ {len(passes)} execution passes remain:\n")
    
    for i, p in enumerate(passes):
        print(f"  {i:3d}: {p}")
    
    print()
    
    # ========================================================================
    # Step #1: Find earliest trigger pass
    # ========================================================================
    reduced_passes, results, first_correct_pass = find_earliest_trigger_pass(
        args.test_script, ll_input, passes
    )
    
    if reduced_passes is None:
        print("\nERROR: Analysis failed")
        sys.exit(1)
    
    print()
    
    # ========================================================================
    # Step #2: Generate intermediate IR file
    # ========================================================================
    input2 = ll_input
    passes_for_step3 = reduced_passes
    
    if first_correct_pass:
        print("="*80)
        print("STEP #2: Generating Intermediate IR")
        print("="*80)
        print()
        
        input2 = args.output_mir
        print(f"Dumping IR before pass: {first_correct_pass}")
        print(f"Output file: {input2}\n")
        
        rc, stdout, stderr = run_test_script(args.test_script, ll_input, 
                                            output_file=input2,
                                            stop_before=first_correct_pass)
        
        if rc == 0:
            print(f"✓ Successfully generated {input2}\n")
            
            # Re-extract pass sequence for input2
            print("="*80)
            print("STEP #2b: Re-extracting Pass Sequence for input2")
            print("="*80)
            print()
            
            print("Extracting pass sequence for input2...")
            passes_input2 = extract_pass_sequence(args.test_script, input2)
            
            if passes_input2:
                print(f"✓ Extracted {len(passes_input2)} passes")
                
                original_count = len(passes_input2)
                passes_input2 = process_pass_sequence(passes_input2, deduplicate=True)
                removed = original_count - len(passes_input2)
                print(f"✓ Removed {removed} passes (analysis/duplicates)")
                print(f"✓ {len(passes_input2)} execution passes remain:\n")
                
                for i, p in enumerate(passes_input2):
                    print(f"  {i:3d}: {p}")
                
                print()
                passes_for_step3 = passes_input2
            else:
                print("WARNING: Could not extract pass sequence for input2\n")
        else:
            print(f"WARNING: Failed to generate {input2}\n")
            input2 = ll_input
    
    # ========================================================================
    # Step #3: Backward Suffix Expansion (Find Minimum Prefix)
    # ========================================================================
    
    if first_correct_pass:
        expanded_passes = phase_backward_suffix_expansion(
            args.test_script, input2, passes_for_step3, first_correct_pass
        )
        
        if expanded_passes:
            passes_for_step3 = expanded_passes
        else:
            print("WARNING: Backward suffix expansion failed, continuing with original passes\n")
    
    # ========================================================================
    # Step #4: DDMIN Reduction
    # ========================================================================
    
    final_passes = phase_ddmin_reduction(
        args.test_script, input2, passes_for_step3
    )
    
    print()
    
    # ========================================================================
    # Step #5: Diagnostic output
    # ========================================================================
    print("="*80)
    print("STEP #5: DIAGNOSTIC COMMANDS")
    print("="*80)
    print()
    
    print("To reproduce the crash with final reduced pass list:")
    cmd = f"./interestingtest.sh {input2} /tmp/output.s --run-pass={','.join(final_passes)}"
    print(cmd)
    print()
    
    # ========================================================================
    # Step #6: Write output file if requested
    # ========================================================================
    if args.output:
        with open(args.output, 'w') as f:
            f.write("LLVM Backend Crash Point Analysis\n")
            f.write(f"\n{'='*80}\nSummary\n{'='*80}\n\n")
            f.write(f"Input: {ll_input}\n")
            f.write(f"Test script: {args.test_script}\n")
            f.write(f"Total passes: {len(passes)}\n")
            f.write(f"Reduced passes: {len(final_passes)}\n")
            
            f.write(f"\n{'='*80}\nPass Sequence\n{'='*80}\n\n")
            for i, p in enumerate(passes):
                f.write(f"  {i:3d}: {p}\n")
            
            f.write(f"\n{'='*80}\nTest Results\n{'='*80}\n\n")
            for r in results:
                f.write(f"{r}\n")
            
            f.write(f"\n{'='*80}\nFinal Reduced Pass List\n{'='*80}\n\n")
            for i, p in enumerate(final_passes):
                f.write(f"  {i:3d}: {p}\n")
            
            f.write(f"\n{'='*80}\nDiagnostic Commands\n{'='*80}\n\n")
            f.write(f"Crash reproduction:\n{cmd}\n")
        
        print(f"✓ Wrote diagnostic output to: {args.output}")
        print()
    
    print("="*80)
    sys.exit(0)


if __name__ == "__main__":
    main()
