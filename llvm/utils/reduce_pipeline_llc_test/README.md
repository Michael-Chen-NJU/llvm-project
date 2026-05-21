# reduce_pipeline_llc — LLVM Backend Pass Pipeline Reducer

Automatically reduces an llc backend pass pipeline to the **minimal set of passes** needed to reproduce a crash or correctness bug.

## Quick Start

```bash
# Crash mode — .ll input, full pipeline crashes
python llvm/utils/reduce_pipeline_llc.py \
    --llc-binary=./build/bin/llc \
    --input=test.ll \
    -mtriple=x86_64 -mattr=+nf

# Crash mode — .mir input, known crashing pass sequence
python llvm/utils/reduce_pipeline_llc.py \
    --llc-binary=./build/bin/llc \
    --input=test.mir \
    --run-pass=livevars,phi-node-elimination,register-coalescer

# Test-script mode — correctness bug (miscompile, missing optimization)
python llvm/utils/reduce_pipeline_llc.py \
    --llc-binary=./build/bin/llc \
    --input=test.ll \
    --test-script=./check_bug.sh
```

## Modes of Operation

### 1. Crash Mode (default)

**Input**: `.ll` file + compilation flags that cause llc to crash.

**Algorithm**:
1. Extract pass pipeline via `-debug-pass=Arguments`
2. Binary search with `--stop-before=X` to find the triggering pass
3. Freeze MIR at the trigger point
4. DDMIN on the pass list from that point onward
5. Single-pass sweep for final minimality

**Output**: Trigger pass name + frozen MIR + reproduction command.

### 2. Run-Pass Mode

**Input**: `.mir` file + `--run-pass=A,B,C,...` that crashes.

**Algorithm**: Skip extraction/binary-search, jump directly to DDMIN on the given pass list.

### 3. Test-Script Mode (`--test-script`)

**Input**: `.ll` file + user-provided oracle script.

The script is called with environment variables:
- `$LLC_BINARY` — path to llc
- `$INPUT_FILE` — path to input file
- `$PASSES` — comma-separated pass list (current candidate)
- `$EXTRA_ARGS` — space-separated extra llc arguments
- `$OUTPUT_DIR` — scratch directory

Exit code 0 = bug reproduces (interesting), non-zero = not interesting.

**Algorithm**:
1. Try `opt-bisect-limit` binary search first (fast, for optional pass bugs)
2. Fall back to `--stop-before` + DDMIN if `opt-bisect-limit` is inconclusive

## Options

| Flag | Description |
|------|-------------|
| `--llc-binary PATH` | Path to llc (default: `llc` in PATH) |
| `--input FILE` | Input `.ll` or `.mir` file |
| `--output PATH` | Save reduced MIR to this path |
| `--run-pass PASSES` | Known crashing pass sequence (comma-separated) |
| `--error-pattern STR` | String to match in stderr for target crash |
| `--no-verify` | Don't add `-verify-machineinstrs` during DDMIN |
| `--test-script PATH` | Interestingness test script |

Extra arguments after the known flags are forwarded to llc.

## Technical Constraints

### Constraint 1: .ll vs .mir

- `.ll` files can run the full pipeline (`-O2`, etc.) or use `--stop-before=X`
- `.mir` files can **only** use `--run-pass=X,Y,Z` — cannot run full pipeline
- `--stop-before` only works for machine passes (legacy PM limitation)

### Constraint 2: Mandatory Pass Chain

Machine passes have strict property dependencies:
```
ISel → finalize-isel → ... → regalloc → prologepilog
```

Deleting any mandatory pass causes `MachineFunctionProperties not met` crashes. DDMIN can only remove **optional** passes (~45 optional vs ~85 mandatory in a typical X86 -O2 pipeline).

### Constraint 3: opt-bisect-limit Scope

`-opt-bisect-limit` controls only optional pass execution. Mandatory passes always run regardless. If the bug is in a mandatory pass, `opt-bisect-limit` cannot isolate it.

### Constraint 4: Pre-ISel Crashes

Crashes in passes before instruction selection (e.g., `lower-amx-type`) cannot be further reduced because `--stop-before` doesn't apply to IR function passes in the legacy PM. The tool detects this case and reports it.

## Testing Methodology

### For the tool itself

The validation harness in `reduce_pipeline_llc_test/` runs the tool against the comp-bench dataset:

```bash
cd llvm/utils/reduce_pipeline_llc_test

# Full experiment (requires worktree_stable/ with built llc)
python run_experiment.py --max-instances=50

# Re-test specific instances
echo "llvm-pr-138500" > retest_ids.txt
python retest_subset.py
```

### Worktree Setup

The experiment requires pre-built llc binaries:

```bash
# Stable worktree (latest main)
git worktree add llvm/utils/reduce_pipeline_llc_test/worktree_stable main
cd llvm/utils/reduce_pipeline_llc_test/worktree_stable
cmake -B build -G Ninja -DLLVM_TARGETS_TO_BUILD=X86 \
    -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;lld"
ninja -C build llc FileCheck

# Fallback worktree (for per-commit checkouts)
git worktree add llvm/utils/reduce_pipeline_llc_test/worktree_fallback HEAD
# (same cmake config)
```

### Ground Truth / Oracle

**Ideal oracle**: Given a `.ll` file that triggers a bug when compiled with full pipeline, the tool should produce the same trigger pass that a developer would identify manually.

**Verified results**:

| Instance | Input | Tool Result | Ground Truth File | Match |
|----------|-------|-------------|-------------------|-------|
| llvm-pr-138500 | .ll + `-mattr=+nf` | `x86-suppress-apx-for-relocation` | X86SuppressAPXForReloc.cpp | Exact |
| llvm-pr-123540 | .ll + `-mattr=+amx-fp8` | `lower-amx-type` (pre-ISel) | X86LowerAMXType.cpp | Exact |

**Known limitation**: The dataset's `already_reduced` cases (`.mir` files with `--run-pass=X`) cannot serve as direct full-pipeline oracles because MIR files cannot run the full pipeline.

### Expanding the Test Set

To find more test cases beyond the comp-bench dataset:
- Search LLVM's git history for commits that fix backend crashes with `.ll` reproducers
- Search GitHub Issues for crash reports with IR attachments
- Any `.ll` file + flags that crashes a specific llc build works as a test case

## File Layout

```
llvm/utils/
├── reduce_pipeline_llc.py              # Main tool (git tracked)
└── reduce_pipeline_llc_test/
    ├── README.md                       # This file
    ├── run_experiment.py               # Batch validation harness
    ├── retest_subset.py                # Subset re-test script
    ├── worktree_stable/                # (gitignored) stable llc build
    ├── worktree_fallback/              # (gitignored) per-commit fallback builds
    ├── test_files/                     # (gitignored) extracted .ll/.mir files
    ├── results.jsonl                   # (gitignored) experiment results
    ├── results_retest.jsonl            # (gitignored) retest results
    └── checkpoint.json                 # (gitignored) resume state
```
