# reduce_pipeline_llc — LLVM 后端 Pass Pipeline 自动化 Reducer

自动将 llc 后端 pass pipeline 缩减为**能复现 crash 或正确性 bug 的最小 pass 集合**。

## 快速上手

```bash
# Crash 模式 — .ll 输入，全量 pipeline 会 crash
python llvm/utils/reduce_pipeline_llc.py \
    --llc-binary=./build/bin/llc \
    --input=test.ll \
    -mtriple=x86_64 -mattr=+nf

# Crash 模式 — .mir 输入，已知哪些 pass 会 crash
python llvm/utils/reduce_pipeline_llc.py \
    --llc-binary=./build/bin/llc \
    --input=test.mir \
    --run-pass=livevars,phi-node-elimination,register-coalescer

# Test-script 模式 — 正确性 bug（误编译、缺失优化等）
python llvm/utils/reduce_pipeline_llc.py \
    --llc-binary=./build/bin/llc \
    --input=test.ll \
    --test-script=./check_bug.sh
```

## 工作模式

### 模式 1：Crash 模式（默认）

**输入**：一个 `.ll` 文件 + 编译 flags，使得 llc 编译时 crash。

**算法流程**：
1. 通过 `-debug-pass=Arguments` 提取完整 pass pipeline
2. 用 `--stop-before=X` 做二分查找，定位触发 crash 的 pass
3. 在触发点冻结 MIR（`--stop-before` 导出中间状态）
4. 对触发点之后的 pass 列表执行 DDMIN 最小化
5. 逐个 pass 扫描，确保最终结果不可再删

**输出**：触发 pass 名称 + 冻结的 MIR 文件 + 复现命令。

### 模式 2：Run-Pass 模式

**输入**：`.mir` 文件 + `--run-pass=A,B,C,...`（已知这组 pass 会 crash）。

**算法**：跳过 pipeline 提取和二分查找，直接对给定 pass 列表执行 DDMIN。

### 模式 3：Test-Script 模式（`--test-script`）

**输入**：`.ll` 文件 + 用户自定义的判定脚本（oracle）。

脚本通过环境变量接收参数：
- `$LLC_BINARY` — llc 路径
- `$INPUT_FILE` — 输入文件路径
- `$PASSES` — 当前候选的 pass 列表（逗号分隔）
- `$EXTRA_ARGS` — 其他 llc 参数（空格分隔）
- `$OUTPUT_DIR` — 可用的临时目录

退出码 0 = bug 可复现（interesting），非 0 = bug 消失（not interesting）。

**算法**：
1. 先尝试 `opt-bisect-limit` 二分查找（快速，适用于可选 pass 中的 bug）
2. 若 `opt-bisect-limit` 无法定位（mandatory pass bug），回退到 `--stop-before` + DDMIN

**示例：检测误编译的 test script**：
```bash
#!/bin/bash
# 检查编译结果中是否包含错误指令
$LLC_BINARY --run-pass=$PASSES -o $OUTPUT_DIR/out.mir $INPUT_FILE $EXTRA_ARGS || exit 1
$LLC_BINARY -o $OUTPUT_DIR/out.s $OUTPUT_DIR/out.mir $EXTRA_ARGS || exit 1
grep -q "wrong_instruction" $OUTPUT_DIR/out.s
```

## 命令行参数

| 参数 | 说明 |
|------|------|
| `--llc-binary PATH` | llc 路径（默认：PATH 中的 `llc`） |
| `--input FILE` | 输入 `.ll` 或 `.mir` 文件 |
| `--output PATH` | 保存缩减后的 MIR 到此路径 |
| `--run-pass PASSES` | 已知会 crash 的 pass 序列（逗号分隔） |
| `--error-pattern STR` | 匹配 stderr 中的字符串以识别目标 crash |
| `--no-verify` | DDMIN 过程中不添加 `-verify-machineinstrs` |
| `--test-script PATH` | 用户自定义判定脚本 |

所有未被识别的参数会直接转发给 llc（如 `-mtriple`, `-mattr` 等）。

## 技术约束

### 约束 1：.ll 与 .mir 的区别

| | .ll 文件 | .mir 文件 |
|---|---|---|
| 全量 pipeline | 支持（`-O2` 等） | **不支持**（会因 Properties 不满足而假 crash） |
| `--stop-before=X` | 支持（机器 pass） | 不适用 |
| `--run-pass=X,Y,Z` | 支持 | 支持 |

核心结论：**只有 .ll 文件才能作为全量 pipeline reduction 的输入**。

### 约束 2：Mandatory Pass 链

机器 pass 之间存在严格的属性依赖：
```
ISel → finalize-isel → ... → regalloc → prologepilog
```

删除任何一个 mandatory pass 都会导致 `MachineFunctionProperties not met` 的假 crash。DDMIN 只能删除**可选 pass**（典型 X86 -O2 pipeline 约 45 个可选 vs 85 个 mandatory）。

### 约束 3：opt-bisect-limit 的作用范围

`-opt-bisect-limit` 仅控制可选 pass 的执行。Mandatory pass 无论 limit 设为多少都会执行。如果 bug 出在 mandatory pass 中，`opt-bisect-limit` 无法隔离。

### 约束 4：ISel 之前的 crash

发生在指令选择（ISel）之前的 crash（如 `lower-amx-type`）无法进一步 reduce，因为 `--stop-before` 不适用于 legacy PM 中的 IR function pass。工具会检测到这种情况并报告。

### 约束 5：Correctness Bug 的 Oracle

- Crash bug：`exit code != 0` 即可判定
- Correctness bug：需要比较编译输出（如 FileCheck），但 `--run-pass` 输出的是 MIR 而非汇编
- 解决方案：在 test-script 中自行完成 `compile → check` 完整流程

## 测试方案

### 如何验证工具本身的正确性

`reduce_pipeline_llc_test/` 目录下有批量验证框架，可对 comp-bench 数据集跑实验：

```bash
cd llvm/utils/reduce_pipeline_llc_test

# 跑完整实验（需要先构建 worktree_stable/ 中的 llc）
python run_experiment.py --max-instances=50

# 重测特定实例
echo "llvm-pr-138500" > retest_ids.txt
python retest_subset.py
```

### 环境准备（Worktree）

实验需要预编译的 llc：

```bash
# Stable worktree（最新 main 分支）
git worktree add llvm/utils/reduce_pipeline_llc_test/worktree_stable main
cd llvm/utils/reduce_pipeline_llc_test/worktree_stable
cmake -B build -G Ninja -DLLVM_TARGETS_TO_BUILD=X86 \
    -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;lld"
ninja -C build llc FileCheck

# Fallback worktree（用于按 commit 切换重编）
git worktree add llvm/utils/reduce_pipeline_llc_test/worktree_fallback HEAD
# （同样的 cmake 配置）
```

**开销估算**：
- 首次编译 llc：~10 分钟
- 单个实例 reduce：5-30 秒（crash 模式），30-120 秒（test-script 模式）
- 全量 249 实例：约 2-3 小时（含 fallback 重编）

### Ground Truth（测试预言）

**理想场景**：给工具一个 `.ll` + full pipeline，它应该输出与开发者手写的 `--run-pass=X` 相同的结果。

**已验证的端到端结果**：

| 实例 | 输入 | 工具输出 | 正确答案（buggy file） | 匹配 |
|------|------|----------|----------------------|------|
| llvm-pr-138500 | .ll + `-mattr=+nf` | `x86-suppress-apx-for-relocation` | X86SuppressAPXForReloc.cpp | 完全正确 |
| llvm-pr-123540 | .ll + `-mattr=+amx-fp8` | `lower-amx-type` (pre-ISel) | X86LowerAMXType.cpp | 完全正确 |

**已知限制**：数据集中 `already_reduced` 的 case 全是 MIR 文件（有 `--run-pass=X`），不能直接作为 full-pipeline oracle——因为 MIR 文件不能跑全量 pipeline。

### 扩展测试集

不局限于 comp-bench 的 249 个实例：
- 从 LLVM git history 中找修复后端 crash 的 commit（带 `.ll` reproducer）
- 从 GitHub Issues 中找带 IR 附件的 crash report
- 任何 `.ll` + flags 能让特定 llc 版本 crash 的文件都可作为测试用例

## 文件结构

```
llvm/utils/
├── reduce_pipeline_llc.py              # 主工具（git tracked）
└── reduce_pipeline_llc_test/
    ├── README.md                       # 英文文档
    ├── README_zh.md                    # 本文件（中文文档）
    ├── run_experiment.py               # 批量验证框架
    ├── retest_subset.py                # 子集重测脚本
    ├── worktree_stable/                # (gitignored) 稳定版 llc 编译
    ├── worktree_fallback/              # (gitignored) 按 commit 切换的编译
    ├── test_files/                     # (gitignored) 提取的 .ll/.mir 文件
    ├── results.jsonl                   # (gitignored) 实验结果
    ├── results_retest.jsonl            # (gitignored) 重测结果
    └── checkpoint.json                 # (gitignored) 断点续跑状态
```

## 典型工作流

```
用户有一个 crash bug 的 .ll 文件
        │
        ▼
  reduce_pipeline_llc.py --input=crash.ll --llc-binary=./build/bin/llc
        │
        ├── Phase 0: 验证 crash 可复现，提取 pipeline
        ├── Phase 1: 二分查找 trigger pass
        ├── Phase 2: 在 trigger 点冻结 MIR
        ├── Phase 3: DDMIN 最小化 pass 集合
        └── Phase 4: 逐个扫描，确认最小性
        │
        ▼
  输出: "Trigger pass: x86-suppress-apx-for-relocation"
        + frozen.mir + 复现命令
```
