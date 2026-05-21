#!/usr/bin/env python3
"""Re-test a specific subset of instances from the experiment.

Reads instance IDs from retest_ids.txt, processes each one using the same
logic as run_experiment.py, and writes results to results_retest.jsonl.
"""
import json
import sys
import os
import logging
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiment import (
    load_and_filter_dataset, extract_test_file, parse_run_line,
    try_instance, checkout_and_rebuild,
    STABLE_LLC, STABLE_FILECHECK, FALLBACK_LLC, TEST_FILES_DIR, log
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

RETEST_IDS_FILE = Path(__file__).parent / 'retest_ids.txt'
RETEST_RESULTS_FILE = Path(__file__).parent / 'results_retest.jsonl'


def main():
    if not STABLE_LLC.exists():
        log.error(f"Stable LLC not found: {STABLE_LLC}")
        sys.exit(1)

    with open(RETEST_IDS_FILE) as f:
        target_ids = set(line.strip() for line in f if line.strip())
    log.info(f"Re-testing {len(target_ids)} instances")

    instances = load_and_filter_dataset()

    targets = [inst for inst in instances if inst['instance_id'] in target_ids]
    log.info(f"Found {len(targets)} matching instances in dataset")

    if RETEST_RESULTS_FILE.exists():
        RETEST_RESULTS_FILE.unlink()

    for i, instance in enumerate(targets):
        iid = instance['instance_id']
        log.info(f"[{i+1}/{len(targets)}] Processing {iid}")

        test_file = extract_test_file(instance)
        if not test_file or not test_file.exists():
            log.warning(f"  Failed to extract test file")
            result = {'instance_id': iid, 'status': 'extract_failed'}
            with open(RETEST_RESULTS_FILE, 'a') as f:
                f.write(json.dumps(result) + '\n')
            continue

        test_content = test_file.read_text()
        run_configs = parse_run_line(test_content)

        result = try_instance(STABLE_LLC, STABLE_FILECHECK, test_file,
                             run_configs, instance)
        result['build_used'] = 'stable'

        if result.get('status') in ('no_bug_found', 'crash_reduce_failed') and FALLBACK_LLC.exists():
            base_commit = instance.get('base_commit', '')
            if base_commit:
                log.info(f"  Trying fallback build at {base_commit[:12]}...")
                if checkout_and_rebuild(base_commit):
                    result = try_instance(FALLBACK_LLC, STABLE_FILECHECK, test_file,
                                         run_configs, instance)
                    result['build_used'] = 'fallback'
                else:
                    log.warning(f"  Fallback build failed")

        gt_files = instance.get('buggy_files', [])
        result['ground_truth'] = {'buggy_files': gt_files}

        with open(RETEST_RESULTS_FILE, 'a') as f:
            f.write(json.dumps(result) + '\n')

        log.info(f"  -> status={result['status']}, "
                f"method={result.get('reduction_result',{}).get('method')}, "
                f"build={result.get('build_used')}")

    log.info("\n=== SUMMARY ===")
    results = []
    with open(RETEST_RESULTS_FILE) as f:
        for line in f:
            results.append(json.loads(line))

    statuses = Counter(r['status'] for r in results)
    for s, c in statuses.most_common():
        log.info(f"  {s}: {c}")

    builds = Counter(r.get('build_used', '?') for r in results)
    log.info(f"\nBuilds used:")
    for b, c in builds.most_common():
        log.info(f"  {b}: {c}")


if __name__ == '__main__':
    main()
