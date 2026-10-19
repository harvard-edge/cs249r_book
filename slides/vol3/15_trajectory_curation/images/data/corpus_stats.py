"""Measure the public trajectory corpus that Lecture 15 shows as real data.

Corpus: nebius/SWE-agent-trajectories on Hugging Face, revision
68195a1450865274106246d0d0296a1d6807b88e (2024-12-23), CC BY 4.0:
80,036 SWE-agent runs on GitHub issues, each with its instance id, model,
pass/fail target, exit status, message list (role, mask, text), and patch.
Download the twelve parquet shards from
https://huggingface.co/datasets/nebius/SWE-agent-trajectories/tree/<revision>/data
into one folder, then run (needs pyarrow, pandas, datasketch):

    python3 corpus_stats.py <folder with train-*.parquet>

Writes corpus_stats.json beside this file. calcs.py prints its "print"
block, and make_figs.py draws the measured plots from the rest. Every
number is counted from the corpus; nothing here is typed by hand.
"""

import glob
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from datasketch import MinHash, MinHashLSH

HERE = Path(__file__).resolve().parent
REFUSAL = "Your proposed edit has introduced new syntax error"  # SWE-agent's edit linter
SEED = 15


def repo_of(instance_id):
    """'owner__repo-1234' -> 'owner__repo' (the issue number is the suffix)."""
    return instance_id.rsplit("-", 1)[0]


def patch_shingles(patch, n=5):
    """Token 5-grams over the changed lines of a unified diff."""
    toks = []
    for line in (patch or "").splitlines():
        if (line.startswith("+") or line.startswith("-")) and not line.startswith(("+++", "---")):
            toks += re.findall(r"\w+|[^\w\s]", line[1:])
    return {" ".join(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}


def main(folder):
    files = sorted(glob.glob(str(Path(folder) / "train-*.parquet")))
    assert len(files) == 12, f"expected 12 shards, found {len(files)}"
    rows = []          # (instance, target, model, exit_status)
    chars = Counter()  # (resolved?, role) -> characters
    mask_by_role = Counter()
    refused_runs = []  # per run: number of refused edits
    run_chars = []     # per run: (all chars, ai chars)
    patches = {}       # run index -> shingle set, resolved runs only
    excerpt = None
    for f in files:
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=2000, columns=[
                "instance_id", "model_name", "target", "exit_status", "trajectory", "generated_patch", "eval_logs"]):
            d = b.to_pydict()
            for i in range(len(d["instance_id"])):
                idx = len(rows)
                iid, tgt = d["instance_id"][i], bool(d["target"][i])
                rows.append((iid, tgt, d["model_name"][i], d["exit_status"][i]))
                n_ref, tot, ai = 0, 0, 0
                traj = d["trajectory"][i]
                for m in traj:
                    L = len(m["text"] or "")
                    chars[(tgt, m["role"])] += L
                    mask_by_role[(m["role"], bool(m["mask"]))] += 1
                    tot += L
                    if m["role"] == "ai":
                        ai += L
                    if m["role"] == "user" and REFUSAL in (m["text"] or ""):
                        n_ref += 1
                refused_runs.append(n_ref)
                run_chars.append((tot, ai))
                if tgt:
                    patches[idx] = patch_shingles(d["generated_patch"][i])
                # A short resolved run with exactly one refused edit, for the slide excerpt
                if excerpt is None and tgt and n_ref == 1 and len(traj) <= 16:
                    excerpt = dict(instance_id=iid, model=d["model_name"][i], run_index=idx,
                                   messages=[dict(role=m["role"], mask=bool(m["mask"]),
                                                  text=(m["text"] or "")[:1500]) for m in traj],
                                   patch=d["generated_patch"][i],
                                   eval_tail=(d["eval_logs"][i] or "")[-300:])
    n = len(rows)
    inst = defaultdict(list)
    for k, (iid, tgt, _, _) in enumerate(rows):
        inst[iid].append(k)
    counts = sorted(len(v) for v in inst.values())
    n_inst = len(inst)

    # Random 80/20 split over traces, and the formula's prediction per task
    rng = random.Random(SEED)
    side = [rng.random() < 0.8 for _ in range(n)]
    straddle = sum(1 for v in inst.values() if len({side[k] for k in v}) == 2)
    predicted = sum(1 - 0.8 ** len(v) - 0.2 ** len(v) for v in inst.values())
    # The same random split over task groups: no task straddles by construction;
    # count test tasks whose repository also has a training task.
    rng = random.Random(SEED)
    task_side = {iid: rng.random() < 0.8 for iid in sorted(inst)}
    train_repos = {repo_of(i) for i, s in task_side.items() if s}
    test_tasks = [i for i, s in task_side.items() if not s]
    test_repo_shared = sum(1 for i in test_tasks if repo_of(i) in train_repos)
    hist = Counter(min(c, 40) for c in counts)

    # Outcome mix per task
    mix = Counter()
    rates = []
    for v in inst.values():
        r = sum(rows[k][1] for k in v)
        rates.append(r / len(v))
        mix["all_fail" if r == 0 else "all_pass" if r == len(v) else "mixed"] += 1

    # Recovery records: resolved runs with at least one refused (linted) edit
    res = [k for k in range(n) if rows[k][1]]
    rec = [k for k in res if refused_runs[k] > 0]
    dire = [k for k in res if refused_runs[k] == 0]
    sum_all = lambda ks: sum(run_chars[k][0] for k in ks)
    sum_ai = lambda ks: sum(run_chars[k][1] for k in ks)
    mean_chars = lambda ks: sum_all(ks) / len(ks)

    # A per-issue cap that keeps the 30 shortest resolved runs (a published recipe's rule)
    by_task = defaultdict(list)
    for k in res:
        by_task[rows[k][0]].append(k)
    kept = [k for ks in by_task.values() for k in sorted(ks, key=lambda k: run_chars[k][0])[:30]]
    kept_rec = sum(1 for k in kept if refused_runs[k] > 0)

    # Near-duplicate resolved patches: MinHash LSH at Jaccard 0.8
    lsh = MinHashLSH(threshold=0.8, num_perm=128)
    sigs = {}
    for k, sh in patches.items():
        if len(sh) < 3:
            continue
        mh = MinHash(num_perm=128, seed=SEED)
        for s in sh:
            mh.update(s.encode())
        sigs[k] = mh
        lsh.insert(str(k), mh)
    same_task_dup, cross_task_pairs, cross_repo_pairs = set(), set(), set()
    for k, mh in sigs.items():
        for c in lsh.query(mh):
            j = int(c)
            if j == k:
                continue
            exact = len(patches[k] & patches[j]) / len(patches[k] | patches[j])
            if exact < 0.8:
                continue
            if rows[j][0] == rows[k][0]:
                same_task_dup.add(k)
            else:
                pair = tuple(sorted((k, j)))
                cross_task_pairs.add(pair)
                if repo_of(rows[j][0]) != repo_of(rows[k][0]):
                    cross_repo_pairs.add(pair)
    cross_tasks = {rows[k][0] for p in cross_task_pairs for k in p}

    s = dict(
        runs=n, tasks=n_inst, repos=len({repo_of(i) for i in inst}),
        resolved=len(res), models=dict(Counter(r[2] for r in rows).most_common()),
        exit_status=dict(Counter(r[3] for r in rows).most_common()),
        runs_per_task=dict(min=counts[0], median=counts[len(counts) // 2], mean=n / n_inst,
                           max=counts[-1], single=sum(1 for c in counts if c == 1)),
        runs_per_task_hist={str(k): v for k, v in sorted(hist.items())},
        straddle=dict(measured=straddle / n_inst, predicted=predicted / n_inst, tasks=straddle),
        group_split=dict(test_tasks=len(test_tasks), test_repo_shared=test_repo_shared),
        outcome_mix=dict(mix), task_pass_rates=sorted(rates),
        mask_by_role={f"{r}|{m}": v for (r, m), v in sorted(mask_by_role.items())},
        chars={f"{'resolved' if t else 'unresolved'}|{r}": v for (t, r), v in sorted(chars.items())},
        recovery=dict(direct=len(dire), recovery=len(rec),
                      direct_chars=sum_all(dire), recovery_chars=sum_all(rec),
                      direct_ai=sum_ai(dire), recovery_ai=sum_ai(rec),
                      runs_with_refusal=sum(1 for x in refused_runs if x > 0),
                      cap30_kept=len(kept), cap30_recovery=kept_rec, tasks_resolved=len(by_task)),
        dup=dict(resolved_with_signature=len(sigs), same_task_dup=len(same_task_dup),
                 cross_task_pairs=len(cross_task_pairs), cross_repo_pairs=len(cross_repo_pairs),
                 cross_tasks=len(cross_tasks)),
    )
    rc = s["recovery"]
    ai_all = sum(v for (t, r), v in chars.items() if r == "ai")
    tot_all = sum(chars.values())
    s["print"] = {
        "runs": f"{n:,}",
        "tasks (issues)": f"{n_inst:,}",
        "repositories": f"{s['repos']:,}",
        "resolved runs": f"{len(res):,} ({len(res) / n:.1%})",
        "runs per task: median / mean / max": f"{s['runs_per_task']['median']} / {n / n_inst:.1f} / {counts[-1]}",
        "tasks with one run": f"{s['runs_per_task']['single']:,}",
        "random 80/20 trace split: tasks on both sides (measured)": f"{straddle / n_inst:.1%} ({straddle:,})",
        "same split: formula prediction per task": f"{predicted / n_inst:.1%}",
        "task-group split: test tasks whose repo is in training": f"{test_repo_shared / len(test_tasks):.1%} of {len(test_tasks):,}",
        "tasks all-fail / mixed / all-pass": f"{mix['all_fail']:,} / {mix['mixed']:,} / {mix['all_pass']:,}",
        "share of tasks mixed": f"{mix['mixed'] / n_inst:.1%}",
        "share of tasks all-fail": f"{mix['all_fail'] / n_inst:.1%}",
        "ai messages with mask True": f"{mask_by_role[('ai', True)]:,} of {mask_by_role[('ai', True)] + mask_by_role[('ai', False)]:,}",
        "non-ai messages with mask True": f"{mask_by_role[('user', True)] + mask_by_role[('system', True)]:,}",
        "model-written share of characters (all runs)": f"{ai_all / tot_all:.1%}",
        "runs with a refused edit": f"{rc['runs_with_refusal']:,} ({rc['runs_with_refusal'] / n:.1%})",
        "resolved runs: direct / with refused edit": f"{len(dire):,} / {len(rec):,}",
        "recovery share of resolved records": f"{len(rec) / len(res):.1%}",
        "recovery share of resolved characters": f"{rc['recovery_chars'] / (rc['recovery_chars'] + rc['direct_chars']):.1%}",
        "recovery share of model-written characters": f"{rc['recovery_ai'] / (rc['recovery_ai'] + rc['direct_ai']):.1%}",
        "recovery chars per model char": f"{rc['recovery_chars'] / rc['recovery_ai']:.2f}",
        "mean characters per resolved run: direct / with refused edit": f"{mean_chars(dire):,.0f} / {mean_chars(rec):,.0f}",
        "shortest-30 cap: resolved runs kept": f"{len(kept):,} of {len(res):,}",
        "shortest-30 cap: recovery share of kept": f"{kept_rec / len(kept):.1%}",
        "issues with a resolved run": f"{len(by_task):,}",
        "direct chars per model char": f"{rc['direct_chars'] / rc['direct_ai']:.2f}",
        "resolved patches with a same-task near-duplicate (J>=0.8)": f"{len(same_task_dup):,} of {len(sigs):,} ({len(same_task_dup) / len(sigs):.1%})",
        "near-duplicate pairs across different tasks": f"{len(cross_task_pairs):,} ({len(cross_repo_pairs):,} across repos)",
    }
    (HERE / "corpus_stats.json").write_text(json.dumps(s, indent=1))
    if excerpt:
        (HERE / "corpus_excerpt.json").write_text(json.dumps(excerpt, indent=1))
    for k, v in s["print"].items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main(sys.argv[1])
