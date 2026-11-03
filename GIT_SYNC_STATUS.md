# Git sync status: VM vs jobo322/MRAi_P3 (remote `fork`)

Repository: `/home/nmrbox/0070/abolanos/git/mrai_p3`
Branch: `install_in_mac_m1`
Remotes:
- `origin` -> https://github.com/Amir-Jahangiri/MRAi_P3.git
- `fork`   -> https://github.com/jobo322/MRAi_P3.git  (the target remote to match)

## 1. Commits present locally but NOT on `fork`

Two commits ahead of `fork/install_in_mac_m1`:

| Commit | Message | Files changed |
|--------|---------|---------------|
| `75883f2` | chore: poolK should passed in htcondor submition | `htcondor/preextract.sub` (+2/-2) |
| `c0415be` | chore: keep peaks and tap in the batch inference | `htcondor/infer.sub` (+5/-5) |

To bring the other machine (or `fork`) up to date, push these commits:

```bash
git checkout install_in_mac_m1
git push fork install_in_mac_m1
```

Or, on the other machine, fetch and fast-forward:

```bash
git fetch fork
git checkout install_in_mac_m1
git merge --ff-only fork/install_in_mac_m1   # after the VM pushes
```

## 2. Uncommitted (tracked) modifications

These edits exist in the working tree but are NOT committed:

- `.gitignore` — adds `htcondor/runs/` to ignored paths (per-run generated .sub copies)
- `Trained/Probability_2D_Elongated/DNN_0_local_serving/DNN_0` — deleted (tracked file removed)
- `docs/runs/2026-10-26-cupo-eje-band2-4e22c41.md` — +8 lines
- `htcondor/README.md` — +116/-? (large rewrite)

To preserve these on the other machine, either commit them on this VM first:

```bash
git add .gitignore docs/runs/2026-10-26-cupo-eje-band2-4e22c41.md htcondor/README.md
git rm Trained/Probability_2D_Elongated/DNN_0_local_serving/DNN_0
git commit -m "chore: sync uncommitted VM edits"
git push fork install_in_mac_m1
```

or carry them as a patch:

```bash
git diff > vm-uncommitted.patch
# on other machine:
git apply vm-uncommitted.patch
```

## 3. Formerly untracked files (now committed in e39ffcd)

These files were untracked; they are now committed in `e39ffcd` (chore: add VM-only scripts, runbook and skill doc) and must be pushed to `fork` to reach the other machine:

- `docs/runs/2026-10-27-pools-3268-ejecucion-40f85a5.md`
- `docs/skills/mrai-p3-htcondor.md`
- `htcondor/train_gpu.sub`
- `scripts/hard_negatives_fast.py`
- `scripts/train_mixed_precision.py`

Recommended: commit + push them so both machines share history:

```bash
git add docs/runs/2026-10-27-pools-3268-ejecucion-40f85a5.md docs/skills/mrai-p3-htcondor.md htcondor/train_gpu.sub scripts/hard_negatives_fast.py scripts/train_mixed_precision.py
git commit -m "chore: add VM-only scripts, runbook and skill doc"
git push fork install_in_mac_m1
```

## 4. Ignored/generated artifacts (NOT to sync)

Large generated outputs under `Trained/`, `htcondor/logs/`, `htcondor/runs/`, `data/`, `__pycache__/`, etc. are gitignored. These must be regenerated or copied separately (rsync) if needed; do NOT commit them.

## 5. Steps for the other machine to match this VM

1. `git fetch fork && git fetch origin`
2. `git checkout install_in_mac_m1`
3. `git pull fork install_in_mac_m1` (or `git merge --ff-only fork/install_in_mac_m1`) to get commits `c0415be`, `75883f2`.
4. Apply the uncommitted patch (section 2) if you want the working-tree edits.
5. Copy the untracked files (section 3) manually, or pull them after they are committed/pushed.
6. Regenerate ignored artifacts (Trained/, logs/) as needed.

## Notes

- The remote to equalize against is `fork` (jobo322/MRAi_P3), not `origin` (Amir-Jahangiri).
- Current branch `install_in_mac_m1` exists on both remotes; `feature-branch` and `main` are already in sync with `fork`.
- There are 2 local commits ahead of `fork/install_in_mac_m1` and no commits behind.