# `assets/`

Bundled split definitions, so the evaluation protocols are self-contained and do
not depend on a research drive. Most of what lives here is read by code; two
files are not, and each says below why it is kept.

These ship at the repository root rather than inside the package because they
are data a reader is meant to open and diff, and because the paper cites them by
repository path. `sourcetrace.config.ASSETS_DIR`
resolves this directory from the package's own location, which is why the
documented install is editable (`pip install -e .`) -- a wheel would not carry
them.

## Read by code

| Path | Read by | Purpose |
|------|---------|---------|
| `protocols/mlaad/{train,dev,eval}.csv` | `datasets/mlaad.py::build_split` | MLAAD v5 sample manifests (`path,model_name`); 56,891 rows total over 82 systems |
| `protocols/panda_merge_map.json` | `datasets/mlaad.py::_load_merge_map` / `_family` | version-merge map (the architecture-merging rule of Neamtu et al.) — collapses true version duplicates into architecture families (82 systems → **77 families** → K=65 known after the 15% family holdout) |
| `protocols/stopa/attacks.json` | `datasets/stopa.py::build_tables` | STOPA AM × vocoder attack grid (`AA01`…`AA13`) |

## Not read by code, but checked against it

### `splits/heldout_families_seed42.json`

The 12 families `build_split` holds out as OOD at `panda_seed=42` -- the realized
answer, written down. Regenerating it means running the split, so it is committed
and regression-tested against the rule rather than trusted.

### `readme/hero-{light,dark}.{html,png}`

The README banner. The HTML is the source and the PNG is a headless-Chrome render
of it. The exact headless-Chrome command that produces each PNG is in the
header of its own HTML file.

## Upstream attribution

### `protocols/mlaad/meta.txt`

Nicolas Müller's protocol description for the MLAAD manifests redistributed
here. Not read by code. Kept because the manifests are upstream work and the
top-level `README.md` states that this repository respects MLAAD's terms.

## Provenance record — NOT read by code

### `provenance/v5_train_split_replicated.json` (2.6 MB)

A dump from an **earlier run of the original grader** at `panda_seed=42`:
`K=66` plus the train/dev/eval path lists (38,191 capped paths). Nothing in the
training or evaluation path loads it; it is the evidence behind the
merge-map-revision account below.

It disagrees with what `build_split` produces today (`K=65`), and the reason is
worth stating because the obvious explanation is wrong:

* **Not a different manifest.** Every one of the dump's 38,191 paths is present
  in the bundled `protocols/mlaad/*.csv` (zero dump-only paths). It is the
  post-cap, post-OOD-removal subset of the same 56,891-path universe — a strict
  subset, not an equal set. The inputs match.
* **A different merge-map revision.** `K = n_families − max(6, round(0.15 ·
  n_families))`. `K=66` requires **78** families; the bundled
  `protocols/panda_merge_map.json` collapses the same 82 systems into **77** (→ 12 OOD
  families → `K=65`). The dump was produced with a slightly less aggressive map
  that left one extra family unmerged.

The split *logic* is identical. Treat the dump as a provenance record, **not** as
an equality target.
