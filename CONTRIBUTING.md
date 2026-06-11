# Contributing to apairo

Thanks for considering a contribution. This document explains how the
project is organised, the design rules that keep it coherent, and the
mechanics of getting a change merged.

---

## The one rule that decides where code goes

> **Mechanisms live in the core. Collections live in satellites.**
> Repositories are split by **dependency profile and execution context**,
> not by concept.

| Repository | Role | Dependencies |
|---|---|---|
| [`apairo`](https://github.com/apairo-robotics/apairo) | Core: dataset model, views, loaders, preprocess/persistence machinery, contracts | numpy, PyYAML only |
| [`apairo_transform`](https://github.com/apairo-robotics/apairo_transform) | Collection of runtime ops (transforms, augmentations, interpolators) | numpy only |
| [`apairo_preprocess`](https://github.com/apairo-robotics/apairo_preprocess) | Collection of heavy offline preprocessors (CSF, odometry, segmentation models) | scipy, CSF, optional deep models |
| [`apairo_extractor`](https://github.com/apairo-robotics/apairo_extractor) | Rosbag → apairo layouts, CLI/TUI | rosbags |
| [`apairo_rr`](https://github.com/apairo-robotics/apairo_rr) | Visualization bridge | rerun |

Practical consequences:

- A new *way of doing something* (a view, a matching strategy contract, a
  clock resolution) belongs in `apairo` — if it is generic, closed-ended,
  and numpy-only.
- A new *thing to do* (a filter, an augmentation, an interpolator, a
  traversability label) belongs in a satellite — collections are
  open-ended and opinionated.
- The core never imports a satellite. Satellites import the core.
- Anything that pulls a dependency beyond numpy/PyYAML does **not** go in
  the core. If a core mechanism needs one optionally, use an extra
  (`apairo[...]`), and ask first in an issue.

When in doubt: would a training container need it at `__getitem__` time?
If yes and it is generic → core or `apairo_transform`. If it runs once on
a data server → `apairo_preprocess`.

## Design invariants

These are the properties every change must preserve:

1. **Everything is a dataset.** Operations (`filter`, `select`, `cache`,
   `join`, `concat`, `synchronize`) return full `AbstractDataset`
   objects that chain. A feature that breaks chaining is wrongly shaped.
2. **Views are lazy.** A view computes *indices* at construction and
   reads data only on access. No hidden I/O, no hidden materialisation —
   `cache()` is the single, explicit exception.
3. **Samples are honest.** `Sample.data` only contains what is real:
   `timestamp` is `None` iff the data is synchronous; `is_synchronous`
   must tell the truth through every view; a synthesized value (e.g.
   interpolation) must be distinguishable via the view's metadata
   (`frame_indices`, `time_offsets`).
4. **`.apairo` is the source of truth on disk.** Derived channels are
   registered there, never hard-coded. Expensive results are persisted
   and reloaded transparently; cheap results are recomputed.
5. **numpy in, numpy out.** The core knows no training framework.
6. **Extension points are small callables or single-method contracts.**
   `transform(fn)`, `filter(fn)`, `method=callable`,
   `Interpolator.__call__`, `FramePreprocessor.process`. If your
   extension needs more surface than that, propose the contract in an
   issue before implementing.
7. **Additive evolution.** Public behaviour changes require a
   deprecation path. The strongest review signal we have: *existing
   tests should pass unchanged*. If you had to edit an existing test,
   explain why in the PR.

Two known, deliberate deviations (do not "fix" them without discussion):
`SynchronizedView` reads parent loaders directly and bypasses parent
transforms (they are written for single-event samples); asynchronous
datasets expose an implicit `timestamps: dict` + `loaders: dict`
protocol that `synchronize()` relies on.

## Adding things

**A dataset** — prefer a YAML profile + a 2-line `ProfiledDataset`
subclass (see `docs/datasets/adding-a-dataset.md`). If your layout needs
code, keep discovery in `__init__` and loading in `_load`, and implement
`__len__`/`_load` against the contracts in `AbstractDataset`.

**A transform / an interpolator** — goes to `apairo_transform`.

**A preprocessor** — goes to `apairo_preprocess` (or your own project:
preprocessors are designed to live anywhere).

**A core mechanism** — open an issue first describing the contract; the
API surface is deliberately small and grows slowly.

## Mechanics

```bash
git clone https://github.com/apairo-robotics/apairo && cd apairo
make env && source .venv/bin/activate
make install        # pip install -e ".[dev]"
make test           # pytest — must stay green
```

- Tests live in `test/`, mirroring the package layout. Synthetic data is
  built per-test in `tmp_path`; real-data smoke fixtures live in
  `test/assets/` (never load them in place — copy to `tmp_path` first,
  datasets write a `.apairo` sidecar on first load).
- Every behaviour fix ships with a regression test that fails without it.
- Docstrings are Google style and are rendered by mkdocs — new public API
  must appear in `docs/api/reference.md`.
- Keep PRs single-purpose. Deleting dead code is a contribution.

## Versioning

Pre-1.0: minor versions may break APIs, but only with a changelog entry
and a deprecation alias when feasible (see `KeysEmptyWarning` →
`KeysEmptyError` for the pattern). Don't pin git dependencies in
satellites — depend on released versions.
