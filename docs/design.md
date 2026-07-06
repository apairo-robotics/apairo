# Design & Philosophy

This page explains *why* apairo is shaped the way it is. If you are
deciding where a new feature belongs, read this together with
[CONTRIBUTING.md](https://github.com/apairo-robotics/apairo/blob/main/CONTRIBUTING.md).

---

## The problem apairo solves

Robotics sensor data arrives as multi-rate streams and is stored in
incompatible per-dataset layouts. Every lab rewrites the same three
layers: file discovery, temporal alignment, and derived-channel caching —
usually as an imperative pipeline whose choices (sync rate, filtering,
label generation) are *baked into the files it writes*, so changing any
parameter means re-running everything.

apairo's answer is to make each of those layers a **lazy, composable
dataset operation** instead of a pipeline stage:

```
raw files ──(profiles / .apairo)──> dataset
dataset   ──synchronize/filter/select/transform──> views   (index math, no I/O)
dataset   ──run_preprocess──> persisted derived channels    (heavy, run once)
```

A choice expressed as a *view* costs nothing to change. A choice
expressed as a *preprocessor* is persisted once and reloaded
transparently. Nothing else writes to disk.

## The data model

- **`Sample`** — a dict of numpy arrays plus an optional timestamp. The
  timestamp follows the frame's *clock*: an asynchronous event carries its own
  timestamp; a synchronous frame resampled onto a reference clock (a
  `synchronize()` result) carries that reference tick; a clockless synchronous
  frame (a profiled dataset) carries `timestamp=None`.
- **Synchronous datasets** (`ProfiledDataset` + YAML profiles) — index =
  frame. Random access, `DataLoader`-ready.
- **Asynchronous datasets** (`AsyncLayoutDataset` layouts; `RawDataset`,
  `TartanKittiDataset`) — index = position in the merged, timestamp-ordered
  event timeline.
- **`synchronize()`** bridges the two: it resamples an asynchronous
  dataset onto a reference clock (a channel, a fixed rate, or distance
  ticks) and returns a view that *is* synchronous — so the entire
  downstream API applies identically to both worlds.

## Views, not copies

`filter`, `select`, `cache`, `join`, `concat`, `synchronize` all return
full `AbstractDataset` objects. All except `cache()` are **lazy**: they
compute index mappings at construction and read data only on
`__getitem__`. This is what makes experimentation cheap — a filtered,
synchronized, channel-projected view of a 100 GB dataset costs a few
arrays of integers.

Sweeps that *are* expensive (predicate filters) expose their result
(`view.indices`) so you can persist and reload it: *sweep once, reload
free* is a recurring pattern (`.apairo` plays the same role for derived
channels).

## Where code lives: mechanisms vs collections

The ecosystem is several repositories, split by **dependency profile and
execution context** — not by concept:

- **`apairo` (core)** — mechanisms: the dataset model, views, loaders,
  persistence machinery, and the *contracts* that extensions implement.
  numpy + PyYAML, nothing else. Closed-ended by design.
- **`apairo_transform`** — a collection of runtime ops (filters,
  augmentations, pose utilities, interpolators). numpy-only, safe to
  install in any training container. Open-ended.
- **`apairo_preprocess`** — a collection of heavy offline preprocessors
  (ground segmentation, odometry, learned models). Runs once on a data
  server; its dependencies never contaminate training environments.
- **`apairo_extractor`** — rosbags in, apairo layouts out.
- **`apairo_rr`** — visualization via rerun.

The contracts crossing these boundaries are deliberately tiny: a
transform is a callable on an array or a `Sample`; a preprocessor is a
callable on a `Sample` plus declarative I/O attributes; a synchronization
strategy is a callable on timestamp arrays; an interpolator is one
`__call__(t, t0, v0, t1, v1)`. Small contracts are what let collections
grow without the core changing.

## Honesty guarantees

Several APIs exist purely so the data never lies to you:

- `is_synchronous` tells the truth through every view and composition.
- `SynchronizedView.frame_indices` and `.time_offsets()` expose exactly
  which event (or bracketing pair) backs every value, and how stale it is.
- `tolerance` drops frames rather than silently serving stale data;
  interpolation requires both bracketing events rather than extrapolating.
- `describe()` and `verify()` report what is actually on disk versus what
  `.apairo` declares.

When extending apairo, preserve this property: it is the difference
between a loader and a source of silent experimental error.
