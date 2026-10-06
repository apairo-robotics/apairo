# Adding a Dataset

A dataset joins apairo in one of two forms, both added with a pull request
and installed for everyone with apairo:

- **A declaration**, an `apairo.yaml`, when `RawDataset` can read the dataset
  as it lies on disk. This is the case for TUM RGB-D and EuRoC MAV. Add it to
  `apairo/dataset/declarations/<name>.yaml`, where it becomes usable as
  `declare="<name>"`. Add a miniature of the layout to `test/assets/` with a
  test that loads it, and write a guide in `docs/datasets/`.
- **A class**, a YAML profile and a short subclass, for a fixed synchronous
  layout such as SemanticKITTI, Rellis-3D or GOOSE. The steps are below.
  Register the class in `apairo/dataset/registry.py` so that
  `apairo init --as <Class>` knows it.

Either way, add an extra in `pyproject.toml` named after the dataset, listing
what reading it needs (Pillow for images). Users then install it with
`pip install apairo[<dataset>]`. And add its sample, below: CI refuses a
standard dataset without one.

A dataset you would rather keep in a package of your own registers its class
through an entry point. It is then selectable by name like a standard one:

```toml
[project.entry-points."apairo.datasets"]
MyLabDataset = "my_lab_dataset:MyLabDataset"
```

The rest of this page builds a class. It requires:

1. A YAML profile describing the directory structure
2. A 2-line Python subclass
3. Tests

---

## Step 1 -- Write the YAML profile

Create `apairo/dataset/profiles/<your_dataset>.yaml`. Study your dataset's folder layout and map it onto the [layer DSL](yaml-profiles.md#layers).

**Example -- a hypothetical PandaSet-style layout:**

```
<root>/
  001/
    lidar/    00.bin  01.bin  ...
    camera/   00.png  01.png  ...
```

Profile:

```yaml
# apairo/dataset/profiles/pandaset.yaml
layers:
  - sequence            # root/001/, root/002/, ...
  - modality            # root/001/lidar/, root/001/camera/

modalities:
  lidar:
    ext: .bin
    dtype: float32
    reshape: [-1, 4]
  camera:
    ext: .png
    loader: img         # explicit: torchvision read_image
    dtype: uint8        # ignored for non-binary, but required field
```

!!! tip "Choosing layer order"
    Write out one example file path relative to the dataset root, then annotate each component:

    ```
    001 / lidar / 000000.bin
     ↑      ↑         ↑
    seq  modality   file
    ```

    Then list the layers in that order: `[sequence, modality]`.

---

## Step 2 -- Create the Python subclass

Create `apairo/dataset/pandaset/dataset.py`:

```python
from apairo.core.profiled_dataset import ProfiledDataset


class PandaSetDataset(ProfiledDataset):
    _profile = "pandaset.yaml"
```

Create `apairo/dataset/pandaset/__init__.py`:

```python
from .dataset import PandaSetDataset

__all__ = ["PandaSetDataset"]
```

Export from `apairo/dataset/__init__.py`:

```python
from .pandaset import PandaSetDataset
```

And add to `apairo/__init__.py` and its `__all__`.

---

## Step 3 -- Write tests

Create `test/dataset/test_pandaset.py`. Use `tmp_path` to build a minimal mock tree:

```python
import numpy as np
import pytest
from apairo.dataset.pandaset import PandaSetDataset

N = 40

@pytest.fixture
def pandaset_root(tmp_path):
    for seq in ["001", "002"]:
        (tmp_path / seq / "lidar").mkdir(parents=True)
        for i in range(3):
            np.random.rand(N, 4).astype(np.float32).tofile(
                tmp_path / seq / "lidar" / f"{i:06d}.bin"
            )
    return tmp_path

def test_len(pandaset_root):
    ds = PandaSetDataset(pandaset_root, keys=["lidar"])
    assert len(ds) == 6   # 2 seqs x 3 frames

def test_getitem_shape(pandaset_root):
    ds = PandaSetDataset(pandaset_root, keys=["lidar"])
    sample = ds[0]
    assert sample.data["lidar"].shape == (N, 4)

def test_available_keys():
    assert PandaSetDataset.available_keys == frozenset({"lidar", "camera"})
```

---

## Step 4 -- Add a sample, and describe it

Every standard dataset has a sample in `test/assets/`: a few frames, in the
dataset's real tree and file names. CI fails if a registered dataset or a
shipped declaration has none. Use a real excerpt whenever the licence
allows it, made by a script that fetches it from the official archive. See
`test/assets/fetch_public_samples.py`, which reads a few members of a zip with
HTTP range requests. A synthetic layout is the fallback; say which real data
it was checked against.

Then describe the sample in `test/assets/samples.yaml`:

```yaml
  mini_pandaset:
    dataset: PandaSetDataset
    root: mini_pandaset
    source: >-
      real excerpt: 6 scans of 2 sequences, strided to 1024 points;
      test/assets/fetch_public_samples.py
    license: CC BY 4.0
    real_env: APAIRO_REAL_PANDASET
    keys: [lidar]
    frames: 6
    clock: true
    expect:
      lidar: {shape: [null, 4], dtype: float32}
```

`test/dataset/test_samples.py` runs `apairo.testing.check_dataset` on it:

- it runs `init` and loading the way a user does;
- it reads the first, middle and last frames of every channel, and checks
  their dtype, rank and shape against `expect`;
- it checks the clock: finite, one timestamp per frame, never going back
  within a sequence;
- it checks that `apairo check` reports nothing;
- it checks the splits;
- it writes a preprocess output into the sample and reads it back.

`expect` is where the dataset's own knowledge goes. A label `range`, for
example, is what caught GOOSE's instance ids hidden in the upper bits of its
labels.

A sample shows the layout, but only the full dataset shows that the layout
holds on every sequence. Point the sample's `real_env` variable at a full copy
and run `pytest -m realdata` before a release. The check works on a stand-in
whose files are links, so it writes nothing next to the data.

A dataset kept in a package of its own runs the same check in its own tests:

```python
from apairo.testing import check_dataset

def test_my_dataset_follows_the_contract():
    check_dataset(MyLabDataset, "tests/sample", keys=["lidar"], clock=True)
```

---

## Derived key support

Derived keys (preprocessed outputs) work automatically for any `ProfiledDataset` subclass. The `derived_path()` method generates the correct output path by replacing the modality component in an existing file path.

```python
PandaSetDataset.run_preprocess(MyPreprocessor(), "/data/pandaset")

ds = PandaSetDataset("/data/pandaset", keys=["lidar", "my_output"])
```

No extra code needed in your subclass.

---

## When to extend SynchronousDataset directly

Use a direct `SynchronousDataset` subclass (instead of `ProfiledDataset`) only when:

- The directory structure is too irregular to describe with the layer DSL
- You need custom loading logic per frame that cannot be expressed with dtype/reshape/mask

In that case, implement `__len__`, `__getitem__`, `__iter__`, `__next__`, and override `derived_path()` and `_seq_root()` as needed. See `apairo/dataset/goose/dataset.py` (the pre-refactor version in git history) as a reference.
