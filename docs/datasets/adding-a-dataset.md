# Adding a Dataset

Adding a new synchronous dataset to apairo requires:

1. A YAML profile describing the directory structure
2. A 2-line Python subclass
3. Tests

---

## Step 1 — Write the YAML profile

Create `apairo/dataset/profiles/<your_dataset>.yaml`. Study your dataset's folder layout and map it onto the [layer DSL](yaml-profiles.md#layers).

**Example — a hypothetical PandaSet-style layout:**

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

## Step 2 — Create the Python subclass

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

## Step 3 — Write tests

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
    assert len(ds) == 6   # 2 seqs × 3 frames

def test_getitem_shape(pandaset_root):
    ds = PandaSetDataset(pandaset_root, keys=["lidar"])
    sample = ds[0]
    assert sample.data["lidar"].shape == (N, 4)

def test_available_keys():
    assert PandaSetDataset.available_keys == frozenset({"lidar", "camera"})
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
