# YAML Profiles

A YAML profile fully describes the structure of a synchronous dataset — which directories to traverse, how files map to keys, and how to cast raw bytes into tensors. Profiles live in `apairo/dataset/profiles/`.

A profile has two top-level sections: `layers` and `modalities`.

---

## `layers`

`layers` is a list that describes the directory hierarchy from the dataset root down to individual files. Each element is either a plain string (layer type with no value) or a single-key mapping (layer type with a value).

```yaml
layers:
  - fixed: sequences   # always "sequences/" at this depth
  - sequence           # one directory per recording session
  - modality:          # one directory per channel
      lidar: velodyne  # key "lidar" maps to folder "velodyne"
      labels: labels
```

### Layer types

#### `fixed`

A literal folder name that is always present at this depth in the tree.

```yaml
- fixed: sequences       # string value
- fixed: Rellis-3D
```

#### `split`

A partition directory. When `split=` is passed to the dataset constructor, only files whose path contains this directory name are returned. Omitting `split=` loads all partitions.

```yaml
- split: [train, val, test]
```

!!! note
    GOOSE has `split` at two levels (before and after the modality directory) because both `lidar/train/` and `labels/train/` exist independently.

#### `sequence`

Marks the depth at which individual recording sessions live. The system uses this position to determine where to look for `.apairo` sidecar files and how to compute derived file paths.

```yaml
- sequence
```

#### `modality`

Marks the depth at which per-channel subdirectories live. Has two forms:

**Plain string** — the folder name equals the key name:

```yaml
- modality
# key "lidar" → folder "lidar/"
# key "labels" → folder "labels/"
```

**Mapping** — explicit key → folder renames:

```yaml
- modality:
    lidar: velodyne         # key "lidar" → folder "velodyne/"
    labels: labels          # key "labels" → folder "labels/"
```

---

## `modalities`

`modalities` is a mapping from key name to field spec. Each entry describes one loadable channel.

```yaml
modalities:
  lidar:
    ext: .bin
    dtype: float32
    reshape: [-1, 4]
  labels:
    ext: .label
    dtype: int32
    mask: 65535
    torch_dtype: int64
```

### Fields

| Field | Required | Description |
|---|---|---|
| `ext` | yes | File extension, with or without leading dot (`.bin`, `bin`, `.label`, `.npy`, `.png`, `.jpg`, `.pt`) |
| `dtype` | yes | NumPy dtype string used by `np.fromfile` for binary formats (`.bin`, `.label`). Ignored for structured formats. |
| `reshape` | no | List of ints passed to `ndarray.reshape()` after loading. Use `[-1, 4]` for `(N, 4)` point clouds. |
| `mask` | no | Integer bitmask applied as `arr & mask` before tensor conversion. Use `65535` (`0xFFFF`) to strip SemanticKITTI instance bits. |
| `torch_dtype` | no | Torch dtype name for a final `.to()` cast — e.g., `int64` to convert `int32` labels to `torch.int64`. |
| `loader` | no | Override the loader selected from the file extension. One of `bin`, `npy`, `img`, `pt`. |
| `subpath` | no | Extra path components below the modality directory. Use when there is an additional sub-folder between the channel directory and the files. |
| `optional` | no | If `true`, the key is silently skipped when absent from disk instead of raising `FileNotFoundError`. Default: `false`. |

### Loader selection

For binary formats (`.bin`, `.label`), apairo uses `np.fromfile` directly — `dtype`, `reshape`, `mask`, and `torch_dtype` all apply.

For structured formats, the appropriate loader is selected by extension:

| Extension | Loader | Notes |
|---|---|---|
| `.npy` | `npy` | `np.load` → `torch.from_numpy` |
| `.png`, `.jpg` | `img` | `torchvision.io.read_image` → `torch.Tensor` (CHW, uint8) |
| `.pt` | `pt` | `torch.load(weights_only=True)` |

`reshape`, `mask`, and `torch_dtype` are applied after loading for structured formats as well.

---

## Annotated examples

=== "SemanticKITTI"

    ```yaml
    layers:
      - fixed: sequences    # root/sequences/ always present
      - sequence            # root/sequences/00/, root/sequences/01/, ...
      - modality:
          lidar: velodyne   # "lidar" key → velodyne/ subdirectory
          labels: labels    # "labels" key → labels/ subdirectory

    modalities:
      lidar:
        ext: .bin
        dtype: float32
        reshape: [-1, 4]      # (N*4,) → (N, 4): x, y, z, intensity
      labels:
        ext: .label
        dtype: int32
        mask: 65535           # 0xFFFF: strip upper 16 bits (instance ID)
        torch_dtype: int64    # labels must be int64 for nn.CrossEntropyLoss
    ```

=== "GOOSE"

    ```yaml
    layers:
      - split: [train, val, test]   # root/lidar/train/
      - modality                    # root/lidar/train/seq/ — folder = key name
      - split: [train, val, test]   # root/labels/train/
      - sequence                    # root/labels/train/seq_001/

    modalities:
      lidar:
        ext: .bin
        dtype: float32
        reshape: [-1, 4]
      labels:
        ext: .label
        dtype: int32
        torch_dtype: int64
    ```

    !!! info "Why two `split` layers?"
        GOOSE separates lidar and label data under independent trees: `lidar/train/seq/` and `labels/train/seq/`. The `split` layer appears at depth 0 (before the modality directory) and at depth 2 (after the modality directory), mirroring both branches.

=== "Rellis-3D"

    ```yaml
    layers:
      - fixed: Rellis-3D            # mandatory top-level folder
      - sequence                    # Rellis-3D/00000/, Rellis-3D/00001/, ...
      - modality:
          lidar: os1_cloud_node_kitti_bin
          labels: os1_cloud_node_kitti_label

    modalities:
      lidar:
        ext: .bin
        dtype: float32
        reshape: [-1, 4]
      labels:
        ext: .label
        dtype: int32
        torch_dtype: int64
    ```
