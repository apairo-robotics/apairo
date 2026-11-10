# Inference: deploy the training chain

A robot running a model re-runs most of the chain the model was trained on:
the same synchronization, the same filtering, the same voxelization. Writing
that chain a second time, by hand, in the robot's code is how training and
deployment drift apart.

apairo reads the chain back from the dataset you trained on. `inference()`
keeps what the model's outputs need and removes what only belongs in
training. `on()` runs the result on another source, such as another recording
or a stream. `save()` writes it beside the model checkpoint.

```python
from apairo import RawDataset
from apairo_transform import RangeFilter, RandomRotation, StackByPose, VoxelDownsample

train = RawDataset(root, keys=["lidar", "pose", "trav_gt"])
train = train.synchronize(reference="lidar")
train = train.window(size=3, reduce=StackByPose())
train = train.transform("lidar", RangeFilter(max=40.0))
train = train.transform("lidar", RandomRotation(axis="z"))          # augmentation
train = train.transform("lidar", VoxelDownsample(0.1), output="vox")
train = train.transform(TravModel(ckpt))                          # writes trav_pred

val = train.filter_sequences(val_ids).inference(["trav_pred", "trav_gt"])
print(val.summary)

deployed = train.inference(["trav_pred"])
deployed.save("trav_chain.json")                                    # beside the checkpoint
```

---

## What `inference()` keeps

`inference(outputs)` returns an `InferenceView`, a dataset like any other.
Its samples hold the requested outputs only. Three kinds of step are left
out:

| Left out | How it is recognised |
|---|---|
| Steps that do not lead to the outputs | From what each step reads and writes |
| Training-only steps (augmentations) | Their class sets `inference = False` |
| `cache()` and `repeat()` | They change speed or epoch length, not values |

Frame filters (`filter`, `filter_sequences`, `filter_split`) are kept, so a
validation split stays a validation split. The view is a snapshot: steps
added to `train` afterwards do not reach it, and `train` itself is not
modified.

### Pruning by dependency

A per-channel step, `transform(key, fn, output=...)`, reads `key` and writes
`output` (or `key`). A preprocessor declares `input_keys` and its outputs.
Working back from the outputs, every step whose result nobody needs is
removed.

A sample-level function, `transform(fn)`, receives the whole sample, so
apairo cannot tell what it reads. It is kept, flagged as opaque, and every
step before it is kept too. To let apairo prune around it, declare its
channels:

```python
class TravModel:
    input_keys = ["vox"]
    output_keys = ["trav_pred"]

    def __call__(self, sample):
        sample.data["trav_pred"] = self.net(sample.data["vox"])
        return sample
```

A window reducer declares its channels the same way: `StackByPose` does.

### Training-only steps

An augmentation declares it on its class:

```python
class RandomDropout:
    inference = False
    ...
```

The augmentations of `apairo_transform` do (`RandomRotation`, `RandomFlip`,
`RandomScale`, `RandomTranslation`, `GaussianNoise`, `RandomPointDrop`,
`ShufflePoints`). Inside a `Compose`, only those parts are removed:
`Compose([RangeFilter(), RandomRotation()])` keeps its filter.

---

## The summary

```text
>>> print(val.summary)
inference(outputs=['trav_pred', 'trav_gt'])
  kept    window: RangeFilter('lidar')
  removed window: RandomRotation('lidar')  -- training only
  kept    window: VoxelDownsample('lidar' -> 'vox')
  kept    window: TravModel
  ⚠       filter_sequences: 120 frames  -- kept on this data, dropped by .on() on another source
```

`summary.kept`, `summary.removed` and `summary.flags` list the entries. The
flags are the points to check before deploying:

| Flag | Meaning |
|---|---|
| non-causal | A `synchronize` strategy needs a later event: `"next"`, `"nearest"`, an interpolator. Online, it adds latency. |
| opaque | A sample-level step declares no channels: everything before it is kept. |
| keeps state | The step depends on read order (`AccumulateFrames`): it behaves differently on a stream. |
| reference resolved | `synchronize(reference=None)` picked the lowest-rate channel. The name is fixed, so another source uses the same clock. |
| joins a second source | A `join()` brings a dataset that must also be available online. |
| built directly | A view was built with its class rather than a dataset method: what is upstream of it is not inspected. |

`inference(outputs, causal=True)` refuses a non-causal chain instead of
flagging it.

---

## Another source — `on()`

`deployed.on(source)` calls every operation again on `source`: `synchronize`
with the same reference and strategies, `window` with the same reducer, and
the steps registered on the original source are registered on the new one.
Frame filters, caches and repeats are dropped, since they selected or
repeated frames of the original data.

```python
other = RawDataset(other_root, keys=["lidar", "pose"])
predictions = deployed.on(other)
deployed.chain.channels_needed()   # ['lidar', 'pose']: what a source must provide
```

The source must provide those channels under the same names. A chain with
several sources (`join` with a second dataset, `concat`) cannot be replayed
as a whole: run each branch on its own source.

Replaying a recording through the deployed chain and comparing with the
training-time outputs is the parity test for a deployment: the same frames,
read from disk or from a stream, must give the same outputs.

---

## Saving the chain — `save()` and `load_chain()`

```python
deployed.save("trav_chain.json")

from apairo import load_chain
chain = load_chain("trav_chain.json", trusted=["my_robot"])
predictions = chain.on(source)
```

The file is a JSON declaration, not a pickle: for each step, the import path
of its class and the values that rebuild it. Loading imports those classes
and sets those values. It never runs code stored in the file, and it only
imports from apairo, its satellites, and the packages listed in `trusted=`:
each name must be a class or a function defined in one of them, and each
operation one of the dataset methods a chain records.

A chain file still decides which trusted classes are built, and with which
arguments. Load a chain as you would load a configuration file: from a
source you know.

A step must be declarable:

- functions defined at module level (not lambdas or closures), and
  `functools.partial` of them;
- instances of classes defined at module level, rebuilt from their
  attributes (numbers, strings, lists, dicts, numpy arrays, other
  declarable objects);
- an object holding something else, such as model weights, defines
  `get_config()`, and is rebuilt as `cls(**get_config())`:

```python
class TravModel:
    def __init__(self, ckpt):
        self.ckpt = ckpt
        self.net = load(ckpt)

    def get_config(self):
        return {"ckpt": self.ckpt}
```

Anything else makes `save()` refuse and name it. `load_chain` warns when the
packages that wrote the file had other versions: replay a recording to check
the outputs did not change.

---

## On a bag, on the robot

- **A recording**: `BagDataset` (in `apairo_ros`) reads a rosbag in
  place, converted and stamped as an extraction would. A chain trained on
  extracted data replays on the bag it was extracted from with the same
  outputs: `deployed.on(BagDataset("run_42", topics=[...]))`.
- **A live robot**: `apairo_ros.live` feeds a saved chain one message at a
  time (`LiveSource`) and wires it to the subscriptions of a ROS 2 node
  (`ChainNode`). It runs causal chains only. It is on the `feat/live` branch
  of `apairo_ros`, released once tested in a running ROS 2 system.

## Limits

- `on()` needs a source with the same channel names.
- A `synchronize` on an explicit clock array cannot be replayed on another
  source.
- `synchronize` reads its source's raw channels: steps registered on the
  source before it never applied (a warning says so when the view is
  built), and still do not.
