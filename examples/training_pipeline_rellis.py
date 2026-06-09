"""Register and run a custom preprocessing pipeline on a TartanDrive sequence."""
import Path
import os
from queue import LifoQueue

import numpy as np
from scipy.spatial import KDTree

from apairo.preprocess import FramePreprocessor, SequencePreprocessor
from apairo.dataset import Rellis3DDataset
from apairo.core.sample import Sample


ROOT_DIR = os.path.join(Path.home(), "data", "rellis")

# --- Transform Robot Filter ------------------------------------------------

def norm_inf(x):
    return np.max(np.abs(x), axis=1)

class RobotFilter:
    """ Crop the points to close of the lidar.
    Args:
        d : Distance to crop
        norm : Norm used to crop Linf is for a cube, L2 for a sphere
    """
    def __init__(
        self,
        d = 1.0,
        norm  = norm_inf,
    ) -> None:
        if d<0:
            raise ValueError(f"d must be positive")

        if norm(0) > 0:
            raise ValueError(f"Warning norm not definite norm(0) = {norm(0)}")
        self._d = d    
        self._norm = norm

    def compute_mask(self, pc: np.ndarray) -> np.ndarray:
        """Return the boolean keep-mask without applying it.

        Useful when multiple aligned arrays (e.g. point cloud + labels) must
        be filtered with the same mask::

            mask   = RangeFilter(max=50.0).compute_mask(pc)
            pc     = pc[mask]
            labels = labels[mask]
        """
        pc = np.asarray(pc)
        ranges = self._norm(pc[:, :3])
        mask = np.ones(len(pc), dtype=bool)
        mask &= ranges >= self._d
        return mask

    def __call__(self, pc: np.ndarray) -> np.ndarray:
        return np.asarray(pc)[self.compute_mask(pc)]

    def __repr__(self) -> str:
        return f"RobotFilter(d={self._d},, norm={self._norm!r})"



# --- Frame-by-frame preprocessor (one output file per scan) ----------------
_RELLIS_TRAVERSABLE_IDS = {1, 3, 10, 23, 31, 33}

class TraversabilityFromLabels(FramePreprocessor):
    """Label each point traversable based on its semantic class ID.

    Args:
        labels_key:      Input channel for per-point semantic labels.
        traversable_ids: Set of semantic class IDs considered traversable.
                         Defaults to the RELLIS-3D traversable classes.
        output_key:      Override the default output channel name ``"trav_label"``.
    """

    output_key =        "trav_label"
    output_loader =     "npys"
    input_keys =        ["labels"]
    timestamps_from =   "lidar"
    sources =           ["labels"]

    def __init__(
        self,
        labels_key = "labels",
        traversable_ids = None,
        output_key = None,
    ) -> None:
        self._labels_key = labels_key
        self._trav_ids = (
            traversable_ids if traversable_ids is not None else _RELLIS_TRAVERSABLE_IDS
        )
        self.input_keys = [labels_key]
        self.sources = [labels_key]
        if output_key is not None:
            self.output_key = output_key

    def process(self, sample: Sample) -> np.ndarray:
        labels = np.asarray(sample.data[self._labels_key])
        return np.isin(labels, list(self._trav_ids)).astype(np.uint8)



class TraversabilityFromTrajectory(SequencePreprocessor):
    """Label each point traversable if it lies in the robot's forward footprint.

    Poses and point clouds are loaded from the dataset via ``input_keys``.

    Sequence boundary detection and look-ahead are computed in ``process()``
    where the full trajectory is available.

    Args:
        lidar_key:             Input channel for point cloud data.
        poses_key:             Input channel for per-frame poses — must be ``(4, 4)``
                               float64.  Apply ``PoseTo4x4()`` from ``apairo_transform``
                               if needed.
        robot_radius:          Half-width of the robot footprint in XY (metres).
        height_min:            Minimum point height relative to the nearest robot
                               position to be traversable (metres, < 0).
        height_max:            Maximum point height relative to the nearest robot
                               position (metres, ≥ 0).
        forward_window:        Maximum number of future poses to look ahead.
                               ``None`` (default) uses the entire remaining trajectory.
        sequence_gap:          Distance threshold (metres) to detect sequence
                               boundaries and avoid look-ahead across discontinuous
                               sessions.
        output_key:            Override the default output channel name ``"trav_traj"``.
    """

    output_key =        "trav_traj"
    output_loader =     "npys"
    input_keys =        ["lidar", "poses"]
    timestamps_from =   "lidar"
    sources =           ["lidar", "poses"]

    def __init__(
        self,
        lidar_key: str = "lidar",
        poses_key: str = "poses",
        robot_radius: float = 0.75,
        height_min: float = -1.0,
        height_max: float = 0.5,
        forward_window: int | None = None,
        sequence_gap: float = 5.0,
        near_exclusion_radius: float = 0.0,
        near_exclusion_norm: float = np.inf,
        output_key: str | None = None,
    ) -> None:
        self._lidar_key = lidar_key
        self._poses_key = poses_key
        self._robot_radius = robot_radius
        self._height_min = height_min
        self._height_max = height_max
        self._forward_window = forward_window
        self._sequence_gap = sequence_gap


        self.input_keys = [lidar_key, poses_key]
        self.sources = [lidar_key, poses_key]
        if output_key is not None:
            self.output_key = output_key


    def process(self, frames) -> np.ndarray:
        all_samples = list(frames)
        n = len(all_samples)

        poses = np.asarray(
            np.stack([s.data[self._poses_key] for s in all_samples]), dtype=np.float64
        )
        if poses.shape[1:] != (4, 4):
            raise ValueError(
                f"poses must be (N, 4, 4) — apply PoseTo4x4() from apairo_transform first. "
                f"Got {poses.shape}"
            )

        # Sequence boundary detection — done here where poses are available.
        positions = poses[:, :3, 3]
        dists = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        boundaries = np.where(dists > self._sequence_gap)[0]
        ends = np.concatenate([boundaries, [n - 1]])
        seq_end = ends[np.searchsorted(ends, np.arange(n))]

        results = []
        for idx, sample in enumerate(all_samples):
            pc = np.asarray(sample.data[self._lidar_key])
            xyz_sensor = pc[:, :3].astype(np.float64)
            n_pts = len(xyz_sensor)

            T = poses[idx]
            xyz_h = np.column_stack([xyz_sensor, np.ones(n_pts)])
            xyz_world = (T @ xyz_h.T).T[:, :3]

            seq_end_idx = int(seq_end[idx]) + 1
            end = min(
                idx + 1 + self._forward_window
                if self._forward_window is not None
                else seq_end_idx,
                seq_end_idx,
            )
            future_pos = poses[idx + 1 : end, :3, 3]

            if len(future_pos) == 0:
                results.append(np.zeros(n_pts, dtype=np.uint8))
                continue

            tree = KDTree(future_pos[:, :2])
            # workers=-1 deadlocks when called from a ThreadPoolExecutor (e.g. viewer)
            dist_xy, nn_idx = tree.query(xyz_world[:, :2], k=1)
            dz = xyz_world[:, 2] - future_pos[nn_idx, 2]

            trav = (
                (dist_xy < self._robot_radius)
                & (dz >= self._height_min)
                & (dz <= self._height_max)
            )
            if self._near_filter is not None:
                trav &= self._near_filter.compute_mask(xyz_sensor)

            results.append(trav.astype(np.uint8))

        return np.stack(results)


def pipeline():
    robot_filter = RobotFilter(d=1.0)
    rellis_ds_preprocess = Rellis3DDataset(ROOT_DIR)
    rellis_ds_preprocess.transform(robot_filter, "lidar", output="lidar_filtered")
    
    print("Preprocess 1/2 : Traversability From Labels (ground truth)")
    trav_from_labels = TraversabilityFromLabels(
        labels_key="labels",
        traversable_ids=_RELLIS_TRAVERSABLE_IDS,
        output_key="trav_gt"
    )

    rellis_ds_preprocess.run_preprocess(trav_from_labels)

# --- Run -------------------------------------------------------------------

TartanKittiDataset.run_preprocess(TravLabel(), SEQ_DIR)
TartanKittiDataset.run_preprocess(GICPPoses(), SEQ_DIR)

# Load the sequence including preprocessed channels
ds = TartanKittiDataset(SEQ_DIR, keys=["velodyne_0", "trav_label", "gicp_poses"])
print("Keys   :", ds.keys)
print("Length :", len(ds))


