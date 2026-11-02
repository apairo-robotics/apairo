"""The ``video`` format and ``decode: video`` in ``hdf5``: frames of an encoded
video with random access, read from a file or from bytes inside a container."""

import io
import pickle

import numpy as np
import pytest
import yaml

import apairo
from apairo.loader.formats import Hdf5Format, VideoFormat
from apairo.loader.video_loader import VideoLoader, VideoReader
from apairo.testing import check_format

av = pytest.importorskip("av")

N, FPS, GOP = 24, 10, 5


def _encode(target, n=N, fps=FPS, gop=GOP) -> None:
    """An H.264 video whose frame i is a distinct pattern, a keyframe every gop."""
    with av.open(target, "w", format="mp4") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.codec_context.gop_size = gop
        stream.options = {"crf": "10", "bf": "2"}
        for i in range(n):
            img = np.zeros((48, 64, 3), np.uint8)
            img[:, : 2 + 2 * i] = (200, 40 + 8 * i, 90)
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(img, format="rgb24")
            ):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def _video_bytes(n=N) -> bytes:
    buf = io.BytesIO()
    _encode(buf, n)
    return buf.getvalue()


def _sequential(path_or_bytes):
    src = (
        io.BytesIO(path_or_bytes)
        if isinstance(path_or_bytes, bytes)
        else str(path_or_bytes)
    )
    return [f.to_ndarray(format="rgb24") for f in av.open(src).decode(video=0)]


@pytest.fixture
def video_dir(tmp_path):
    d = tmp_path / "camera"
    d.mkdir()
    _encode(str(d / "camera.mp4"))
    return d


def test_follows_the_format_contract(video_dir):
    check_format(VideoFormat(), video_dir, {"key": {"pts": True}})


def test_random_access_equals_a_sequential_decode(video_dir):
    reference = _sequential(video_dir / "camera.mp4")
    reader = VideoReader(video_dir / "camera.mp4")
    assert len(reader) == N == len(reference)
    # Backwards, across keyframes, then forwards one by one.
    for i in [17, 3, 23, 0, 11, 12, 13, -1]:
        np.testing.assert_array_equal(reader[i], reference[i])
    assert reader[0].shape == (48, 64, 3) and reader[0].dtype == np.uint8
    with pytest.raises(IndexError):
        reader[N]


def test_pts_clock_is_in_seconds(video_dir):
    loader = VideoFormat().open(video_dir, {})
    clock = VideoFormat().clock(loader, {"pts": True}, "camera")
    np.testing.assert_allclose(np.diff(clock), 1 / FPS, atol=1e-6)
    with pytest.raises(ValueError, match="pts: true"):
        VideoFormat().clock(loader, {"pts": "yes"}, "camera")


def test_a_video_directory_is_detected_and_synchronizes(tmp_path):
    seq = tmp_path / "seq"
    (seq / "camera").mkdir(parents=True)
    _encode(str(seq / "camera" / "camera.mp4"))
    (seq / "imu").mkdir()
    np.save(seq / "imu" / "imu.npy", np.arange(100.0).reshape(50, 2))
    np.savetxt(seq / "imu" / "timestamps.txt", np.arange(50) * 0.05)
    (seq / "apairo.yaml").write_text(
        yaml.safe_dump({"version": 1, "channels": {"camera": {"key": {"pts": True}}}})
    )
    ds = apairo.RawDataset(seq, keys=["camera", "imu"])
    assert ds.loaders["camera"].__class__ is VideoLoader
    frames = ds.synchronize(reference="camera", method="nearest", tolerance=0.03)
    assert len(frames) == N
    assert frames[5].data["camera"].shape == (48, 64, 3)
    np.testing.assert_array_equal(frames[5].data["imu"], [20.0, 21.0])  # t = 0.5 s


def test_reader_pickles_for_dataloader_workers(video_dir):
    reader = VideoReader(video_dir / "camera.mp4")
    first = reader[7]
    clone = pickle.loads(pickle.dumps(reader))
    np.testing.assert_array_equal(clone[7], first)


def test_a_video_inside_an_hdf5_container(tmp_path):
    h5py = pytest.importorskip("h5py")
    data = _video_bytes()
    seq = tmp_path / "demo"
    seq.mkdir()
    with h5py.File(seq / "demo.h5", "w") as f:
        f["cam"] = np.void(data)
        f["timestamps/cam"] = 100.0 + np.arange(N) / FPS
    (seq / "apairo.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "channels": {
                    "cam": {
                        "loader": "hdf5",
                        "directory": ".",
                        "array": "/cam",
                        "decode": "video",
                        "key": {"array": "/timestamps/cam", "units": ["s"]},
                    }
                },
            }
        )
    )
    ds = apairo.RawDataset(seq, keys=["cam"])
    reference = _sequential(data)
    assert len(ds.timestamps["cam"]) == N
    np.testing.assert_array_equal(ds.loaders["cam"][9], reference[9])
    clone = pickle.loads(pickle.dumps(ds.loaders["cam"]))
    np.testing.assert_array_equal(clone[3], reference[3])


def test_decode_needs_one_encoded_value(tmp_path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "rec.h5", "w") as f:
        f["rows"] = np.zeros((5, 3))
    with pytest.raises(ValueError, match="decode: video reads one value"):
        Hdf5Format().open(tmp_path, {"array": "/rows", "decode": "video"})
    assert Hdf5Format().validate("x", {"decode": "audio"}, tmp_path)
