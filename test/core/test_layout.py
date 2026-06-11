"""DatasetLayout: spec resolution, scan, and write -> loader round-trip."""

import numpy as np
import pytest

from apairo import ChannelSpec, DatasetLayout
from apairo.utils.naming import integer_frame_index


@pytest.fixture
def layout():
    return DatasetLayout(
        channels={
            "image": ChannelSpec(
                path=("images.tar",),
                store="tar_jpeg",
                name_to_index=integer_frame_index,
                member_name=lambda i: f"{i:06d}.jpg",
                write_options={"quality": 95},
            ),
            "position": ChannelSpec(
                path=("trajectory.zarr", "positions.zarr"),
                dtype=np.float32,
                chunks=lambda shape: (min(1024, shape[0]), *shape[1:]),
            ),
        },
        compression=("zstd", 5),
        default=ChannelSpec(path=(), dtype=np.float32),
    )


def test_spec_table_and_convention(layout):
    assert layout.spec("position").path == ("trajectory.zarr", "positions.zarr")
    # Unknown key -> flat convention with the default template
    spec = layout.spec("my_new_channel")
    assert spec.path == ("my_new_channel.zarr",)
    assert spec.dtype == np.float32


def test_write_then_load_roundtrip(layout, tmp_path):
    data = np.arange(20, dtype=np.float64).reshape(10, 2)
    layout.write(tmp_path, "position", data)

    loader = layout.loader(tmp_path, "position", 10)
    assert loader is not None
    assert len(loader) == 10
    # dtype imposed by the spec at write time
    np.testing.assert_array_equal(loader[3], data[3].astype(np.float32))


def test_convention_write_and_scan(layout, tmp_path):
    layout.write(tmp_path, "extra_channel", np.zeros((4, 3), np.float32))
    layout.write(tmp_path, "position", np.zeros((4, 2), np.float32))

    assert layout.exists(tmp_path, "extra_channel")
    assert layout.scan(tmp_path) == ["position", "extra_channel"]

    loader = layout.loader(tmp_path, "extra_channel", 4)
    assert loader[0].shape == (3,)


def test_image_roundtrip(layout, tmp_path):
    frames = np.random.default_rng(0).integers(
        0, 255, (3, 8, 8, 3), dtype=np.uint8
    )
    layout.write(tmp_path, "image", frames)

    loader = layout.loader(tmp_path, "image", 3)
    img = loader[2]
    assert img.shape == (8, 8, 3)
    assert loader.member_name(2) == "000002.jpg"


def test_missing_channel_loader_is_none(layout, tmp_path):
    assert layout.loader(tmp_path, "position", 5) is None
    assert not layout.exists(tmp_path, "position")
