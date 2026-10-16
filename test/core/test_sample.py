import numpy as np
import pytest

from apairo.core.sample import Sample


def test_temporal_sample():
    s = Sample(data={"image_left": np.zeros((3, 224, 224))}, timestamp=1234.5)
    assert s.timestamp == 1234.5
    assert "image_left" in s.data
    assert isinstance(s.data["image_left"], np.ndarray)


def test_synchronous_sample():
    s = Sample(data={"lidar": np.zeros((1000, 4)), "label": np.zeros(1000)})
    assert s.timestamp is None
    assert "lidar" in s.data
    assert "label" in s.data


def test_sample_data_is_dict():
    s = Sample(data={"x": np.array(1.0)})
    assert isinstance(s.data, dict)


def test_sample_is_dataclass():
    from dataclasses import fields

    names = {f.name for f in fields(Sample)}
    assert names == {"data", "timestamp"}


def test_subscript_reads_a_channel():
    lidar = np.zeros((10, 4))
    s = Sample(data={"lidar": lidar}, timestamp=1.0)
    assert s["lidar"] is lidar
    assert "lidar" in s
    assert "labels" not in s


def test_subscript_timestamp_is_a_channel_name():
    # Channel names are free: a channel called "timestamp" wins.
    s = Sample(data={"timestamp": np.array([5.0])}, timestamp=1.0)
    assert s["timestamp"][0] == 5.0


def test_missing_channel_names_the_channels():
    s = Sample(data={"lidar": np.zeros(1)})
    with pytest.raises(KeyError, match=r"no channel 'labels'.*\['lidar'\]"):
        s["labels"]


def test_missing_timestamp_points_to_the_attribute():
    s = Sample(data={"imu": np.zeros(6)}, timestamp=1.0)
    with pytest.raises(KeyError, match=r"sample\.timestamp"):
        s["timestamp"]


def test_sample_is_not_iterable():
    # Iterating would otherwise fall back to s[0] and raise a KeyError.
    with pytest.raises(TypeError, match="not iterable"):
        iter(Sample(data={"x": np.zeros(1)}))
