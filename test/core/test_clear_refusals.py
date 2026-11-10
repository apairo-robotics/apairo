"""Calls that cannot work are refused where they are made, naming the cause."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from apairo.dataset.stream import StreamDataset


def _stream():
    t = np.arange(4) * 0.1
    values = [np.full(2, float(i)) for i in range(4)]
    return StreamDataset({"a": (t, values), "b": (t + 0.05, list(values))})


def test_synchronize_on_a_filtered_async_view_names_the_cause():
    with pytest.raises(
        ValueError, match="view over an asynchronous dataset.*then filter"
    ):
        _stream().filter([0, 1, 2]).synchronize(reference="a")


def test_synchronize_on_a_synchronous_dataset_says_so():
    with pytest.raises(ValueError, match="already synchronous"):
        _stream().synchronize(reference="a").synchronize(reference="a")


def test_synchronize_on_concatenated_async_datasets_is_refused_at_the_call():
    with pytest.raises(ValueError, match="Synchronize each dataset before combining"):
        _stream().concat(_stream()).synchronize(reference="a")


def test_a_slice_is_refused_with_the_way_to_do_it():
    with pytest.raises(TypeError, match=r"not by a slice.*ds\.filter\(range\(0, 3\)\)"):
        _stream()[0:3]


def test_transforms_registered_before_synchronize_warn():
    ds = _stream().transform("a", lambda x: x * 10, in_place=False)
    with pytest.warns(UserWarning, match="1 transform step.*not applied") as caught:
        view = ds.synchronize(reference="a")
    assert caught[0].filename == __file__  # points at the call, not at apairo
    assert view[0].data["a"][0] == 1.0  # the stored channel (10.0 if transformed)


def test_no_warning_without_prior_transforms_nor_on_replay():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _stream().synchronize(reference="a")
    ds = _stream().transform("a", lambda x: x * 10, in_place=False)
    with pytest.warns(UserWarning):
        view = ds.synchronize(reference="a")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        view.inference("a").on(_stream())


def test_a_cache_without_sequence_ids_follows_the_base_contract():
    cached = _stream().synchronize(reference="a").cache()
    assert getattr(cached, "frame_sequence_ids", None) is None
    with pytest.raises(AttributeError, match="no frame_sequence_ids"):
        _ = cached.frame_sequence_ids
    assert cached.frame_info(0).sequence is None
    with pytest.raises(AttributeError, match="no frame_sequence_ids"):
        cached.filter_sequences(["x"])
