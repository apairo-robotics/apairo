"""The standard datasets, checked on full copies -- opt in, outside CI.

A sample shows the layout; only the dataset itself shows that the layout
holds on every sequence. Point the variable a card names (``real_env`` in
``test/assets/samples.yaml``) at a full copy and run::

    APAIRO_REAL_GOOSE=/data/goose pytest -m realdata

Each copy is checked with the sample's expectations, the frame count aside,
and without writing a preprocess output (a copy of a channel of a full
dataset). Nothing is written next to the data: the check works on a stand-in
whose files are links. A variable left unset skips its dataset.
"""

from __future__ import annotations

import os

import pytest

from apairo.testing import check_dataset
from test.dataset.test_samples import CARDS, check_args


@pytest.mark.realdata
@pytest.mark.parametrize("name", sorted(CARDS))
def test_the_full_dataset_follows_the_dataset_contract(name):
    card = CARDS[name]
    root = os.environ.get(card["real_env"])
    if not root:
        pytest.skip(f"{card['real_env']} is not set")
    args = {k: v for k, v in check_args(card).items() if k != "frames"}
    args.update(card.get("real", {}))
    check_dataset(card["dataset"], root, round_trip=False, **args)
