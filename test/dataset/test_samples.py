"""Every standard dataset, checked on its sample against the dataset contract.

``test/assets/samples.yaml`` describes one sample per standard dataset and
shipped declaration: where it comes from, and what
:func:`apairo.testing.check_dataset` must find in it. A standard dataset, or a
declaration, without a sample fails here -- adding one to apairo means adding
its sample, its source, and what to expect of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from apairo.core.profiled_dataset import ProfiledDataset
from apairo.dataset import registry
from apairo.testing import check_dataset

ASSETS = Path(__file__).parent.parent / "assets"
CARDS: dict[str, dict] = yaml.safe_load((ASSETS / "samples.yaml").read_text())[
    "samples"
]
CHECK_ARGS = ("keys", "declare", "expect", "paired", "clock", "splits", "frames")


def check_args(card: dict) -> dict:
    return {k: card[k] for k in CHECK_ARGS if k in card}


@pytest.mark.parametrize("name", sorted(CARDS))
def test_the_sample_follows_the_dataset_contract(name):
    card = CARDS[name]
    check_dataset(card["dataset"], ASSETS / card["root"], **check_args(card))


@pytest.mark.parametrize("name", sorted(CARDS))
def test_the_sample_says_where_it_comes_from(name):
    card = CARDS[name]
    for field in ("dataset", "root", "source", "license", "real_env"):
        assert card.get(field), f"sample '{name}' has no '{field}'"
    assert (ASSETS / card["root"]).is_dir(), f"sample '{name}': no {card['root']}"
    if card["source"].startswith("synthetic"):
        assert card.get("checked"), (
            f"sample '{name}' is synthetic: say which real data its layout was "
            f"checked against ('checked')"
        )


def test_every_standard_dataset_has_a_sample():
    registry.dataset_names()
    covered = {card["dataset"] for card in CARDS.values()}
    missing = sorted(set(registry._BUILTINS) - covered)
    assert not missing, (
        f"standard dataset(s) without a sample: {missing} -- add one to "
        f"test/assets/ and describe it in test/assets/samples.yaml"
    )


def test_every_shipped_declaration_has_a_sample():
    covered = {card.get("declare") for card in CARDS.values()}
    missing = sorted(set(registry.declaration_names()) - covered)
    assert not missing, (
        f"shipped declaration(s) without a sample: {missing} -- add one to "
        f"test/assets/ and describe it in test/assets/samples.yaml"
    )


def test_the_check_catches_the_goose_label_bug_it_was_written_for(tmp_path):
    """The GOOSE profile read labels without masking the instance id out of
    their upper 16 bits, and its tests drew labels in 0..63: they passed. On
    the real sample, the card's expected range catches it."""
    profile = yaml.safe_load(
        (Path(registry.__file__).parent / "profiles" / "goose.yaml").read_text()
    )
    del profile["modalities"]["labels"]["mask"]
    (tmp_path / "goose_unmasked.yaml").write_text(yaml.safe_dump(profile))

    class UnmaskedGoose(ProfiledDataset):
        _profile = str(tmp_path / "goose_unmasked.yaml")

    card = CARDS["mini_goose"]
    with pytest.raises(AssertionError) as err:
        check_dataset(UnmaskedGoose, ASSETS / card["root"], **check_args(card))
    breaches = str(err.value).splitlines()[1:]
    assert breaches and all(
        "'labels' frame" in b and "outside [0, 63]" in b for b in breaches
    )
