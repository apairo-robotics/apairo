from __future__ import annotations
from abc import abstractmethod
from pathlib import Path
from typing import Optional

from apairo.core.config import (
    register_channel as _register_channel,
    register_raw_channel as _register_raw_channel,
    verify_config as _verify_config,
    read_config,
    write_config,
    config_exists,
)
from apairo.core.preprocessor import Preprocessor


class ConfigurableDataset:
    """Mixin for datasets that support preprocessed-channel extensibility via ``.apairo``.

    Any dataset class that wants to be extensible at runtime (i.e. allow users to
    register new preprocessed channels without touching source code) should inherit
    from this mixin alongside its normal base class.

    Concrete subclasses must implement :meth:`_bootstrap_config`, which describes
    how to auto-discover the dataset's raw channels when ``.apairo`` does not yet
    exist.

    Usage pattern for preprocessing scripts::

        MyDataset.register_channel(
            seq_dir, "my_channel", "npys",
            timestamps_from="lidar",
            sources=["lidar"],
        )

    Usage in dataset ``__init__``::

        config = self._load_or_create_config(sequence_dir)
    """

    @classmethod
    def register_channel(
        cls,
        sequence_dir: str | Path,
        key: str,
        loader: str,
        *,
        timestamps_from: Optional[str] = None,
        sources: Optional[list[str]] = None,
    ) -> None:
        """Register a preprocessed channel in ``sequence_dir/.apairo``.

        Args:
            sequence_dir: Dataset sequence directory.
            key: Channel name -- must match its subdirectory name.
            loader: Data format: ``"npy"``, ``"npys"``, ``"bin"``, or ``"img"``.
            timestamps_from: Channel whose timestamps to share when this channel
                has no ``timestamps.txt`` of its own.
            sources: Provenance -- channels this channel was derived from.
        """
        _register_channel(
            sequence_dir,
            key,
            loader,
            timestamps_from=timestamps_from,
            sources=sources,
        )

    @abstractmethod
    def _bootstrap_config(self, sequence_dir: Path) -> dict:
        """Return an initial ``.apairo`` config for this dataset.

        Called when no ``.apairo`` exists yet.  Should auto-discover all raw
        channels present in ``sequence_dir`` and return a config dict of the form::

            {
                "version": 1,
                "channels": {
                    "channel_name": {"loader": "npys", "has_timestamps": True},
                    ...
                },
            }
        """
        ...

    @classmethod
    def run_preprocess(
        cls,
        preprocessor: Preprocessor,
        root_dir: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        """Run a preprocessor on a dataset and persist the output channel.

        Delegates to :func:`apairo.preprocess.run`, which handles file placement
        via ``derived_path()``, format-specific saving, timestamp writing, and
        ``.apairo`` registration at ``root_dir``.

        Args:
            preprocessor: A :class:`~apairo.core.preprocessor.FramePreprocessor`
                or :class:`~apairo.core.preprocessor.SequencePreprocessor`.
            root_dir: Dataset root directory.
            overwrite: Recompute if output already exists.

        Example::

            MyDataset.run_preprocess(MyPreprocessor(), "/data/my_dataset")
        """
        from apairo.preprocess.runner import run

        run(preprocessor, cls, root_dir, overwrite=overwrite)

    @classmethod
    def describe(cls, sequence_dir: str | Path) -> dict:
        """Describe what is available in a sequence directory.

        Reads ``.apairo`` (creating it if absent) and cross-references it with
        the class's :attr:`available_keys` to produce a three-way breakdown:

        - **raw / present** -- raw channels on disk and registered
        - **raw / missing** -- raw channels known from the profile but not on disk
        - **preprocess** -- channels produced by a preprocessing pipeline

        Returns the breakdown as a dict and prints a human-readable summary.

        Example::

            MyDataset.describe("/data/my_dataset/sequence_01")
        """
        sequence_dir = Path(sequence_dir)
        config = (
            read_config(sequence_dir)
            if config_exists(sequence_dir)
            else cls(sequence_dir)._load_or_create_config(sequence_dir)
        )
        channels = config.get("channels", {})

        raw_present = sorted(
            k for k, v in channels.items() if v.get("kind", "raw") == "raw"
        )
        preprocess = {
            k: v for k, v in channels.items() if v.get("kind") == "preprocess"
        }
        raw_missing = sorted(
            k for k in getattr(cls, "available_keys", frozenset()) if k not in channels
        )

        # --- pretty print ---
        print(f"\n{cls.__name__} -- {sequence_dir.name}")
        print("─" * 50)

        print("Raw channels")
        if raw_present:
            print("  present  :", ", ".join(raw_present))
        if raw_missing:
            print("  missing  :", ", ".join(raw_missing))
        if not raw_present and not raw_missing:
            print("  (none)")

        print("Preprocessed channels")
        if preprocess:
            for key, meta in sorted(preprocess.items()):
                ts_info = (
                    f"<- timestamps from {meta['timestamps_from']}"
                    if "timestamps_from" in meta
                    else "<- own timestamps"
                )
                src_info = (
                    f"  sources: {meta['sources']}" if meta.get("sources") else ""
                )
                print(f"  {key:<20} {meta['loader']:<6} {ts_info}{src_info}")
        else:
            print("  (none)")
        print()

        return {
            "raw": {"present": raw_present, "missing": raw_missing},
            "preprocess": preprocess,
        }

    @classmethod
    def register_raw_channel(
        cls,
        sequence_dir: str | Path,
        key: str,
        loader: str,
        *,
        has_timestamps: Optional[bool] = None,
    ) -> None:
        """Declare a raw channel in ``sequence_dir/.apairo``.

        Use this to manually add or override a raw channel declaration, for
        example after :meth:`init` detected the wrong loader type.

        Args:
            sequence_dir: Dataset sequence directory.
            key: Channel name -- must match its subdirectory name.
            loader: Data format: ``"npy"``, ``"npys"``, ``"bin"``, or ``"img"``.
            has_timestamps: Whether the channel has ``timestamps.txt``.
                Auto-detected when ``None``.
        """
        _register_raw_channel(sequence_dir, key, loader, has_timestamps=has_timestamps)

    @classmethod
    def verify(cls, sequence_dir: str | Path) -> bool:
        """Verify that ``.apairo`` is coherent with what is on disk.

        Prints any issues found and returns ``True`` when the config is clean.

        Args:
            sequence_dir: Dataset sequence directory containing ``.apairo``.

        Returns:
            ``True`` if no issues were found, ``False`` otherwise.

        Example::

            ok = MyDataset.verify("/data/my_dataset/seq_01")
        """
        issues = _verify_config(sequence_dir)
        if not issues:
            print(f"OK  {Path(sequence_dir)}/.apairo")
            return True
        print(f"{len(issues)} issue(s) in {Path(sequence_dir)}/.apairo :")
        for issue in issues:
            print(f"  - {issue}")
        return False

    def _load_or_create_config(self, root_dir: Path) -> dict:
        """Read ``.apairo/channels.yaml`` if it exists, otherwise bootstrap and write it."""
        if not config_exists(root_dir):
            config = self._bootstrap_config(root_dir)
            write_config(root_dir, config)
        else:
            config = read_config(root_dir)
        return config
