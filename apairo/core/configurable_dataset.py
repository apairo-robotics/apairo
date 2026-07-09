from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path

from apairo.core.config import (
    config_exists,
    read_config,
    write_config,
)
from apairo.core.config import (
    register_channel as _register_channel,
)
from apairo.core.config import (
    register_raw_channel as _register_raw_channel,
)
from apairo.core.config import (
    remove_channel as _remove_channel,
)
from apairo.core.config import (
    verify_config as _verify_config,
)

logger = logging.getLogger(__name__)


class _RunPreprocessDescriptor:
    """Descriptor that makes run_preprocess work on both classes and instances.

    - ``Dataset.run_preprocess(preprocessor, root_dir, **kwargs)`` — standard
    - ``ds.run_preprocess(preprocessor, **kwargs)``  — root_dir inferred from instance
    """

    def __get__(self, obj, cls):
        from apairo.preprocess.runner import run

        if obj is not None:
            root_dir = obj.root_dir

            def _instance_run(preprocessor, *, overwrite=False, **dataset_kwargs):
                run(preprocessor, cls, root_dir, overwrite=overwrite, **dataset_kwargs)

            return _instance_run

        def _class_run(preprocessor, root_dir, *, overwrite=False, **dataset_kwargs):
            run(preprocessor, cls, root_dir, overwrite=overwrite, **dataset_kwargs)

        return _class_run


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
        timestamps_from: str | None = None,
        sources: list[str] | None = None,
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

    @classmethod
    def remove_channel(
        cls, sequence_dir: str | Path, key: str, *, data: bool = False
    ) -> dict:
        """Remove a channel from ``sequence_dir/.apairo`` (inverse of
        :meth:`register_channel`).

        By default only the declaration is dropped, which is reversible; pass
        ``data=True`` to also delete the channel's directory on disk. Returns the
        removed metadata entry. For an interactive guard on *raw* channels and
        data deletion, prefer the CLI (``apairo channel remove``).

        Args:
            sequence_dir: Dataset sequence directory.
            key: Channel name to remove (its on-disk directory name).
            data: Also delete the channel's directory from disk (destructive).
        """
        return _remove_channel(sequence_dir, key, data=data)

    @abstractmethod
    def _bootstrap_config(self, sequence_dir: Path) -> dict:
        """Return an initial ``.apairo`` config for this dataset.

        Called when no ``.apairo`` exists yet.  Should auto-discover all raw
        channels present in ``sequence_dir`` and return a config dict of the form::

            {
                "version": 1,
                "channels": {
                    "channel_name": {"loader": "npys", "kind": "raw"},
                    ...
                },
            }
        """
        ...

    run_preprocess = _RunPreprocessDescriptor()
    """Run a preprocessor and persist the output channel.

    Can be called on the class or on an existing instance:

    **Class form** -- root_dir required::

        Goose3DDataset.run_preprocess(preprocessor, "/data/GOOSE_3D", split="train")

    **Instance form** -- root_dir inferred from the dataset::

        ds = Goose3DDataset("/data/GOOSE_3D", keys=["lidar"], split="train")
        ds.run_preprocess(preprocessor)

    Extra keyword arguments are forwarded to the dataset constructor (class form)
    or ignored (instance form, since the instance is already configured).
    ``overwrite=True`` recomputes even if the output already exists.

    To preview a :class:`~apairo.core.preprocessor.FramePreprocessor` without
    writing anything, run the same instance lazily first:
    ``ds.transform(preprocessor)``.
    """

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
            else cls(sequence_dir)._load_or_create_config(sequence_dir)  # type: ignore[call-arg]
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
        alias: str | None = None,
        directory: str | None = None,
        suffix: str | None = None,
    ) -> None:
        """Declare a raw channel in ``sequence_dir/.apairo``.

        Use this to manually add or override a raw channel declaration, for
        example after :meth:`init` detected the wrong loader type.

        Args:
            sequence_dir: Dataset sequence directory.
            key: Channel name -- must match its subdirectory name.
            loader: Data format: ``"npy"``, ``"npys"``, ``"bin"``, ``"img"``, or
                ``"zarr"``.
            alias: Public name to expose the channel under at load time (the
                directory keeps its real name). See
                :func:`~apairo.core.config.set_alias`.
            directory: On-disk subdirectory this channel's files live in, when
                different from *key* (a suffixed sub-channel sharing another
                channel's directory). Defaults to *key*.
            suffix: Frame-file suffix to load from *directory* (e.g.
                ``"intensity"`` for ``000000_intensity.npy``). Pairs with
                *directory*.
        """
        _register_raw_channel(
            sequence_dir, key, loader, alias=alias, directory=directory, suffix=suffix
        )

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
        """Read ``.apairo/channels.yaml`` if it exists, otherwise bootstrap it.

        The bootstrapped config is written back so the next load skips
        detection. Loading must never *require* write access (shared datasets
        often sit on read-only mounts): when the sidecar cannot be written, the
        config is kept on the instance (``_config_fallback``) and picked up by
        :class:`~apairo.dataset.async_layout.AsyncLayoutDataset` in place of
        the on-disk file.
        """
        if not config_exists(root_dir):
            config = self._bootstrap_config(root_dir)
            try:
                write_config(root_dir, config)
            except OSError as exc:
                logger.warning(
                    "Cannot write the .apairo sidecar in '%s' (%s); using the "
                    "bootstrapped config in memory for this instance. Run "
                    "`apairo init` on a writable copy to persist it.",
                    root_dir,
                    exc,
                )
                self._config_fallback = config
        else:
            config = read_config(root_dir)
        return config
