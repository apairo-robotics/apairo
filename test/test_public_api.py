import apairo


def test_public_names():
    expected = [
        "Sample",
        "SynchronousDataset",
        "RawDataset",
        "ConcatDataset",
        "split_sequences",
        "SemanticKittiDataset",
        "Rellis3DDataset",
        "Goose3DDataset",
    ]
    for name in expected:
        assert hasattr(apairo, name), f"apairo.{name} not found"


def test_version():
    assert hasattr(apairo, "__version__")
    assert isinstance(apairo.__version__, str)
