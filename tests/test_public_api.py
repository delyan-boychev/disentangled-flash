import inspect


def test_public_api_imports():
    import re

    import disentangled_flash as df

    assert callable(df.optimize_deberta)
    assert callable(df.enable_deberta_inference)
    assert df.KernelConfig(64, 64, 4).block_m == 64
    assert df.KernelTuningOptions().mode == "auto"
    assert re.match(r"^\d+\.\d+\.\d+$", df.__version__) is not None


def test_unpadded_fast_path_is_exposed_publicly():
    import disentangled_flash as df

    optimize_signature = inspect.signature(df.optimize_deberta)
    enable_signature = inspect.signature(df.enable_deberta_inference)

    assert "assume_unpadded" in optimize_signature.parameters
    assert "assume_unpadded" in enable_signature.parameters

    assert optimize_signature.parameters["assume_unpadded"].default is False
    assert enable_signature.parameters["assume_unpadded"].default is False
