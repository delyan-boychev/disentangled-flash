import inspect


def test_public_api_imports():
    import re

    import disentangled_flash as df

    assert callable(df.optimize_deberta)
    assert callable(df.enable_deberta_inference)
    assert callable(df.pack_padded_with_info)
    assert df.KernelConfig(64, 64, 4).block_m == 64
    assert df.KernelTuningOptions().mode == "auto"
    assert callable(df.enable_deberta_training)
    assert df.DebertaV2OptimizedEncoder is not None
    assert re.match(r"^\d+\.\d+\.\d+$", df.__version__) is not None


def test_unpadded_fast_path_is_exposed_publicly():
    import disentangled_flash as df

    optimize_signature = inspect.signature(df.optimize_deberta)
    enable_signature = inspect.signature(df.enable_deberta_inference)

    assert "assume_unpadded" in optimize_signature.parameters
    assert "assume_unpadded" in enable_signature.parameters

    assert optimize_signature.parameters["assume_unpadded"].default is False
    assert enable_signature.parameters["assume_unpadded"].default is False


def test_unified_backend_defaults_are_explicit():
    import disentangled_flash as df

    optimize_signature = inspect.signature(df.optimize_deberta)
    training_signature = inspect.signature(df.enable_deberta_training)

    assert optimize_signature.parameters["backend"].default == "triton"
    assert optimize_signature.parameters["inference"].default is True
    assert training_signature.parameters["backend"].default == "triton"
