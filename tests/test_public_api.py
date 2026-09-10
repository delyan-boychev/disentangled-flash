def test_public_api_imports():
    import re

    import disentangled_flash as df

    assert callable(df.optimize_deberta)
    assert callable(df.enable_deberta_inference)
    assert df.KernelConfig(64, 64, 4).block_m == 64
    assert df.KernelTuningOptions().mode == "auto"
    assert re.match(r"^\d+\.\d+\.\d+$", df.__version__) is not None
