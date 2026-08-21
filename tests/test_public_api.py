def test_public_api_imports():
    import re

    import disentangled_flash as df

    assert callable(df.optimize_deberta)
    assert callable(df.enable_deberta_inference)
    assert re.match(r"^\d+\.\d+\.\d+$", df.__version__) is not None
