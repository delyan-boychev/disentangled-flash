def test_public_api_imports():
    import disentangled_flash as df

    assert callable(df.optimize_deberta)
    assert callable(df.enable_deberta_inference)
    assert df.__version__ == "0.1.0"
