import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_source_tree_imports_without_cycles():
    import vlmintune
    import vlmintune.eval.__main__
    import vlmintune.eval.methods.local_method
    import vlmintune.training.__main__

    assert vlmintune.Method.__name__ == "Method"
    assert vlmintune.FreezeTuningMethod.__name__ == "FreezeTuningMethod"
    assert callable(vlmintune.registry.build_training_method)
