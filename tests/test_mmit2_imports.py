import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_source_tree_imports_without_cycles():
    import mmit2
    import mmit2.eval.__main__
    import mmit2.training.__main__

    assert mmit2.Method.__name__ == "Method"
    assert mmit2.FreezeTuningMethod.__name__ == "FreezeTuningMethod"
    assert callable(mmit2.registry.build_training_method)
