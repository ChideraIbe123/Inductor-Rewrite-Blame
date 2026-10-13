import pytest


def pytest_collection_modifyitems(config, items):
    # tests in files named test_u_*.py are unit tests; test_i_*.py compile with Inductor
    for item in items:
        fn = item.fspath.basename
        if fn.startswith("test_u_"):
            item.add_marker(pytest.mark.unit)
        elif fn.startswith("test_i_"):
            item.add_marker(pytest.mark.inductor)


# Planted-rewrite expectations below encode what Inductor's CPU backend does (oneDNN folding, the
# sfdp rule pair, pointless_convert on the C++ side). On a GPU the same graphs take other rules, so
# those tests are skipped there and the GPU behaviour is covered by the verify tables instead.
from rewrite_blame import env as _env  # noqa: E402

cpu_only = pytest.mark.skipif(_env.is_cuda(), reason="expectation specific to the CPU backend")
