import pytest


def pytest_collection_modifyitems(config, items):
    # tests in files named test_u_*.py are unit tests; test_i_*.py compile with Inductor
    for item in items:
        fn = item.fspath.basename
        if fn.startswith("test_u_"):
            item.add_marker(pytest.mark.unit)
        elif fn.startswith("test_i_"):
            item.add_marker(pytest.mark.inductor)
