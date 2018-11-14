import os
import pytest
import trafficgraphnn


def test_env_var():
    assert 'SUMO_HOME' in os.environ


def test_sumo_tools_dir():
    try:
        import sumolib
    except ImportError:
        pytest.fail("Could not find sumolib")
