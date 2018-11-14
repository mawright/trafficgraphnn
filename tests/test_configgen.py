import os
import pytest
import tempfile

from trafficgraphnn.genconfig import ConfigGenerator

@pytest.fixture()
def cleandir():
    with tempfile.TemporaryDirectory() as path:
        yield path


@pytest.fixture()
def default_config_gen(cleandir):
    return ConfigGenerator('test', os.path.join(cleandir, 'data/networks'))


def test_constructor_defaults(cleandir, default_config_gen):
    genconfig = default_config_gen
    assert not os.path.exists(genconfig.output_data_dir)

    net_dir = os.path.join(cleandir, 'data/networks', 'test')
    assert genconfig.net_output_file == os.path.join(
        net_dir, 'test.net.xml')

def test_netgenerate_bin(default_config_gen):
    from trafficgraphnn.utils import get_sumo_dir

    genconfig = default_config_gen
    expected_binfile = os.path.join(get_sumo_dir(), 'bin', 'netgenerate')

    assert genconfig.netgenerate_bin == expected_binfile
