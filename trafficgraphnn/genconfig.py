from __future__ import absolute_import, print_function, division

import os
import sys
import numpy as np
import six
from lxml import etree
import logging

# sumo path is appended in __init__.py
from sumolib import checkBinary

from trafficgraphnn.utils import get_sumo_tools_dir

if six.PY2:
    try:
        import subprocess32 as subprocess
    except ImportError:
        import subprocess
else:
    import subprocess

logger = logging.getLogger(__name__)


class ConfigGenerator(object):
    def __init__(
        self, net_name, net_config_dir='data/networks',
        output_data_dir='data/output'
    ):
        self.net_config_dir = net_config_dir
        self.net_name = net_name
        self.output_data_dir = output_data_dir

        self.net_output_file = os.path.join(
            self.net_config_dir, self.net_name + '.net.xml')

        self.netgenerate_bin = checkBinary('netgenerate')

        self.tripfile = None
        self.routefile = None
        self.detector_def_file = None
        self.detector_output_file = None

    def gen_grid_network(
        self, check_lane_foes_all=True,
        grid_number=5, grid_length=100,
        tlstype='static', gen_editable_xml=False
    ):
        netgenerate_bin = checkBinary('netgenerate')

        netgen_args = [
            netgenerate_bin, '--grid',
            '--grid.number', str(grid_number),
            '--grid.length', str(grid_length),
            '--default-junction-type', 'traffic_light',
            '--tls.guess', 'true',
            '--tls.default-type', tlstype,
            '--default.lanenumber', '2',
            '--check-lane-foes.all', str(check_lane_foes_all).lower(),
            '-o', self.net_output_file,
        ]
        if gen_editable_xml:
            plain_output_dir = os.path.join(self.net_config_dir, self.net_name)
            os.makedirs(plain_output_dir)
            netgen_args.extend([
                '--plain-output-prefix', plain_output_dir + self.net_name])

        logger.info('Calling {}'.format(' '.join(netgen_args)))
        netgenproc = subprocess.Popen(netgen_args)
        netgenproc.wait()
        logger.info('Wrote grid network to {}'.format(self.net_output_file))

    def gen_rand_network(
        self, check_lane_foes_all=True,
        rand_iterations=200, seed=None,
        tlstype='static', gen_editable_xml=False
    ):
        netgenerate_bin = checkBinary('netgenerate')

        if seed is None:
            seed = np.random.randint(int(1e5))

        netgen_args = [
            netgenerate_bin, '--rand',
            '--seed', str(seed),
            '--rand.iterations', str(rand_iterations),
            '--default-junction-type', 'traffic_light',
            '--tls.guess', 'true',
            '--tls.default-type', tlstype,
            '--default.lanenumber', '2',
            '--check-lane-foes.all', str(check_lane_foes_all).lower(),
            '-o', self.net_output_file,
        ]
        if gen_editable_xml:
            plain_output_dir = os.path.join(self.net_config_dir, self.net_name)
            os.makedirs(plain_output_dir)
            netgen_args.extend([
                '--plain-output-prefix', plain_output_dir + self.net_name])

        logger.info('Calling {}'.format(' '.join(netgen_args)))
        netgenproc = subprocess.Popen(netgen_args)
        netgenproc.wait()
        logger.info('Wrote random network to {}'.format(self.net_output_file))

    def gen_rand_trips(
        self, tripfile_name=None, routefile_name=None,
        period=None, binomial=None, seed=None,
        start_time=0, end_time=3600, fringe_factor=10,
        trip_attrib="departLane=\"best\" departSpeed=\"max\" departPos=\"random\"",
    ):
        if tripfile_name is None:
            tripfile_name = '{}_rand_trips.trips.xml'.format(self.net_name)
        if routefile_name is None:
            routefile_name = '{}_rand_routes.routes.xml'.format(self.net_name)
        if seed is None:
            seed = 100
        np.random.seed(seed)

        self.tripfile = os.path.join(self.net_config_dir, tripfile_name)
        self.routefile = os.path.join(self.net_config_dir, routefile_name)

        tools_dir = get_sumo_tools_dir()
        pyfile = os.path.join(tools_dir, 'randomTrips.py')

        if binomial is None:
            binomial = np.random.randint(1, 10)
        if period is None:
            period = np.random.uniform(0.2, 1.2)

        randtrip_args = [
            sys.executable, pyfile,
            '--net-file', self.net_output_file,
            '--output-trip-file', self.tripfile,
            '--route-file', self.routefile,
            '--binomial', str(binomial),
            '--period', str(period),
            '--begin', str(start_time),
            '--end', str(end_time),
            '--seed', str(seed),
            '--fringe-factor', str(fringe_factor),
            '--trip-attributes', trip_attrib,
        ]

        logger.info('Calling {}'.format(' '.join(randtrip_args)))
        tripgenproc = subprocess.Popen(randtrip_args)
        tripgenproc.wait()
        logger.info('Wrote random route file to {}'.format(self.routefile))

    def gen_detectors(
        self,
        detector_type,
        detector_def_file_name=None,
        detector_output_file=None,
        distance_to_tls=5,
        frequency=60,
    ):
        # TODO write own script that can place multiple e1 detectors on the same lane
        if detector_type not in ['e1', 'e2', 'e3']:
            raise ValueError('Unknown detector type: {}'.format(detector_type))
        tools_dir = get_sumo_tools_dir()

        if detector_type == 'e1':
            scriptname = 'generateTLSE1Detectors.py'
        elif detector_type == 'e2':
            scriptname = 'generateTLSE2Detectors.py'

        pyfile = os.path.join(tools_dir, 'output', scriptname)

        if detector_def_file_name is None:
            detector_def_file_name = '{}_{}.add.xml'.format(
                self.net_name, detector_type)
        if detector_output_file is None:
            detector_output_file = '{}_{}_output.xml'.format(
                self.net_name, detector_type)

        self.detector_def_file = os.path.join(
            self.net_config_dir, detector_def_file_name)
        self.detector_output_file = os.path.join(
            self.output_data_dir, detector_output_file)

        gendetectors_args = [
            sys.executable, pyfile,
            '--net-file', self.net_output_file,
            '--distance-to-TLS', str(distance_to_tls),
            '--frequency', str(frequency),
            '--output', detector_def_file_name,
            '--results-file', detector_output_file
        ]
        if detector_type == 'e2':
            gendetectors_args.extend(['--detector_length', '-1'])

        logger.info('Calling {}'.format(' '.join(gendetectors_args)))
        gendetproc = subprocess.Popen(gendetectors_args)
        gendetproc.wait()

        if detector_type == 'e1':
            remove_e1_lengths(self.detector_def_file)
        logger.info('Wrote detector file to {}'.format(self.detector_def_file))

        return self.detector_def_file

    def gen_e1_detectors(
        self,
        detector_def_file_name=None,
        detector_output_file=None,
        distance_to_tls=5,
        frequency=60,
    ):
        return self.gen_detectors(
            'e1', detector_def_file_name, detector_output_file,
            distance_to_tls, frequency)

    def gen_e2_detectors(
        self,
        detector_def_file_name=None,
        detector_output_file=None,
        distance_to_tls=5,
        frequency=60,
    ):
        return self.gen_detectors(
            'e2', detector_def_file_name, detector_output_file,
            distance_to_tls, frequency)


def remove_e1_lengths(detector_def_file):
    det_etree = etree.parse(detector_def_file)
    for element in det_etree.getroot().iter('e1Detector'):
        if 'length' in element.attrib.keys():
            del element.attrib['length']

    det_etree.write(detector_def_file)


def update_e2_tag(detector_def_file):
    det_etree = etree.parse(detector_def_file)
    for element in det_etree.getroot().iter('laneAreaDetector'):
        element.tag = 'e2Detector'

    det_etree.write(detector_def_file)
