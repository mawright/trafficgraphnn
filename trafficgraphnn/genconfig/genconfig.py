from __future__ import absolute_import, print_function, division

import os
import sys
import numpy as np
import six
from lxml import etree
import logging

# sumo path is appended in __init__.py
from sumolib import checkBinary
import sumolib

from trafficgraphnn.utils import get_sumo_tools_dir, get_net_dir, get_net_name
from trafficgraphnn.genconfig import detectors, tls_config

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
        self, net_name, net_config_dir='data/networks'
        ):
        self.net_config_dir = os.path.join(net_config_dir, net_name)
        self.net_name = net_name
        self.output_data_dir = os.path.join(self.net_config_dir, 'output')

        self.net_output_file = os.path.join(
            self.net_config_dir, self.net_name + '.net.xml')

        self.netgenerate_bin = checkBinary('netgenerate')

        self.tripfile = None
        self.routefile = None
        self.detector_def_files = []
        self.detector_output_files = []
        self.non_detector_addl_files = []

    def gen_grid_network(
        self, check_lane_foes_all=True,
        grid_number=5, grid_length=100, num_lanes=3,
        tlstype='static', gen_editable_xml=False, simplify_tls=False
    ):
        netgenerate_bin = checkBinary('netgenerate')

        netgen_args = [
            netgenerate_bin, '--grid',
            '--grid.number', str(grid_number),
            '--grid.length', str(grid_length),
            '--grid.attach-length', str(grid_length),
            '--default-junction-type', 'traffic_light',
            '--tls.guess', 'true',
            '--tls.default-type', tlstype,
            '--default.lanenumber', str(num_lanes),
            '--check-lane-foes.all', str(check_lane_foes_all).lower(),
            '-o', self.net_output_file,
        ]
        if gen_editable_xml:
            plain_output_dir = os.path.join(self.net_config_dir, self.net_name)
            os.makedirs(plain_output_dir)
            netgen_args.extend([
                '--plain-output-prefix', plain_output_dir + self.net_name])

        logger.debug('Calling %s', ' '.join(netgen_args))

        if not os.path.exists(self.net_config_dir):
            os.makedirs(self.net_config_dir)
        netgenproc = subprocess.Popen(netgen_args, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT,
                                      universal_newlines=True)
        out, _ = netgenproc.communicate()
        if out is not None:
            logger.debug('Returned %s', out)
        logger.debug('Wrote grid network to {}'.format(self.net_output_file))

        if num_lanes == 3 and simplify_tls == True:
            tls_config.tls_config(self.net_output_file)

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

        logger.debug('Calling %s', ' '.join(netgen_args))
        if not os.path.exists(self.net_config_dir):
            os.makedirs(self.net_config_dir)
        netgenproc = subprocess.Popen(netgen_args, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT,
                                      universal_newlines=True)
        out, _ = netgenproc.communicate()
        if out is not None:
            logger.debug('Returned %s', out)
        logger.debug('Wrote random network to %s', self.net_output_file)

    def gen_rand_trips(
        self, tripfile_name=None, routefile_name=None,
        period=None, binomial=None, seed=None,
        start_time=0, end_time=3600, thru_only=True, fringe_factor=100,
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
        # lower period leads to more cars

        if thru_only:
            trip_weights_prefix = os.path.join(
                self.net_config_dir, 'trip-weights-temp')
            weight_file_args = [
                sys.executable, pyfile,
                '--net-file', self.net_output_file,
                '--output-trip-file', self.tripfile,
                '--weights-output-prefix', trip_weights_prefix
            ]
            weight_file_proc = subprocess.Popen(weight_file_args,
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.STDOUT,
                                                universal_newlines=True)
            out, _ = weight_file_proc.communicate()
            if out is not None:
                logger.debug('Returned %s', out)
            set_non_fringe_weights_to_zero(
                trip_weights_prefix, self.net_output_file)

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
        if thru_only:
            randtrip_args.extend([
                '--weights-prefix', trip_weights_prefix])

        logger.debug('Calling %s', ' '.join(randtrip_args))
        tripgenproc = subprocess.Popen(randtrip_args,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       universal_newlines=True)
        out, _ = tripgenproc.communicate()
        if out is not None:
            logger.debug('Returned %s', out)
        logger.debug('Wrote random route file to %s', self.routefile)

    def gen_detectors(
        self,
        detector_type,
        detector_def_file=None,
        detector_output_file=None,
        distance_to_tls=5,
        frequency=60,
        detector_length=None,
    ):
        def_filepath, output_filepath = detectors.generate_detector_set(
            self.net_output_file, detector_type, distance_to_tls,
            detector_def_file, detector_output_file, detector_length,
            frequency)

        self.detector_def_files.append(def_filepath)
        self.detector_output_files.append(output_filepath)

    def gen_e1_detectors(
        self,
        detector_def_file_name=None,
        detector_output_file=None,
        distance_to_tls=5,
        frequency=60,
    ):
        def_filepath, output_filepath = detectors.generate_e1_detectors(
            self.net_output_file, distance_to_tls=distance_to_tls,
            detector_def_file=detector_def_file_name,
            detector_output_file=detector_output_file, frequency=frequency)

        self.detector_def_files.append(def_filepath)
        self.detector_output_files.append(output_filepath)

    def gen_e2_detectors(
        self,
        detector_def_file_name=None,
        detector_output_file=None,
        distance_to_tls=0,
        detector_length=None,
        frequency=60,
    ):
        def_filepath, output_filepath = detectors.generate_e2_detectors(
            self.net_output_file, distance_to_tls=distance_to_tls,
            detector_def_file=detector_def_file_name,
            detector_output_file=detector_output_file,
            detector_length=detector_length, frequency=frequency)

        self.detector_def_files.append(def_filepath)
        self.detector_output_files.append(output_filepath)

    def define_tls_output_file(
        self,
        tls_subset=None,
        output_file_name=None,
        addl_file_name='tls_output.add.xml'
    ):
        addl_file = define_tls_output_file(
            self.net_output_file, tls_subset=tls_subset,
            output_data_dir=self.output_data_dir,
            output_file_name=output_file_name,
            config_dir=self.net_config_dir, addl_file_name=addl_file_name)

        if addl_file not in self.non_detector_addl_files:
            self.non_detector_addl_files.append(addl_file)

        return os.path.realpath(addl_file)


def define_tls_output_file(
    netfile,
    tls_subset=None,
    output_data_dir=None,
    output_file_name=None,
    config_dir=None,
    addl_file_name='tls_output.add.xml',
):
    if config_dir is None:
        config_dir = get_net_dir(netfile)

    if output_data_dir is None:
        output_data_dir = os.path.join(config_dir, 'output')

    if output_file_name is None:
        net_name = get_net_name(netfile)
        output_file_name = '{}_tls_output.xml'.format(net_name)

    output_file = os.path.join(output_data_dir, output_file_name)

    addl_file = os.path.join(config_dir, addl_file_name)

    relative_output_filename = os.path.relpath(
        output_file,
        os.path.dirname(addl_file))

    tls_addl = sumolib.xml.create_document('additional')

    net = sumolib.net.readNet(netfile)

    for tls in net.getTrafficLights():
        tls_xml_element = tls_addl.addChild('timedEvent')
        tls_xml_element.setAttribute('type', 'SaveTLSSwitchTimes')
        tls_xml_element.setAttribute('source', tls.getID())
        tls_xml_element.setAttribute('dest', relative_output_filename)

    tls_addl_file_fid = open(os.path.realpath(addl_file), 'w')
    tls_addl_file_fid.write(tls_addl.toXML())
    tls_addl_file_fid.close()

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    return os.path.realpath(addl_file)


def set_non_fringe_weights_to_zero(weights_file_prefix, netfile):
    net = sumolib.net.readNet(netfile)

    src_def_file = weights_file_prefix + '.src.xml'
    dst_def_file = weights_file_prefix + '.dst.xml'
    src_etree = etree.parse(src_def_file)
    for element in src_etree.getroot().iter('edge'):
        edge = net.getEdge(element.attrib['id'])
        if edge.is_fringe(edge.getIncoming()):
            element.attrib['value'] = '1.00'  # this is a source edge
        else:
            element.attrib['value'] = '0.00'  # not a source edge
    src_etree.write(src_def_file)

    dst_etree = etree.parse(dst_def_file)
    for element in dst_etree.getroot().iter('edge'):
        edge = net.getEdge(element.attrib['id'])
        if edge.is_fringe(edge.getOutgoing()):
            element.attrib['value'] = '1.00'  # this is a sink edge
        else:
            element.attrib['value'] = '0.00'  # not a sink edge
    dst_etree.write(dst_def_file)


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
