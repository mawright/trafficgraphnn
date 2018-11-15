import logging
import os
import sys
import six
from lxml import etree

if six.PY2:
    try:
        import subprocess32 as subprocess
    except ImportError:
        import subprocess
else:
    import subprocess

import sumolib

from trafficgraphnn.utils import get_sumo_tools_dir, get_net_name, get_net_dir

_logger = logging.getLogger(__name__)

class RandTripGeneratorWrapper(object):
    def __init__(self,
                 netfile,
                 period,
                 binomial,
                 start_time=0,
                 end_time=3600,
                 seed=None,
                 thru_only=True,
                 trip_attrib=('departLane="best" departSpeed="max" departPos="random" '
                              'speedFactor="normc(1,0.1,0.2,2)"'),
    ):
        self.netfile = netfile
        self.period = period
        # lower period means more cars
        self.binomial = binomial
        self.start_time = start_time
        self.end_time = end_time
        self.seed = seed
        self.thru_only = thru_only
        self.trip_attrib = trip_attrib

        self.net_dir = get_net_dir(netfile)
        self.net_name = get_net_name(netfile)

    def generate(self, tag='random'):
        tools_dir = get_sumo_tools_dir()
        pyfile = os.path.join(tools_dir, 'randomTrips.py')

        tripfile = os.path.join(
            self.net_dir,
            '{}_{}.trips.xml'.format(self.net_name, tag))
        routefile = os.path.join(
            self.net_dir,
            '{}_{}.routes.xml'.format(self.net_name, tag))

        if self.thru_only:
            trip_weights_prefix = os.path.join(
                self.net_dir, 'trip-weights-temp')
            weight_file_args = [
                sys.executable, pyfile,
                '--net-file', self.netfile,
                '--output-trip-file', tripfile,
                '--weights-output-prefix', trip_weights_prefix
            ]
            if self.seed is not None:
                weight_file_args.extend(['--seed', str(self.seed)])
            _logger.debug('Calling %s', ' '.join(weight_file_args))
            weight_file_proc = subprocess.Popen(weight_file_args,
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.STDOUT,
                                                universal_newlines=True)
            out, _ = weight_file_proc.communicate()
            if out is not None and len(out) > 0:
                _logger.debug('Returned %s', out)
            set_non_fringe_weights_to_zero(
                trip_weights_prefix, self.netfile)

        randtrip_args = [
            sys.executable, pyfile,
            '--net-file', self.netfile,
            '--output-trip-file', tripfile,
            '--route-file', routefile,
            '--period', str(self.period),
            '--begin', str(self.start_time),
            '--end', str(self.end_time),
            '--trip-attributes', self.trip_attrib,
            '--vehicle-class', 'passenger',
        ]
        if self.binomial is not None:
            randtrip_args.extend([
                '--binomial', str(self.binomial)])
        if self.thru_only:
            randtrip_args.extend([
                '--weights-prefix', trip_weights_prefix])
        if self.seed is not None:
            randtrip_args.extend(['--seed', str(self.seed)])

        _logger.debug('Calling %s', ' '.join(randtrip_args))
        tripgenproc = subprocess.Popen(randtrip_args,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       universal_newlines=True)
        out, _ = tripgenproc.communicate()
        if out is not None and len(out) > 0:
            _logger.debug('Returned %s', out)
        _logger.debug('Wrote random route file to %s', routefile)

        self.tripfile = tripfile
        self.routefile = routefile
        return routefile

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

def gen_data(sumo_network, num_runs, seed=None):
    pass
