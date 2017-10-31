import numpy as np
import scipy.sparse
import six
from collections import OrderedDict
import logging
import os

from sumolib import checkBinary
from sumolib.net import readNet
import traci

from trafficgraphnn.genconfig import ConfigGenerator

logger = logging.getLogger(__name__)


class SumoNetwork(object):
    def __init__(self, netfile, lanewise=False, undirected_graph=False):
        self.netfile = netfile
        self.net = readNet(netfile)

        if not lanewise:
            self.A = get_edge_adj_matrix(netfile, undirected_graph)
            self.tls_to_controlled_dict = {
                tl.getID(): [e.getID() for e in tl.getEdges()]
                for tl in self.net.getTrafficLights()
            }
        else:
            self.A = get_lane_adj_matrix(netfile, undirected_graph)
            self.tls_to_controlled_dict = {
                tl.getID(): list(OrderedDict.fromkeys(
                    [conn[0].getID() for conn in tl.getConnections()]))
                for tl in self.net.getTrafficLights()
            }

        self.routefile = None
        self.additional_files = []
        self.seed = None

        self.binfile = checkBinary('sumo')

    def set_routefile(self, routefile):
        assert os.path.exists(routefile)
        self.routefile = routefile

    def set_seed(self, seed):
        self.seed = seed

    def add_additional_file(self, addlfile):
        assert os.path.exists(addlfile)
        self.additional_files.append(addlfile)

    def clear_additional_files(self):
        self.additional_files = []

    def get_sumo_command(self, with_bin_file=True):
        if self.routefile is None:
            raise ValueError('Route file not set.')
        sumo_args = [
            '--net-file', self.netfile,
            '--route-files', self.routefile,
            '--no-step-log'
        ]

        if len(self.additional_files) > 0:
            sumo_args.extend(
                ['--additional_files'] + self.additional_files
            )
        if self.seed is not None:
            assert type(self.seed) in six.integer_types
            sumo_args.extend(['--seed', str(self.seed)])
        else:
            logger.warn('Seed not set, SUMO seed will be random.')
            sumo_args.extend(['--random', 'true'])

        if with_bin_file:
            return [self.binfile] + sumo_args
        else:
            return sumo_args  # used for passing to traci.load()

    def start(self):
        traci.start(self.get_sumo_command())

    def get_loop_ids(self):
        pass

    def get_edge_adj_matrix(self, undirected=False):
        return get_edge_adj_matrix(self.netfile, undirected)

    def get_lane_adj_matrix(self, undirected=False):
        return get_lane_adj_matrix(self.netfile, undirected)

    def get_config_generator(self):
        return ConfigGenerator(
            os.path.splitext(
                os.path.splitext(
                    os.path.basename(self.netfile))[0])[0],
            net_config_dir=os.path.dirname(self.netflie),
        )


def get_edge_adj_matrix(netfile, undirected=False):
    net = readNet(netfile)

    # net.getEdges() returns a set...make sure it has stable ordering
    edge_list = list(net.getEdges())
    num_edges = len(edge_list)

    edge_ids = [e.getID() for e in edge_list]

    A = scipy.sparse.lil_matrix((num_edges, num_edges))

    for from_index, edge in enumerate(edge_list):
        assert from_index == edge_ids.index(edge.getID())
        for out in edge.getOutgoing().keys():
            to_index = edge_ids.index(out.getID())
            A[from_index, to_index] = 1

    # for node in net.getNodes():
        # for conn in node.getConnections():
            # from_index = edge_ids.index(conn.getFrom().getID())
            # to_index = edge_ids.index(conn.getTo().getID())
            # A[from_index, to_index] = 1

    if undirected:
        A = make_undirected(A)

    if type(A) is not scipy.sparse.csr.csr_matrix:
        A = scipy.sparse.csr_matrix(A)
    return A


def get_lane_adj_matrix(netfile, undirected=False):
    net = readNet(netfile)

    lane_list = [l for e in net.getEdges() for l in e.getLanes()]
    num_lanes = len(lane_list)

    lane_ids = [l.getID() for l in lane_list]

    A = scipy.sparse.lil_matrix((num_lanes, num_lanes))

    for idx, lane in enumerate(lane_list):
        assert idx == lane_ids.index(lane.getID())
        for conn in lane.getOutgoing():
            to_index = lane_ids.index(conn.getToLane().getID())
            A[idx, to_index] = 1

    if undirected:
        A = make_undirected(A)

    if type(A) is not scipy.sparse.csr.csr_matrix:
        A = scipy.sparse.csr_matrix(A)
    return A


def make_undirected(A):
    no_double_edge = A.T > A
    A_und = A - A.multiply(no_double_edge) + A.T.multiply(no_double_edge)
    assert np.abs(A - A.T).max() < 1e-10
    return A_und
