import numpy as np
import scipy.sparse
import six
from collections import OrderedDict
import logging
import os
import networkx as nx
from lxml import etree

from sumolib import checkBinary
from sumolib.net import readNet
import traci

from trafficgraphnn.genconfig import ConfigGenerator
from trafficgraphnn.utils import parse_detector_output_xml

logger = logging.getLogger(__name__)

if six.PY2:
    try:
        import subprocess32 as subprocess
    except ImportError:
        import subprocess
else:
    import subprocess


class SumoNetwork(object):
    def __init__(
        self, netfile, lanewise=False, undirected_graph=False,
        routefile=None, addlfiles=None, seed=None, binfile='sumo'
    ):
        self.netfile = netfile
        self.net = readNet(netfile)
        self.undirected_graph = undirected_graph
        self.lanewise = lanewise
        self.routefile = routefile
        self.seed = seed
        self.data_dfs = []

        if isinstance(addlfiles, six.string_types):
            addlfiles = [addlfiles]
        self.additional_files = addlfiles or []

        self.tls_list = self.net.getTrafficLights()
        # tl.getLinks() returns a dict with a consistent ordering of movements

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

        self.config_gen = self.get_config_generator()

        self.binfile = checkBinary(binfile)

        self.reset_graph()

    @classmethod
    def from_gen_config(
        cls, config_gen, lanewise=False, undirected_graph=False,
        seed=None
    ):
        return cls(
            config_gen.net_output_file, lanewise=lanewise,
            undirected_graph=undirected_graph, routefile=config_gen.routefile,
            addlfiles=[config_gen.detector_def_file]
        )

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

    def get_sumo_command(self, with_bin_file=True, queue_output_file=None):
        if self.routefile is None:
            raise ValueError('Route file not set.')
        sumo_args = [
            '--net-file', self.netfile,
            '--route-files', self.routefile,
            '--no-step-log'
        ]

        if len(self.additional_files) > 0:
            sumo_args.extend(
                ['--additional-files', ','.join(self.additional_files)]
            )
        if queue_output_file is not None:
            sumo_args.extend(
                ['--queue-output', queue_output_file]
            )
        if self.seed is not None:
            assert type(self.seed) in six.integer_types
            sumo_args.extend(['--seed', str(self.seed)])
        else:
            logger.warn('Seed not set, SUMO seed will be random.')
            sumo_args.extend(['--random', 'true'])

        if self.binfile == 'sumo-gui':
            sumo_args.extend(['--start', '--quit'])

        if with_bin_file:
            return [self.binfile] + sumo_args
        else:
            return sumo_args  # used for passing to traci.load()

    def start(self):
        traci.start(self.get_sumo_command())

    def run(self):
        return subprocess.call(self.get_sumo_command())

    def sorted_lanes_for_edge(self, edge_id):
        lanes = self.net.getEdge(edge_id).getLanes()
        lanes.sort(key=lambda x: x.getIndex())
        return [lane.getID() for lane in lanes]

    def get_edges_at_junction(self, node_id):
        pass

    def reset_graph(self, undirected=None, lanewise=None):
        if undirected is not None:
            self.undirected_graph = undirected

        if lanewise is not None:
            self.lanewise = lanewise

        if self.lanewise:
            self.graph = get_lane_graph(
                self.netfile, undirected=self.undirected_graph,
                detector_files=self.additional_files)
        else:
            self.graph = get_edge_graph(
                self.netfile, undirected=self.undirected_graph,
                detector_files=self.additional_files)

    def get_graph(self):
        assert self.graph is not None
        return self.graph

    def get_edge_adj_matrix(self, undirected=False):
        return get_edge_adj_matrix(self.netfile, undirected)

    def get_lane_adj_matrix(self, undirected=False):
        return get_lane_adj_matrix(self.netfile, undirected)

    def get_config_generator(self):
        config_gen = ConfigGenerator(
            os.path.splitext(
                os.path.splitext(
                    os.path.basename(self.netfile))[0])[0],
            net_config_dir=os.path.dirname(self.netfile),
        )
        return config_gen

    def load_data_to_graph(self):
        if self.graph is None:
            return

        self.data_dfs = []

        detectors_in_files = {}
        det_to_data = {}
        for _, detector_data in self.graph.nodes.data('detectors'):
            for det_id, data in detector_data.items():
                if data['file'] not in detectors_in_files.keys():
                    detectors_in_files[data['file']] = []
                detectors_in_files[data['file']].append(det_id)
                det_to_data[det_id] = data

        for file, det_list in detectors_in_files.items():
            filename = os.path.join(os.path.dirname(self.netfile), file)
            if not os.path.exists(filename):
                continue
            df = parse_detector_output_xml(filename, ids=det_list)
            self.data_dfs.append(df)

            for det_id in det_list:
                det_data = det_to_data[det_id]
                det_data['data_series'] = df.xs(det_id, level=1)

    def generate_datasets(
        self, num_simulations=20, simulation_length=3600,
        routefile_period=None, routefile_binomial=None
    ):

        config_gen = self.get_config_generator()

        # if not hasattr

        # for i in range(num_simulations):
        # con


def get_lane_graph(netfile, undirected=False, detector_files=None):
    net = readNet(netfile)

    if undirected:
        graph = nx.Graph()
    else:
        graph = nx.DiGraph()

    for edge in net.getEdges():
        for lane in edge.getLanes():
            graph.add_node(lane.getID())

    for node in net.getNodes():
        for conn in node.getConnections():
            graph.add_edge(
                conn.getFromLane().getID(), conn.getToLane().getID(),
                direction=conn.getDirection())

    if detector_files is not None:
        if isinstance(detector_files, six.string_types):
            detector_files = [detector_files]
        for det_file in detector_files:
            tree = etree.parse(det_file)
            for element in tree.iter():
                if element.tag in [
                    'e1Detector', 'inductionLoop',
                    'e2Detector', 'laneAreaDetector'
                ]:
                    lane_id = element.get('lane')
                    if lane_id in graph.nodes:
                        if 'detectors' not in graph.node[lane_id]:
                            graph.node[lane_id]['detectors'] = {}

                        detector_info_dict = dict(element.items())
                        detector_info_dict['type'] = element.tag
                        graph.node[lane_id]['detectors'][
                            element.get('id')] = detector_info_dict

    return graph


def get_edge_graph(netfile, undirected=False, detector_files=None):
    net = readNet(netfile)

    if undirected:
        graph = nx.Graph()
    else:
        graph = nx.DiGraph()

    for edge in net.getEdges():
        graph.add_node(
            edge.getID(), lanes=[l.getID() for l in edge.getLanes()])
        for conn_list in edge.getOutgoing().values():
            for conn in conn_list:
                graph.add_edge(
                    conn.getFrom().getID(), conn.getTo().getID(),
                    direction=conn.getDirection()
                )

    if detector_files is not None:
        edge_info = dict(graph.nodes.data())
        if isinstance(detector_files, str):
            detector_files = [detector_files]
        for det_file in detector_files:
            tree = etree.parse(det_file)
            for element in tree.iter():
                if element.tag in [
                    'e1Detector', 'inductionLoop',
                    'e2Detector', 'laneAreaDetector'
                ]:
                    lane_id = element.get('lane')
                    edge_id = net.getLane(lane_id).getEdge().getID()
                    detector_info_dict = dict(element.items())
                    detector_info_dict['type'] = element.tag
                    edge_info[edge_id][element.get('id')] = detector_info_dict

        nx.set_node_attributes(graph, edge_info)

    return graph


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
    assert np.abs(A_und - A_und.T).max() < 1e-10
    return A_und
