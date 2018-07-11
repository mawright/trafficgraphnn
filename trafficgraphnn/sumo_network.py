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
from trafficgraphnn.utils import (
    parse_detector_output_xml, parse_tls_output_xml, iterfy)

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
        self, netfile, lanewise=True, undirected_graph=False,
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
        cls, config_gen, lanewise=True, undirected_graph=False,
        seed=None
    ):
        return cls(
            config_gen.net_output_file, lanewise=lanewise,
            undirected_graph=undirected_graph, routefile=config_gen.routefile,
            addlfiles=(
                list(iterfy(config_gen.detector_def_files))
                + list(iterfy(config_gen.non_detector_addl_files)))
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
            logger.warning('Seed not set, SUMO seed will be random.')
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
                additional_files=self.additional_files)
        else:
            self.graph = get_edge_graph(
                self.netfile, undirected=self.undirected_graph,
                additional_files=self.additional_files)

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

    def load_data_to_graph(self, features=None):
        if self.graph is None:
            return

        self.data_dfs = []

        detectors_in_files = {}
        det_to_data = {}
        for node, detector_data in self.graph.nodes.data('detectors'):
            if detector_data is not None:
                for det_id, data in detector_data.items():
                    if data['file'] not in detectors_in_files.keys():
                        detectors_in_files[data['file']] = []
                    detectors_in_files[data['file']].append(det_id)
                    det_to_data[det_id] = data

        det_to_node = self.det_to_node_dict()

        for file, det_list in detectors_in_files.items():
            filename = os.path.join(os.path.dirname(self.netfile), file)
            if not os.path.exists(filename):
                continue
            df = parse_detector_output_xml(filename, det_list, features)

            df['node_id'] = df.index.get_level_values('det_id')
            df['node_id'].replace(det_to_node, inplace=True)

            df.set_index('node_id', append=True, inplace=True)
            df = df.swaplevel('det_id', 'node_id')

            self.data_dfs.append(df)

            for det_id in det_list:
                det_data = det_to_data[det_id]
                det_data['data_series'] = df.xs(det_id, level='det_id')

        for node_id, data in self.graph.nodes.data():
            if 'detectors' in data:
                data['data_series'] = [df.xs(node_id, level='node_id')
                                       for df in self.data_dfs]

        tlses_in_files = {}
        for (in_node, out_node, data
             ) in self.graph.edges.data('tls_output_info'):
            if data is not None:
                file = data['dest']
                if file not in tlses_in_files:
                    tlses_in_files[file] = set()
                tlses_in_files[file].add(data['source'])

        for file, tls_list in tlses_in_files.items():
            filename = os.path.join(os.path.dirname(self.netfile), file)

            df = parse_tls_output_xml(filename)

            for from_lane, to_lane, data in self.graph.edges.data():
                if data['tls'] in tls_list:
                    data['switch_times'] = df.xs(
                        (from_lane, to_lane),
                        level=('fromLane', 'toLane'), drop_level=False)

            self.data_dfs.append(df)

            # TODO? version where graph.nodes are roads instead of lanes

    def get_lane_data_and_adj_matrix(self, node_ordering=None):
        """
        Returns a tuple of (A, X), where A is a Scipy sparse matrix from the
        networkx graph and X is a numpy ndarray of the data.
        X will have dimensions time x node x feature

        :param node_ordering: (optional) Iterable of node names. If passed a
        value, this function will return A and X in that order. If None
        (default) will return the order determined by networkx.
        :type node_ordering: Iterable
        """
        raise NotImplementedError
        if self.graph is None:
            raise ValueError('Graph not set.')
        if len(self.data_dfs) == 0:
            raise ValueError('No data loaded.')

        graph = self.get_graph()

        A = nx.adj_matrix(graph, node_ordering)

        det_to_node = self.det_to_node_dict()

    def det_to_node_dict(self):
        if self.graph is None:
            raise ValueError("Graph not set.")

        graph = self.get_graph()

        det_to_node = {}
        for node, det_dict in graph.nodes('detectors'):
            if det_dict is not None:
                for det_id, _ in det_dict.items():
                    det_to_node[det_id] = node

        return det_to_node

    def get_data(self, nodes, features):
        """Returns the network's data in ndarray format for a particular node
        ordering and feature ordering.

        Given a node and feature ordering, returns the data in the shape
        [Time x Node x Feature]
        :param nodes: Iterable containing node names from the networkx
        graph.
        :type nodes: Iterable
        :param features: Iterable containing feature names
        """
        raise NotImplementedError

    def generate_datasets(
        self, num_simulations=20, simulation_length=3600,
        routefile_period=None, routefile_binomial=None
    ):
        raise NotImplementedError

        config_gen = self.get_config_generator()

        # if not hasattr

        # for i in range(num_simulations):


def get_lane_graph(netfile, undirected=False, additional_files=None):
    net = readNet(netfile)

    if undirected:
        graph = nx.Graph()
    else:
        graph = nx.DiGraph()

    for edge in net.getEdges():
        for lane in edge.getLanes():
            graph.add_node(lane.getID())

    tls_to_edges = {}

    for node in net.getNodes():
        for conn in node.getConnections():
            tls_id = conn.getTLSID()
            if tls_id not in tls_to_edges:
                tls_to_edges[tls_id] = []
            edge_from_to = (conn.getFromLane().getID(),
                            conn.getToLane().getID())
            graph.add_edge(
                *edge_from_to,
                direction=conn.getDirection(),
                tls=tls_id)
            tls_to_edges[tls_id].append(edge_from_to)

    if additional_files is not None:
        if isinstance(additional_files, six.string_types):
            additional_files = [additional_files]
        for addl_file in additional_files:
            tree = etree.parse(addl_file)
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

                elif (element.tag == 'timedEvent' and
                      element.get('type') == 'SaveTLSSwitchTimes'):
                    tls_id = element.get('source')
                    for edge in tls_to_edges[tls_id]:
                        graph.edges[edge].update(
                            {'tls_output_info': dict(element.items())})

    return graph


def e2_detector_graph(
    netfile, detector_file, undirected=False, lanewise=True
):
    net = readNet(netfile)

    tree = etree.parse(detector_file)

    if undirected:
        detector_graph = nx.Graph()
    else:
        detector_graph = nx.DiGraph()

    for element in tree.iter():
        if element.tag in ['e2Detector', 'laneAreaDetector']:
            det_id = element.get('id')
            info_dict = dict(element.items())
            detector_graph.add_node(det_id, **info_dict)

    lane_to_det = {lane: det for det, lane in detector_graph.node('lane')}

    for node in net.getNodes():
        for conn in node.getConnections():
            detector_graph.add_edge(
                lane_to_det[conn.getFromLane().getID()],
                lane_to_det[conn.getToLane().getID()],
                direction=conn.getDirection())

    return detector_graph


def get_edge_graph(netfile, undirected=False, additional_files=None):
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

    if additional_files is not None:
        edge_info = dict(graph.nodes.data())
        if isinstance(additional_files, str):
            additional_files = [additional_files]
        for addl_file in additional_files:
            tree = etree.parse(addl_file)
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
