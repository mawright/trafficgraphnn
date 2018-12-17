import numpy as np
import six
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

_logger = logging.getLogger(__name__)

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

        self.detector_def_files = []
        self.tls_output_def_files = []
        self.other_addl_files = []

        if isinstance(addlfiles, six.string_types):
            addlfiles = [addlfiles]
        self.additional_files = addlfiles or []
        self.classify_additional_files()

        self.tls_list = self.net.getTrafficLights()
        # tl.getLinks() returns a dict with a consistent ordering of movements

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
                + list(iterfy(config_gen.non_detector_addl_files))),
            seed=seed
        )

    def classify_additional_files(self):
        for addlfile in self.additional_files:
            tree = etree.iterparse(addlfile, tag=['e1Detector', 'inductionLoop',
                                                  'e2Detector', 'laneAreaDetector',
                                                  'timedEvent',])
            _, element = next(tree)
            if element.tag in ['e1Detector', 'inductionLoop',
                               'e2Detector', 'laneAreaDetector']:
                if addlfile not in self.detector_def_files:
                    self.detector_def_files.append(addlfile)
            elif element.tag == 'timedEvent' and element.get('type') == 'SaveTLSSwitchTimes':
                if addlfile not in self.tls_output_def_files:
                    self.tls_output_def_files.append(addlfile)
            else:
                if addlfile not in self.other_addl_files:
                    self.other_addl_files.append(addlfile)
            element.clear()

    def set_routefile(self, routefile):
        assert os.path.exists(routefile)
        self.routefile = routefile

    def set_seed(self, seed):
        self.seed = seed

    def add_additional_file(self, addlfile):
        assert os.path.exists(addlfile)
        self.additional_files.append(addlfile)
        self.classify_additional_files()

    def clear_additional_files(self):
        self.additional_files = []
        self.detector_def_files = []
        self.tls_output_def_files = []
        self.other_addl_files = []

    def get_sumo_command(self, with_bin_file=True, queue_output_file=None,
                         **kwargs):
        if self.routefile is None:
            raise ValueError('Route file not set.')
        sumo_args = [
            '--net-file', self.netfile,
            '--route-files', self.routefile,
            '--no-step-log', # remove progress bar
            '--collision.action', 'none', # don't don't teleport vehicles when they collide
            '--time-to-teleport', '-1', # remove teleporting when vehicles queue for extended periods,
            '--device.rerouting.probability', '.5', # give cars the ability to reroute themselves so queues won't grow unboundedly
            '--device.rerouting.period', '60',
        ]
        if kwargs:
            for key, value in kwargs.items():
                if value is not True:
                    sumo_args.extend([f'--{key}', f'{value}'])
                else:
                    sumo_args.extend([f'--{key}'])

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
            _logger.warning('Seed not set, SUMO seed will be random.')
            sumo_args.extend(['--random', 'true'])

        if self.binfile == 'sumo-gui':
            sumo_args.extend(['--start', '--quit'])

        if with_bin_file:
            return [self.binfile] + sumo_args
        else:
            return sumo_args  # used for passing to traci.load()

    def start(self):
        traci.start(self.get_sumo_command())

    def run(self, return_output=False, **kwargs):
        out = subprocess.check_output(self.get_sumo_command(**kwargs))
        if out is not None and len(out) > 0:
            _logger.info('sumo returned: %s', out)
        elif len(out) == 0:
            _logger.info('sumo completed.')
        if return_output:
            return out

    def sorted_lanes_for_edge(self, edge_id):
        lanes = self.net.getEdge(edge_id).getLanes()
        lanes.sort(key=lambda x: x.getIndex())
        return [lane.getID() for lane in lanes]

    def reset_graph(self, undirected=None, lanewise=None):
        if undirected is not None:
            self.undirected_graph = undirected

        if lanewise is not None:
            self.lanewise = lanewise

        if self.lanewise:
            self.graph = get_lane_graph(
                self.netfile, undirected=self.undirected_graph,
                detector_def_files=self.detector_def_files,
                tls_output_def_files=self.tls_output_def_files)
        else:
            self.graph = get_edge_graph(
                self.netfile, undirected=self.undirected_graph,
                additional_files=self.additional_files)

    def get_graph(self):
        assert self.graph is not None
        return self.graph

    def set_new_graph(self, new_graph):
        self.graph = new_graph

    def get_neighboring_lanes(self, lane_id, include_input_lane=False):
        """Get ids of lanes in the same edge as the passed one.

        :param lane_id: ID of the lane to get the neighbors of.
        :type lane_id: str
        :param include_input_lane: Whether to include the input lane_id in the returned list.
        :type include_input_lane: bool
        :raises TypeError: If lane_id is not str
        :return: list of neighboring lanes
        :rtype: list
        """
        if not isinstance(lane_id, str):
            raise TypeError('Expected str, got %s', type(lane_id))
        sumolib_lane = self.net.getLane(lane_id)
        parent_edge = sumolib_lane.getEdge()
        ids = [lane.getID() for lane in parent_edge.getLanes()]
        if not include_input_lane:
            ids.remove(lane_id)
        return ids

    def get_lane_graph_for_neighboring_lanes(self, include_self_adjacency=True):
        """Return networkx graph with edges connecting lanes in the same road (ie same Sumo `edge').

        :param include_self_adjacency: If True, include A[i,i] = 1.
        :type include_self_adjacency: bool
        """
        if not self.lanewise:
            raise ValueError('Cannot use this method for a non-lanewise graph.')
        graph_copy = self._graph_shallow_copy(no_edges=True)
        for lane in self.graph.nodes:
            neigh_lanes = self.get_neighboring_lanes(
                lane, include_input_lane=include_self_adjacency)
            for neigh in neigh_lanes:
                graph_copy.add_edge(lane, neigh)

        return graph_copy

    def _graph_shallow_copy(self, no_edges=False):
        """Utility function: Get copy of graph without data (nodes/edges only).

        :param no_edges: If True, only copy over the nodes.
        :type no_edges: bool
        """
        copy = self.graph.fresh_copy()
        copy.add_nodes_from(self.graph.nodes())
        if not no_edges:
            copy.add_edges_from(self.graph.edges())
        return copy

    def get_lane_graph_for_conn_type(self,
                                     edge_classes,
                                     edge_class_field='direction'):
        """Return networkx graph with edges for only certain class(es) of connection (e.g., direction: l, r).

        :param edge_classes: List of strings of class(es) requested
        :type edge_classes: list
        :param edge_class_field: Field name in edge attribute dict to reference.
        defaults to 'direction'
        :type edge_class_field: str
        """
        if not self.lanewise:
            raise ValueError('Cannot use this method for a non-lanewise graph.')
        edge_classes = iterfy(edge_classes)
        assert all([isinstance(x, str) for x in edge_classes])

        graph_copy = self._graph_shallow_copy(no_edges=True)
        for in_lane, out_lane, edge_class in self.graph.edges.data(edge_class_field):
            if edge_class in edge_classes:
                graph_copy.add_edge(in_lane, out_lane)

        return graph_copy

    def get_lane_graph_for_thru_movements(self):
        """Gets the networkx graph with edges only for through (straight) movements.

        :return: Adjacency matrix
        :rtype: networkx.DiGraph
        """
        return self.get_lane_graph_for_conn_type(['s'])

    def get_lane_graph_for_turn_movements(self):
        """Gets the networkx graph with edges only for turn movements.

        :return: Adjacency matrix
        :rtype: networkx.DiGraph
        """
        return self.get_lane_graph_for_conn_type(['l', 'r'])

    def get_adjacency_matrix(self, undirected=False):
        """Gets the adjacency matrix for the loaded graph.

        :param undirected: Whether to return the undirected (symmetric) matrix, defaults to False
        :param undirected: bool, optional
        :return: Adjacency matrix
        :rtype: Scipy sparse matrix
        """
        graph = self.get_graph()
        if undirected:
            graph = graph.to_undirected(as_view=True)
        A = nx.adjacency_matrix(graph)
        return A

    def get_config_generator(self):
        config_gen = ConfigGenerator(
            os.path.splitext(
                os.path.splitext(
                    os.path.basename(self.netfile))[0])[0],
            net_config_dir=os.path.dirname(
                os.path.dirname(self.netfile)),
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


def get_lane_graph(netfile,
                   undirected=False,
                   detector_def_files=None,
                   tls_output_def_files=None):
    net = readNet(netfile)

    if undirected:
        graph = nx.Graph()
    else:
        graph = nx.DiGraph()

    for edge in net.getEdges():
        for lane in edge.getLanes():
            graph.add_node(lane.getID(), length=lane.getLength())

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

    # sanity check
    tls_to_edges_2 = {
        tl.getID():
        [tuple([lane.getID() for lane in conn[:-1]]) for conn in tl.getConnections()]
        for tl in net.getTrafficLights()
    }

    assert tls_to_edges == tls_to_edges_2

    if detector_def_files is not None:
        if isinstance(detector_def_files, six.string_types):
            detector_def_files = [detector_def_files]
        for detfile in detector_def_files:
            tree = etree.iterparse(detfile, tag=['e1Detector', 'inductionLoop',
                                                 'e2Detector', 'laneAreaDetector'])
            for _, element in tree:
                lane_id = element.get('lane')
                if lane_id in graph.nodes:
                    if 'detectors' not in graph.node[lane_id]:
                        graph.node[lane_id]['detectors'] = {}

                    detector_info_dict = dict(element.items())
                    detector_info_dict['type'] = element.tag
                    graph.node[lane_id]['detectors'][
                        element.get('id')] = detector_info_dict
                element.clear()

    if tls_output_def_files is not None:
        if isinstance(tls_output_def_files, six.string_types):
            tls_output_def_files = [tls_output_def_files]
        for tlsfile in tls_output_def_files:
            tree = etree.iterparse(tlsfile, tag='timedEvent')

            for _, element in tree:
                if element.get('type') == 'SaveTLSSwitchTimes':
                    tls_id = element.get('source')
                    for edge in tls_to_edges[tls_id]:
                        graph.edges[edge].update(
                            {'tls_output_info': dict(element.items())})
                        in_lane = edge[0]
                        if 'tls_output_info' not in graph.nodes[in_lane]:
                            graph.nodes[in_lane].update(
                                {'tls_output_info': dict(element.items())})
                element.clear()

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


def make_undirected(A):
    no_double_edge = A.T > A
    A_und = A - A.multiply(no_double_edge) + A.T.multiply(no_double_edge)
    assert np.abs(A_und - A_und.T).max() < 1e-10
    return A_und
