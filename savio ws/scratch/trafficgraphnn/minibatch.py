from __future__ import print_function, absolute_import, division

import math
from lxml import etree
import pandas as pd
import numpy as np
import networkx as nx
import six

from keras.utils import Sequence

from trafficgraphnn.utils import load_data

class NodeMinibatcher(Sequence):
    """
    Iterates over nodes for supervised learning.


    """

    def __init__(
        self, graph, filter_path_length, data_files, batch_size=64,
        features=['occupancy', 'flow', 'meanSpeed'],
        targets=['maxJamLengthInMeters'],
        include_back_hops=False
    ):
        if isinstance(data_files, str):
            data_files = [data_files]
        self.graph = graph
        self.filter_path_length = filter_path_length
        self.data_files = data_files
        self.batch_size = batch_size
        self.features = features
        self.targets = targets
        self.include_back_hops = include_back_hops
        if graph.is_directed and include_back_hops:
            self.neigh_range = range(-filter_path_length,
                                     filter_path_length + 1)
        else:
            self.neigh_range = range(filter_path_length + 1)

        self.data = self._build_data_df()

        # output_data = [etree.parse(f) for f in data_files]
        # for node, node_data in graph.nodes.data():
            # node_data['data'] = _build_data_df(node_data, output_data)

        self.timesteps = self.data.index.levels[0]

        self.num_datapoints = self._calc_num_datapoints()

        self.indeces = list(self.data.index.droplevel(2).unique())
        assert self.num_datapoints == len(self.indeces)

    def _build_data_df(self):
        records = {}
        columns_to_get = self.features + self.targets
        output_data = [etree.parse(f) for f in self.data_files]
        for node, node_data in self.graph.nodes.data():
            for det_id in node_data.keys():
                for file_data in output_data:
                    for interval in file_data.xpath(
                        "//interval[@id='{}']".format(det_id)
                    ):
                        records[(
                                int(round(float(interval.attrib['begin']))),
                                node,
                                det_id,
                                # )] = dict(interval.items())
                                )] = {col: interval.attrib[col]
                                      for col in columns_to_get
                                      if col in interval.keys()}

        df = pd.DataFrame.from_dict(records, orient='index', dtype=float)
        df.index.set_names(['time', 'node', 'det_id'], inplace=True)

        return df

    def _get_data_metadata(self):
        parsed_files = [etree.parse(f) for f in self.data_files]
        timesteps = timesteps = np.unique([
            float(interval.attrib['begin'])
            for f in parsed_files
            for interval in f.iterfind('interval')
        ])
        detectors = sorted(set([
            interval.attrib['id']
            for f in parsed_files
            for interval in f.iterfind('interval')
        ]))
        return timesteps, detectors

    def _calc_num_datapoints(self):
        return len(self.timesteps) * self.graph.number_of_nodes()

    def __getitem__(self, index):
        batch_indeces = self.indeces[
            index * self.batch_size:(index + 1) * self.batch_size]

        if self.graph.is_directed():
            neighs = self._directed_neighs(self.graph, batch_indeces)
            if self.include_back_hops:  # add negative-hop neighborhoods
                reverse_neighs = self._reverse_neighs(
                    self.graph, batch_indeces, True)
                for n, steps in six.iteritems(neighs):
                    rn = reverse_neighs[n]
                    for reverse_neighbor, r_steps in six.iteritems(rn):
                        if (
                            reverse_neighbor not in steps
                            or abs(steps[reverse_neighbor]) > abs(r_steps)
                        ):
                            steps[reverse_neighbor] = r_steps
        else:
            neighs = self._undirected_neighs(self.graph, batch_indeces)

        time_sliced = self.data.iloc[
            self.data.index.get_level_values('time').isin(
                [i[0] for i in batch_indeces])
        ]

        # done to remove nans if two detector types in one node
        meaned_over_detector = time_sliced.groupby(
            level=['time', 'node'], sort=False).mean()

        batch_x = []
        concatted_y = np.zeros((len(batch_indeces), len(self.targets)))
        for i, index in enumerate(batch_indeces):
            time = index[0]
            neigh = neighs[index[1]]
            neigh_sizes = [neigh.values().count(k) for k in self.neigh_range]
            node_x = np.zeros((
                len(neigh_sizes),
                max(neigh_sizes),
                len(self.features)))

            index_for_neigh = np.zeros(node_x.shape[0], dtype=int)
            for node, k in six.iteritems(neigh):
                node_x[k, index_for_neigh[k]] = meaned_over_detector.loc[
                    (time, node), self.features]
                index_for_neigh[k] += 1

            concatted_y[i] = meaned_over_detector.loc[
                (time, node), self.targets]

            batch_x.append(node_x)

        max_neigh_size = max([x.shape[1] for x in batch_x])

        concatted_x = np.stack(
            [np.pad(
                x,
                [(0, 0), (0, max_neigh_size - x.shape[1]), (0, 0)],
                'constant',
                constant_values=0) for x in batch_x], axis=0
        )

        return concatted_x, concatted_y

    def _directed_neighs(self, graph, indeces):
        directed_neigh = {
            entry[1]: nx.single_source_dijkstra_path_length(
                graph, entry[1], self.filter_path_length)
            for entry in indeces
        }
        return directed_neigh

    def _reverse_neighs(self, graph, indeces, return_negative=True):
        reverse_graph = graph.reverse()
        reverse_neighs = self._directed_neighs(reverse_graph, indeces)
        if return_negative:
            for neigh in six.itervalues(reverse_neighs):
                for k, v in six.iteritems(neigh):
                    neigh[k] = -v
        return reverse_neighs

    def _undirected_neighs(self, graph, indeces):
        undirected_neigh = {
            entry[1]: nx.single_source_shortest_path_length(
                graph, entry[1], self.filter_path_length)
            for entry in indeces
        }
        return undirected_neigh

    def __len__(self):
        return math.ceil(self.num_datapoints / self.batch_size)

    def on_epoch_end(self):
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.indeces)

# def _build_data_df(node_data, output_data):
#     records = {}
#     for det_id in node_data.keys():
#         for fd in output_data:
#             for interval in fd.xpath("//interval[@id='{}']".format(det_id)):
#                 records[
#                     (det_id, int(interval.attrib['begin']))
#                 ] = dict(interval.items())

#     return pd.DataFrame.from_dict(records, orient='index')
