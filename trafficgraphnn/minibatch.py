import math
from lxml import etree
import pandas as pd
import numpy as np
import networkx as nx

from keras.utils import Sequence


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
                                )] = {col: interval.attrib[col]
                                      for col in columns_to_get
                                      if col in interval.keys()}
                                # )] = dict(interval.items())

        df = pd.DataFrame.from_dict(records, orient='index', dtype=float)
        df.index.set_names(['time', 'node', 'det_id'], inplace=True)

        return df

    def _calc_num_datapoints(self):
        return len(self.timesteps) * self.graph.number_of_nodes()

    def __getitem__(self, index):
        batch_indeces = self.indeces[
            index * self.batch_size:(index + 1) * self.batch_size]

        if self.graph.is_directed:
            neighs = [self._directed_neighs(self.graph, batch_indeces)]
            if self.include_back_hops:
                neighs.append(self._reverse_neighs(self.graph, batch_indeces))
        else:
            neighs = [self._undirected_neighs(self.graph, batch_indeces)]

        time_sliced = self.data.iloc[
            self.data.index.get_level_values('time').isin(
                [i[0] for i in batch_indeces])
        ]

        meaned_over_detector = time_sliced.groupby(
            level=['time', 'node'], sort=False).mean()

        batch_x = []
        concatted_y = np.zeros((len(batch_indeces), len(self.targets)))
        for index in batch_indeces:
            time = index[0]
            neigh = neighs[index[1]]
            neigh_sizes = [neigh.values().count(k)
                           for k in range(self.filter_path_length + 1)]
            node_x = np.zeros((
                self.filter_path_length + 1,
                max(neigh_sizes),
                len(self.features)))

            index_for_neigh = np.zeros(node_x.shape[0], dtype=int)
            for node, k in neigh.items():
                node_x[k, index_for_neigh[k]] = meaned_over_detector.loc[
                    (time, node), self.features]
                index_for_neigh[k] += 1

            concatted_y[index] = meaned_over_detector.loc[
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

    def _reverse_neighs(self, graph, indeces):
        reverse_graph = graph.reverse()
        reverse_neigh = self._directed_neighs(reverse_graph, indeces)
        return reverse_neigh

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
