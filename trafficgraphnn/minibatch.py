import math
from lxml import etree
import pandas as pd

from keras.utils import Sequence


class NodeMinibatcher(Sequence):
    """
    Iterates over nodes for supervised learning.


    """
    def __init__(
        self, graph, filter_path_length, data_files, batch_size=64,
        features=['occupancy', 'flow', 'meanSpeed'],
        targets=['maxJamLengthInMeters']
    ):
        if isinstance(data_files, str):
            data_files = [data_files]
        self.graph = graph
        self.filter_path_length = filter_path_length
        self.data_files = data_files
        self.features = features
        self.targets = targets

        self.data = self._build_data_df()

        # output_data = [etree.parse(f) for f in data_files]
        # for node, node_data in graph.nodes.data():
            # node_data['data'] = _build_data_df(node_data, output_data)

        self.timesteps = self.data.index.levels[1]

        self.num_datapoints = self._calc_num_datapoints()

    def _build_data_df(self):
        records = {}
        output_data = [etree.parse(f) for f in self.data_files]
        for node, node_data in self.graph.nodes.data():
            for det_id in node_data.keys():
                for file_data in output_data:
                    for interval in file_data.xpath(
                        "//interval[@id='{}']".format(det_id)
                    ):
                        records[
                            (det_id,
                             int(round(float(interval.attrib['begin'])))
                             )
                        ] = dict(interval.items())

        return pd.DataFrame.from_dict(records, orient='index')

    def _calc_num_datapoints(self):
        return len(self.timesteps) * self.graph.number_of_nodes()


    def __getitem__(self, index):
        pass

    def __len__(self):
        return math.ceil(x)

    def on_epoch_end(self):
        self.shuffle()

    def shuffle():
        pass


def _build_data_df(node_data, output_data):
    records = {}
    for det_id in node_data.keys():
        for fd in output_data:
            for interval in fd.xpath("//interval[@id='{}']".format(det_id)):
                records[
                    (det_id, int(interval.attrib['begin']))
                ] = dict(interval.items())

    return pd.DataFrame.from_dict(records, orient='index')
