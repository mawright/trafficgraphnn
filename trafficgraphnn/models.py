from trafficgraphnn import sumo_network
from trafficgraphnn.graph_layers import LocalGraphLayer
from trafficgraphnn.minibatch import NodeMinibatcher
from keras.layers import Input, Dense
from keras.models import Model


class SupervisedQueueLearner(object):
    def __init__(
        self, network_file, addl_files, output_files, lanewise=True,
        features=['occupancy', 'flow', 'meanSpeed'],
        use_undirected_graph=False, num_filters=32,
        filter_path_length=2, include_back_hops=False,
        fc_layer_sizes=[128, 64], batch_size=64,
        validation_output_files=None
    ):
        if lanewise:
            self.graph = sumo_network.get_lane_graph(
                network_file, undirected=use_undirected_graph,
                detector_files=addl_files)
        else:
            self.graph = sumo_network.get_edge_graph(
                network_file, undirected=use_undirected_graph,
                detector_files=addl_files)

        self.batcher = NodeMinibatcher(
            self.graph, filter_path_length, data_files=output_files,
            batch_size=batch_size, include_back_hops=True)

        if validation_output_files is not None:
            self.validation_batcher = NodeMinibatcher(
                self.graph, filter_path_length, validation_output_files,
                batch_size=batch_size, include_back_hops=True)
        else:
            self.validation_batcher = None

        num_neighborhoods = len(self.batcher.neigh_range)

        input = Input((num_neighborhoods, None, len(self.batcher.features)))

        x = LocalGraphLayer(
            num_filters, filter_path_length, include_back_hops=True)(input)

        for units in fc_layer_sizes:
            x = Dense(units, activation='relu')(x)

        output = Dense(1)(x)

        self.model = Model(inputs=input, outputs=output)

    def train(self):
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        self.model.fit_generator(
            self.batcher, steps_per_epoch=len(self.batcher), epochs=3,
            validation_data=self.validation_batcher,
            validation_steps=20)
