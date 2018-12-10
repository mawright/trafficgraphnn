import tensorflow as tf

from trafficgraphnn.load_data import batches_from_directories


class TFBatcher(object):
    def __init__(self,
                 directories,
                 batch_size,
                 window_size,
                 shuffle=True,
                 A_name_list=['A_downstream',
                              'A_upstream',
                              'A_neighbors'],
                 x_feature_subset=['e1_0/occupancy',
                                   'e1_0/speed',
                                   'e1_1/occupancy',
                                   'e1_1/speed',
                                   'liu_estimated',
                                   'green'],
                 y_feature_subset=['e2_0/nVehSeen',
                                   'e2_0/maxJamLengthInMeters']):

        batches = list(batches_from_directories(
            directories,
            batch_size,
            window_size,
            shuffle=shuffle,
            A_name_list=A_name_list,
            x_feature_subset=x_feature_subset,
            y_feature_subset=y_feature_subset))

        datasets = [tf.data.Dataset.from_generator(
            batch.iterate, (tf.bool, tf.float32, tf.float32),
            output_shapes=(
                (None, None, None, None, None), # A: batch x time x depth x lane x lane
                (None, None, None, len(x_feature_subset)), # X: batch x time x lane x feat
                (None, None, None, len(y_feature_subset)) # Y: batch x time x lane x feat
            ))
            for batch in batches]

        self.datasets = [ds.prefetch(1) for ds in datasets]

        self.iterator = tf.data.Iterator.from_structure(
            self.datasets[0].output_types, self.datasets[0].output_shapes)

        self.init_ops = [self.iterator.make_initializer(ds) for ds in datasets]
        self.tensor = self.iterator.get_next()

    def init_batch(self, session, batch_num):
        session.run(self.init_ops[batch_num])

    @property
    def num_batches(self):
        return len(self.datasets)
