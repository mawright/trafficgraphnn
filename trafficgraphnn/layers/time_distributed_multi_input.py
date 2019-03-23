from __future__ import absolute_import, division, print_function

from six.moves import zip

from keras import backend as K
from keras.engine.base_layer import InputSpec
from keras.layers.wrappers import TimeDistributed, Wrapper
from keras.utils.generic_utils import has_arg, object_list_uid


class TimeDistributedMultiInput(TimeDistributed):
    def __init__(self, layer, **kwargs):
        super(TimeDistributedMultiInput, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        # If a singleton shape tuple is passed in, use the vanilla TimeDistributed
        if not isinstance(input_shape, list):
            super(TimeDistributedMultiInput, self).build(input_shape)
            return

        assert all([len(shape) >= 3 for shape in input_shape if shape is not None])
        # We need to verify that the inputs have the same or broadcastable batch
        # and time dimensions
        batch_sizes = [shape[0] for shape in input_shape if shape is not None]
        batch_sizes = set(batch_sizes)
        batch_sizes -= set([None])
        # Allow batch size of 1 if an input is to be broadcasted
        batch_sizes -= set([1])

        if len(batch_sizes) > 1:
            raise ValueError('Receieved tensors with incompatible batch sizes. '
                             'Got tensors with shapes : ' + str(input_shape))
        timesteps = [shape[1] for shape in input_shape if shape is not None]
        timesteps = set(timesteps)
        timesteps -= set([None])
        # Allow 1 timestep if an input is to be broadcasted
        timesteps -= set([1])

        if len(timesteps) > 1:
            raise ValueError('Receieved tensors with incompatible number of timesteps. '
                             'Got tensors with shapes : ' + str(input_shape))
        self.timesteps = timesteps.pop() if len(timesteps) == 1 else None
        self.input_spec = [InputSpec(shape=s) for s in input_shape]
        child_input_shapes = [(shape[0],) + shape[2:] for shape in input_shape]
        if not self.layer.built:
            self.layer.build(child_input_shapes)
            self.layer.built = True

        Wrapper.build(self)

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            return super(
                TimeDistributedMultiInput, self).compute_output_shape(
                    input_shape)

        child_input_shapes = [(shape[0],) + shape[2:] for shape in input_shape]
        child_output_shape = self.layer.compute_output_shape(child_input_shapes)
        if hasattr(self, 'timesteps'):
            timesteps = self.timesteps
        else:
            timesteps = input_shape[0][1]
        return (child_output_shape[0], timesteps) + child_output_shape[1:]

    def call(self, inputs, training=None, mask=None):
        if not isinstance(inputs, list):
            return super(TimeDistributedMultiInput, self).call(inputs,
                                                               training=training,
                                                               mask=mask)

        kwargs = {}
        if has_arg(self.layer.call, 'training'):
            kwargs['training'] = training
        uses_learning_phase = False

        input_shapes = [K.int_shape(inp) for inp in inputs]
        # batch_sizes = [shape[0] for shape in input_shapes if shape is not None]
        # fixed_batch_size = any([bs is not None for bs in batch_sizes])
        fixed_batch_size = False
        if fixed_batch_size:
            raise NotImplementedError('K.rnn does not support multiple inputs.')
            # batch size matters, use rnn-based implementation
            def step(x, _):
                global uses_learning_phase
                output = self.layer.call(x, **kwargs)
                if hasattr(output, '_uses_learning_phase'):
                    uses_learning_phase = (output._uses_learning_phase or
                                           uses_learning_phase)
                return output, []

            # Note: will likely fail here if K.rnn doesn't like multiple inputs
            _, outputs, _ = K.rnn(step, inputs,
                                  initial_states=[],
                                  input_length=input_shapes[1],
                                  unroll=False)
            y = outputs
        else:
            # No batch size specified, therefore the layer will be able
            # to process batches of any size.
            # We can go with reshape-based implementation for performance.

            input_length = self.timesteps if self.timesteps else K.shape(inputs[0])[1]
            # ^^ assumes input 0 has the correct number of timesteps
            def prep_input(inp):
                inner_input_shape = self._get_shape_tuple((-1,), inp, 2)
                # Shape: (num_samples * timesteps, ...). And track the
                # transformation in self._input_map.
                input_uid = object_list_uid(inp)
                reshaped = K.reshape(inp, inner_input_shape)
                self._input_map[input_uid] = reshaped
                return reshaped
            inputs = [prep_input(inp) for inp in inputs]
            # (num_samples * timesteps, ...)
            if has_arg(self.layer.call, 'mask') and mask is not None:
                inner_mask_shape = self._get_shape_tuple((-1,), mask, 2)
                kwargs['mask'] = K.reshape(mask, inner_mask_shape)
            y = self.layer.call(inputs, **kwargs)

            if isinstance(y, list):
                raise NotImplementedError(
                    'TimeDistributedMultiInput not implemented for multiple '
                    'output tensors yet.')

            if hasattr(y, '_uses_learning_phase'):
                uses_learning_phase = y._uses_learning_phase
            # Shape: (num_samples, timesteps, ...)
            output_shape = self.compute_output_shape(input_shapes)
            output_shape = self._get_shape_tuple(
                (-1, input_length), y, 1, output_shape[2:])
            y = K.reshape(y, output_shape)

        # Apply activity regularizer if any:
        if (hasattr(self.layer, 'activity_regularizer') and
           self.layer.activity_regularizer is not None):
            regularization_loss = self.layer.activity_regularizer(y)
            self.add_loss(regularization_loss, inputs)

        if uses_learning_phase:
            y._uses_learning_phase = True
        return y

    def compute_mask(self, inputs, mask=None):
        """Masks not supported for this layer (yet)."""
        return None
