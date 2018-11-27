from __future__ import print_function

import heapq
import os

import numpy as np

from keras import backend as K
from keras import constraints, initializers, regularizers
from keras.engine.base_layer import _collect_previous_mask
from keras.layers import (GRU, RNN, Bidirectional, Dense, Embedding, GRUCell,
                          Input, InputSpec, Lambda, Layer, TimeDistributed,
                          concatenate)
from keras.models import Model
from keras.optimizers import Adadelta
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.generic_utils import has_arg
from trafficgraphnn.layers.andhus_attention import AttentionCellWrapper


class DenseCausalAttention(AttentionCellWrapper):
    def __init__(self, cell,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DenseCausalAttention, self).__init__(cell, **kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def attention_call(self,
                       inputs,
                       cell_states,
                       attended,
                       attention_states,
                       attended_mask,
                       training=None):
        # only one attended sequence (verified in build)
        [attended, u] = attended
        # attended: hidden states of encoder (h)
        # u: dot product of encoder state and attention weight matrix Ua
        attended_mask = attended_mask[0]
        h_cell_tm1 = cell_states[0]
        tm1 = attention_states[1]

        attended_shape = K.shape(attended)
        length = attended_shape[1]

        timesteps = K.arange(length)
        timesteps = K.expand_dims(timesteps, 0)

        causal_mask = K.less_equal(timesteps, K.cast(tm1, 'int32'))
        if attended_mask is None:
            attended_mask = causal_mask
        else:
            attended_mask = K.minimum(K.cast(attended_mask, 'int32'),
                                      K.cast(causal_mask, 'int32'))

        # compute attention weights
        w = K.repeat(K.dot(h_cell_tm1, self.W_a) + self.b_UW, length)
        e = K.exp(K.dot(K.tanh(w + u), self.v_a) + self.b_v)

        if attended_mask is not None:
            e = e * K.cast(K.expand_dims(attended_mask, -1), K.dtype(e))

        # weighted average of attended
        a = e / K.sum(e, axis=1, keepdims=True)
        c = K.sum(a * attended, axis=1, keepdims=False)

        # timestep
        t = tm1 + 1

        return c, [c, t]

    def attention_build(self, input_shape, cell_state_size, attended_shape):
        if not len(attended_shape) == 2:
            raise ValueError('There must be two attended tensors')
        for a in attended_shape:
            if not len(a) == 3:
                raise ValueError('only support attending tensors with dim=3')
        [attended_shape, u_shape] = attended_shape

        # NOTE _attention_size must always be set in `attention_build`
        self._attention_size = attended_shape[-1]
        units = u_shape[-1]

        kernel_kwargs = dict(initializer=self.kernel_initializer,
                             regularizer=self.kernel_regularizer,
                             constraint=self.kernel_constraint)
        self.W_a = self.add_weight(shape=(cell_state_size[0], units),
                                   name='W_a', **kernel_kwargs)
        self.v_a = self.add_weight(shape=(units, 1),
                                   name='v_a', **kernel_kwargs)

        bias_kwargs = dict(initializer=self.bias_initializer,
                           regularizer=self.bias_regularizer,
                           constraint=self.bias_constraint)
        self.b_UW = self.add_weight(shape=(units,),
                                    name="b_UW", **bias_kwargs)
        self.b_v = self.add_weight(shape=(1,),
                                   name="b_v", **bias_kwargs)

    def get_config(self):
        config = {
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(DenseCausalAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @property
    def attention_state_size(self):
        return [self.attention_size, 1]


if __name__ == '__main__':
    DATA_DIR = 'data/wmt16_mmt'
    FROM_LANGUAGE = 'en'
    TO_LANGUAGE = 'de'

    # Meta parameters
    MAX_UNIQUE_WORDS = 30000
    MAX_WORDS_PER_SENTENCE = 40  # inf in [1]
    EMBEDDING_SIZE = 620  # `m` in [1]
    RECURRENT_UNITS = 1000  # `n` in [1]
    DENSE_ATTENTION_UNITS = 1000  # fixed equal to `n` in [1]
    READOUT_HIDDEN_UNITS = 500  # `l` in [1]
    OPTIMIZER = Adadelta(rho=0.95, epsilon=1e-6)
    BATCH_SIZE = 80
    EPOCHS = 20

    # Load and tokenize the data
    start_token = "'start'"
    end_token = "'end'"
    # NOTE: using single quotes (which are not dropped by Tokenizer by default)
    # for the tokens to be distinguished from other use of "start" and "end"

    def get_sentences(partion, language):
        fpath = os.path.join(DATA_DIR, partion + '.' + language)
        with open(fpath, 'r') as f:
            sentences = f.readlines()
        return ["{} {} {}".format(start_token, sentence, end_token)
                for sentence in sentences]

    input_texts_train = get_sentences("train", FROM_LANGUAGE)
    input_texts_val = get_sentences("val", FROM_LANGUAGE)
    target_texts_train = get_sentences("train", TO_LANGUAGE)
    target_texts_val = get_sentences("val", TO_LANGUAGE)

    input_tokenizer = Tokenizer(num_words=MAX_UNIQUE_WORDS)
    target_tokenizer = Tokenizer(num_words=MAX_UNIQUE_WORDS)
    input_tokenizer.fit_on_texts(input_texts_train + input_texts_val)
    target_tokenizer.fit_on_texts(target_texts_train + target_texts_val)
    input_max_word_idx = max(input_tokenizer.word_index.values())
    target_max_word_idx = max(target_tokenizer.word_index.values())

    input_seqs_train = input_tokenizer.texts_to_sequences(input_texts_train)
    input_seqs_val = input_tokenizer.texts_to_sequences(input_texts_val)
    target_seqs_train = target_tokenizer.texts_to_sequences(target_texts_train)
    target_seqs_val = target_tokenizer.texts_to_sequences(target_texts_val)

    input_seqs_train, input_seqs_val, target_seqs_train, target_seqs_val = (
        pad_sequences(seq, maxlen=MAX_WORDS_PER_SENTENCE, padding='post')
        for seq in [input_seqs_train,
                    input_seqs_val,
                    target_seqs_train,
                    target_seqs_val])

    # Build the model
    x = Input((None,), name="input_sequences")
    y = Input((None,), name="target_sequences")
    x_emb = Embedding(input_max_word_idx + 1, EMBEDDING_SIZE, mask_zero=True)(x)
    y_emb = Embedding(target_max_word_idx + 1, EMBEDDING_SIZE, mask_zero=True)(y)

    encoder_rnn = Bidirectional(GRU(RECURRENT_UNITS,
                                    return_sequences=True,
                                    return_state=True))
    x_enc, h_enc_fwd_final, h_enc_bkw_final = encoder_rnn(x_emb)

    # the final state of the backward-GRU (closest to the start of the input
    # sentence) is used to initialize the state of the decoder
    initial_state_gru = Dense(RECURRENT_UNITS, activation='tanh')(h_enc_bkw_final)
    initial_attention_h = Lambda(lambda x: K.zeros_like(x)[:, 0, :])(x_enc)
    initial_state = [initial_state_gru, initial_attention_h]

    cell = DenseCausalAttention(cell=GRUCell(RECURRENT_UNITS),
                                    input_mode="concatenate",
                                    output_mode="cell_output")
    # TODO output_mode="concatenate", see TODO(3)/A
    decoder_rnn = RNN(cell=cell, return_sequences=True, return_state=True)
    u = TimeDistributed(Dense(DENSE_ATTENTION_UNITS, use_bias=False))(x_enc)
    h1_and_state = decoder_rnn(y_emb,
                               initial_state=initial_state,
                               constants=[x_enc, u])
    h1 = h1_and_state[0]

    def dense_maxout(x_):
        """Implements a dense maxout layer where max is taken
        over _two_ units"""
        x_ = Dense(READOUT_HIDDEN_UNITS * 2)(x_)
        x_1 = x_[:, :READOUT_HIDDEN_UNITS]
        x_2 = x_[:, READOUT_HIDDEN_UNITS:]
        return K.max(K.stack([x_1, x_2], axis=-1), axis=-1, keepdims=False)

    maxout_layer = TimeDistributed(Lambda(dense_maxout))
    h2 = maxout_layer(concatenate([h1, y_emb]))

    output_layer = TimeDistributed(Dense(target_max_word_idx + 1,
                                         activation='softmax'))
    y_pred = output_layer(h2)

    model = Model([y, x], y_pred)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=OPTIMIZER)

    # Run training
    model.fit([target_seqs_train[:, :-1], input_seqs_train],
              target_seqs_train[:, 1:, None],
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(
                  [target_seqs_val[:, :-1], input_seqs_val],
                  target_seqs_val[:, 1:, None]))

    # Save model
    model.save('rec_att_mt.h5')

    # Inference
    # Let's use the model to translate new sentences! To do this efficiently, two
    # things must be done in preparation:
    #  1) Build separate model for the encoding that is only done _once_ per input
    #     sequence.
    #  2) Build a model for the decoder (and output layers) that takes input states
    #     and returns updated states for the recurrent part of the model, so that
    #     it can be run one step at a time.
    encoder_model = Model(x, [x_enc] + initial_state)

    x_enc_new = Input(batch_shape=K.int_shape(x_enc))
    initial_state_new = [Input((size,)) for size in cell.state_size]
    h1_and_state_new = decoder_rnn(y_emb,
                                   initial_state=initial_state_new,
                                   constants=x_enc_new)
    h1_new = h1_and_state_new[0]
    updated_state = h1_and_state_new[1:]
    h2_new = maxout_layer(concatenate([h1_new, y_emb]))
    y_pred_new = output_layer(h2_new)
    decoder_model = Model([y, x_enc_new] + initial_state_new,
                          [y_pred_new] + updated_state)

    def translate_greedy(input_text, t_max=None):
        """Takes the most probable next token at each time step until the end-token
        is predicted or t_max reached.
        """
        t = 0
        y_t = np.array(target_tokenizer.texts_to_sequences([start_token]))
        y_0_to_t = [y_t]
        x_ = np.array(input_tokenizer.texts_to_sequences([input_text]))
        encoder_output = encoder_model.predict(x_)
        x_enc_ = encoder_output[0]
        state_t = encoder_output[1:]
        if t_max is None:
            t_max = x_.shape[-1] * 2
        end_idx = target_tokenizer.word_index[end_token]
        score = 0  # track the cumulative log likelihood
        while y_t[0, 0] != end_idx and t < t_max:
            t += 1
            decoder_output = decoder_model.predict([y_t, x_enc_] + state_t)
            y_pred_ = decoder_output[0]
            state_t = decoder_output[1:]
            y_t = np.argmax(y_pred_, axis=-1)
            score += np.log(y_pred_[0, 0, y_t[0, 0]])
            y_0_to_t.append(y_t)
        y_ = np.hstack(y_0_to_t)
        output_text = target_tokenizer.sequences_to_texts(y_)[0]
        # length normalised score, skipping start token
        score = score / (len(y_0_to_t) - 1)

        return output_text, score

    def translate_beam_search(input_text,
                              search_width=20,
                              branch_factor=None,
                              t_max=None):
        """Perform beam search to approximately find the translated sentence that
        maximises the conditional probability given the input sequence.

        Returns the completed sentences (reached end-token) in order of decreasing
        score (the first is most probable) followed by incomplete sentences in order
        of decreasing score - as well as the score for the respective sentence.

        References:
            [1] "Sequence to sequence learning with neural networks"
            (https://arxiv.org/pdf/1409.3215.pdf)
        """

        if branch_factor is None:
            branch_factor = search_width
        elif branch_factor > search_width:
            raise ValueError("branch_factor must be smaller than search_width")
        elif branch_factor < 2:
            raise ValueError("branch_factor must be >= 2")

        def k_largest_val_idx(a, k):
            """Returns top k largest values of a and their indices, ordered by
            decreasing value"""
            top_k = np.argpartition(a, -k)[-k:]
            return sorted(zip(a[top_k], top_k))[::-1]

        # initialisation of search
        t = 0
        y_0 = np.array(target_tokenizer.texts_to_sequences([start_token]))[0]
        end_idx = target_tokenizer.word_index[end_token]

        # run input encoding once
        x_ = np.array(input_tokenizer.texts_to_sequences([input_text]))
        encoder_output = encoder_model.predict(x_)
        x_enc_ = encoder_output[0]
        state_t = encoder_output[1:]
        # repeat to a batch of <search_width> samples
        x_enc_ = np.repeat(x_enc_, search_width, axis=0)

        if t_max is None:
            t_max = x_.shape[-1] * 2

        # A "search beam" is represented as the tuple:
        #   (score, outputs, state)
        # where:
        #   score: the average log likelihood of the output tokens
        #   outputs: the history of output tokens up to time t, [y_0, ..., y_t]
        #   state: the most recent state of the decoder_rnn for this beam

        # A list of the <search_width> number of beams with highest score is
        # maintained through out the search. Initially there is only one beam.
        incomplete_beams = [(0., [y_0], [s[0] for s in state_t])]
        # All beams that reached the end-token are kept separately.
        complete_beams = []

        while len(complete_beams) < search_width and t < t_max:
            t += 1
            # create a batch of inputs representing the incomplete_beams
            y_tm1 = np.vstack([beam[1][-1] for beam in incomplete_beams])
            state_tm1 = [
                np.vstack([beam[2][i] for beam in incomplete_beams])
                for i in range(len(state_t))
            ]

            # predict next tokes for every incomplete beam
            batch_size = len(incomplete_beams)
            decoder_output = decoder_model.predict(
                [y_tm1, x_enc_[:batch_size]] + state_tm1)
            y_pred_ = decoder_output[0]
            state_t = decoder_output[1:]
            # from each previous beam create new candidate beams and save the once
            # with highest score for next iteration.
            beams_updated = []
            for i, beam in enumerate(incomplete_beams):
                l = len(beam[1]) - 1  # don't count 'start' token
                for proba, idx in k_largest_val_idx(y_pred_[i, 0], branch_factor):
                    new_score = (beam[0] * l + np.log(proba)) / (l + 1)
                    not_full = len(beams_updated) < search_width
                    ended = idx == end_idx
                    if not_full or ended or new_score > beams_updated[0][0]:
                        # create new successor beam with next token=idx
                        beam_new = (new_score,
                                    beam[1] + [np.array([idx])],
                                    [s[i] for s in state_t])
                        if ended:
                            complete_beams.append(beam_new)
                        elif not_full:
                            heapq.heappush(beams_updated, beam_new)
                        else:
                            heapq.heapreplace(beams_updated, beam_new)
                    else:
                        # if score is not among to candidates we abort search
                        # for this ancestor beam (next token processed in order of
                        # decreasing likelihood)
                        break
            # faster to process beams in order of decreasing score next iteration,
            # due to break above
            incomplete_beams = sorted(beams_updated, reverse=True)

        # want to return in order of decreasing score
        complete_beams = sorted(complete_beams, reverse=True)

        output_texts = []
        scores = []
        for beam in complete_beams + incomplete_beams:
            output_texts.append(target_tokenizer.sequences_to_texts(
                np.concatenate(beam[1])[None, :])[0])
            scores.append(beam[0])

        return output_texts, scores

    # Translate one of sentences from validation data
    input_text = input_texts_val[0]
    print("Translating:\n", input_text)
    output_greedy, score_greedy = translate_greedy(input_text)
    print("Greedy output:\n", output_greedy)
    outputs_beam, scores_beam = translate_beam_search(input_text)
    print("Beam search output:\n", outputs_beam[0])
