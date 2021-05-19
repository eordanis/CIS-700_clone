import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


class Discriminator():
    def __init__(self, embedding_size, vocab_size, non_static, hidden_unit, sequence_length, batch_size,
                 num_classes, filter_sizes, num_filters, l2_reg_lambda=0.0, start_token=0):
        embedding_size = vocab_size
        self.num_vocabulary = vocab_size
        self.emb_dim = embedding_size
        self.hidden_dim = hidden_unit
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.batch_size = tf.compat.v1.constant(value=[batch_size])
        self.batch_size_scale = batch_size
        self.hidden_unit = hidden_unit
        self.d_params = []
        l2_loss = tf.compat.v1.constant(0.0)
        self.start_token = tf.compat.v1.constant([start_token] * batch_size, dtype=tf.compat.v1.int32)

        with tf.compat.v1.variable_scope('discriminator'):
            self.g_recurrent_unit = self.create_recurrent_unit(self.d_params)  # maps h_tm1 to h_t for generator
            self.g_output_unit = self.create_output_unit(self.d_params)  # maps h_t to o_t (output token logits)

        self.input_x = tf.compat.v1.placeholder(tf.compat.v1.float32, [batch_size, sequence_length, vocab_size], name='input_x')
        self.input_y = tf.compat.v1.placeholder(tf.compat.v1.float32, [batch_size, num_classes], name='input_y')
        self.one_hot = tf.compat.v1.constant(np.eye(vocab_size), dtype=tf.compat.v1.float32)
        self.h_0 = tf.compat.v1.constant(value=0, dtype=tf.compat.v1.float32, shape=[batch_size, hidden_unit])
        self.c_0 = tf.compat.v1.constant(value=0, dtype=tf.compat.v1.float32, shape=[batch_size, hidden_unit])
        self.h0 = tf.compat.v1.stack([self.h_0, self.c_0])

        score = self.predict(input_x=self.input_x)
        self.score = score

        with tf.compat.v1.name_scope('Dloss'):
            pred_loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits=score, labels=self.input_y)
            # todo reg loss
            reg_loss = 0
            self.loss = tf.compat.v1.reduce_mean(pred_loss)

        self.params = [param for param in tf.compat.v1.trainable_variables() if 'discriminator' in param.name]
        d_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
        self.grad_clip = 5.0
        self.pretrain_grad, _ = tf.compat.v1.clip_by_global_norm(tf.compat.v1.gradients(self.loss, self.params), self.grad_clip)
        self.train_op = d_optimizer.apply_gradients(zip(self.pretrain_grad, self.params))
        return

    def predict(self, input_x, h_0=None):
        if h_0 is None:
            h_0 = self.h_0
        def _g_recurrence(i, x_t, h_tm1, o_t):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            o_t = self.g_output_unit(h_t)  # batch x vocab , logits not prob
            x_tp1 = tf.compat.v1.squeeze(tf.compat.v1.slice(input_x, begin=[0, i, 0], size=[self.batch_size_scale, 1, self.num_vocabulary]))
            return i + 1, x_tp1, h_t, o_t

        o_0 = tf.compat.v1.constant(np.zeros(shape=[self.batch_size_scale, self.num_classes]))
        o_0 = tf.compat.v1.cast(o_0, dtype=tf.compat.v1.float32)
        _, _, h_t, output = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.compat.v1.constant(0, dtype=tf.compat.v1.int32),
                       tf.compat.v1.nn.embedding_lookup(self.one_hot, self.start_token), self.h0, o_0))

        return output

    def init_matrix(self, shape):
        return tf.compat.v1.random_normal(shape, stddev=0.1)

    def create_recurrent_unit(self, params):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.compat.v1.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Ui = tf.compat.v1.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bi = tf.compat.v1.Variable(self.init_matrix([self.hidden_dim]))

        self.Wf = tf.compat.v1.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uf = tf.compat.v1.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bf = tf.compat.v1.Variable(self.init_matrix([self.hidden_dim]))

        self.Wog = tf.compat.v1.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uog = tf.compat.v1.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bog = tf.compat.v1.Variable(self.init_matrix([self.hidden_dim]))

        self.Wc = tf.compat.v1.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uc = tf.compat.v1.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bc = tf.compat.v1.Variable(self.init_matrix([self.hidden_dim]))
        params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.compat.v1.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.compat.v1.sigmoid(
                tf.compat.v1.matmul(x, self.Wi) +
                tf.compat.v1.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.compat.v1.sigmoid(
                tf.compat.v1.matmul(x, self.Wf) +
                tf.compat.v1.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.compat.v1.sigmoid(
                tf.compat.v1.matmul(x, self.Wog) +
                tf.compat.v1.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.compat.v1.nn.tanh(
                tf.compat.v1.matmul(x, self.Wc) +
                tf.compat.v1.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.compat.v1.nn.tanh(c)

            return tf.compat.v1.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self, params):
        self.Wo = tf.compat.v1.Variable(self.init_matrix([self.hidden_dim, self.num_classes]))
        self.bo = tf.compat.v1.Variable(self.init_matrix([self.num_classes]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.compat.v1.unstack(hidden_memory_tuple)
            logits = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(hidden_state, self.Wo) + self.bo)
            return logits

        return unit
