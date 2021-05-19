import tensorflow as tf
import numpy as np
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


# An alternative to tf.compat.v1.nn.rnn_cell._linear function, which has been removed in Tensorfow 1.0.1
# The highway layer is borrowed from https://github.com/mkroutikov/tf-lstm-char-cnn
def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.compat.v1.variable_scope(scope or "SimpleLinear", reuse=tf.compat.v1.AUTO_REUSE):
        matrix = tf.compat.v1.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.compat.v1.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.compat.v1.matmul(input_, tf.compat.v1.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.compat.v1.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.compat.v1.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


def cosine_distance(y_s, y_u, gamma=1.0):
    return gamma * tf.compat.v1.reduce_sum(y_s * y_u) / (tf.compat.v1.norm(y_s) * tf.compat.v1.norm(y_u))


def get_rank_score(emb_test, embs_ref):
    p = embs_ref.shape
    ref_size = p.as_list()[0]

    def _loop_body(i, ret_v, emb_test, embs_ref):
        return i + 1, ret_v + cosine_distance(emb_test, tf.compat.v1.nn.embedding_lookup(embs_ref, i)), emb_test, embs_ref

    _, ret, _, _ = control_flow_ops.while_loop(
        cond=lambda i, _1, _2, _3: i < ref_size,
        body=_loop_body,
        loop_vars=(tf.compat.v1.constant(0, dtype=tf.compat.v1.int32), tf.compat.v1.constant(0.0, dtype=tf.compat.v1.float32), emb_test, embs_ref)
    )
    return ret / ref_size


class Discriminator(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
            self, sequence_length, num_classes, vocab_size,
            emd_dim, filter_sizes, num_filters, l2_reg_lambda=0.0, batch_size=32, reference_size=16, dropout_keep_prob = .75):
        # Placeholders for input, output and dropout
        self.input_x = tf.compat.v1.placeholder(tf.compat.v1.int32, [batch_size,  sequence_length], name="input_x")
        self.input_ref = tf.compat.v1.placeholder(tf.compat.v1.int32, [reference_size, sequence_length], name="input_ref")
        self.input_y = tf.compat.v1.placeholder(tf.compat.v1.float32, [batch_size,  num_classes], name="input_y")
        self.dropout_keep_prob = dropout_keep_prob

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.compat.v1.constant(0.0)

        with tf.compat.v1.variable_scope('discriminator'):

            # Embedding layer
            with tf.compat.v1.device('/cpu:0'), tf.compat.v1.name_scope("embedding"):
                self.W = tf.compat.v1.Variable(
                    tf.compat.v1.random_uniform([vocab_size, emd_dim], -1.0, 1.0),
                    name="W")
                self.embedded_chars = tf.compat.v1.nn.embedding_lookup(self.W, self.input_x)
                self.embedded_chars_expanded = tf.compat.v1.expand_dims(self.embedded_chars, -1)
                self.embedded_chars_ref = tf.compat.v1.nn.embedding_lookup(self.W, self.input_ref)
                self.embedded_chars_expanded_ref = tf.compat.v1.expand_dims(self.embedded_chars_ref, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            pooled_outputs_ref = []
            for filter_size, num_filter in zip(filter_sizes, num_filters):
                with tf.compat.v1.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, emd_dim, 1, num_filter]
                    W = tf.compat.v1.Variable(tf.compat.v1.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[num_filter]), name="b")
                    conv = tf.compat.v1.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    conv_ref = tf.compat.v1.nn.conv2d(
                        self.embedded_chars_expanded_ref,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv_ref"
                    )
                    # Apply nonlinearity
                    h = tf.compat.v1.nn.relu(tf.compat.v1.nn.bias_add(conv, b), name="relu")
                    h_ref = tf.compat.v1.nn.relu(tf.compat.v1.nn.bias_add(conv_ref, b, name="relu_ref"))
                    # Maxpooling over the outputs
                    pooled = tf.compat.v1.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_ref = tf.compat.v1.nn.max_pool(
                        h_ref,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool_ref")
                    pooled_outputs.append(pooled)
                    pooled_outputs_ref.append(pooled_ref)

            # Combine all the pooled features
            num_filters_total = sum(num_filters)
            self.h_pool = tf.compat.v1.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.compat.v1.reshape(self.h_pool, [-1, num_filters_total])

            self.h_pool_ref = tf.compat.v1.concat(pooled_outputs_ref, 3)
            self.h_pool_flat_ref = tf.compat.v1.reshape(self.h_pool_ref, [-1, num_filters_total])

            # Add highway
            with tf.compat.v1.name_scope("highway"):
                self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0, scope="highway")
                self.h_highway_ref = highway(self.h_pool_flat_ref, self.h_pool_flat_ref.get_shape()[1], 1, 0,
                                             scope="highway")

            # Add dropout
            with tf.compat.v1.name_scope("dropout"):
                self.h_drop = tf.compat.v1.nn.dropout(self.h_highway, self.dropout_keep_prob)
                self.h_drop_ref = tf.compat.v1.nn.dropout(self.h_highway_ref, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.compat.v1.name_scope("output"):
                """
                scores = tf.compat.v1.TensorArray(dtype=tf.compat.v1.float32, size=batch_size, dynamic_size=False, infer_shape=True)
                def rank_recurrence(i, scores):
                    rank_score = get_rank_score(tf.compat.v1.nn.embedding_lookup(self.h_drop, i), self.h_drop_ref)
                    scores = scores.write(i, rank_score)
                    return i + 1, scores
                _, self.scores = control_flow_ops.while_loop(
                    cond=lambda i, _1: i < batch_size,
                    body=rank_recurrence,
                    loop_vars=(tf.compat.v1.constant(0, dtype=tf.compat.v1.int32), scores)
                )
                """
                score = []
                """
                for i in range(batch_size):
                    value = tf.compat.v1.constant(0.0, dtype=tf.compat.v1.float32)
                    for j in range(reference_size):
                        value += cosine_distance(tf.compat.v1.nn.embedding_lookup(self.h_drop, i),
                                                 tf.compat.v1.nn.embedding_lookup(self.h_drop_ref, j))
                    score.append(value) 
                self.scores = tf.compat.v1.stack(score)
                self.scores = tf.compat.v1.reshape(self.scores, [-1])
                """
                self.reference = tf.compat.v1.reduce_mean(tf.compat.v1.nn.l2_normalize(self.h_drop_ref, axis=-1), axis=0, keep_dims=True)
                self.feature = tf.compat.v1.nn.l2_normalize(self.h_drop, axis=-1)
                self.scores = tf.compat.v1.reshape(self.feature @ tf.compat.v1.transpose(self.reference, perm=[1, 0]), [-1])
                self.ypred_for_auc = tf.compat.v1.reshape(tf.compat.v1.nn.softmax(self.scores), [-1])
                self.log_score = tf.compat.v1.log(self.ypred_for_auc)

            # CalculateMean cross-entropy loss
            with tf.compat.v1.name_scope("loss"):
                self.neg_vec = tf.compat.v1.nn.embedding_lookup(tf.compat.v1.transpose(self.input_y), 1)
                self.pos_vec = tf.compat.v1.nn.embedding_lookup(tf.compat.v1.transpose(self.input_y), 0)
                losses_minus = self.log_score * self.neg_vec
                losses_posit = self.log_score * self.pos_vec
                self.loss = (- tf.compat.v1.reduce_sum(losses_minus) / tf.compat.v1.maximum(tf.compat.v1.reduce_sum(self.neg_vec), 1e-5) + tf.compat.v1.reduce_sum(
                    losses_posit) / tf.compat.v1.maximum(tf.compat.v1.reduce_sum(self.pos_vec), 1e-5)) / reference_size

        self.params = [param for param in tf.compat.v1.trainable_variables() if 'discriminator' in param.name]
        d_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars)
