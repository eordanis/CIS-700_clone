import tensorflow as tf



class Discriminator(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes,
            emd_dim, filter_sizes, num_filters, g_embeddings=None,
            l2_reg_lambda=0.0, dropout_keep_prob=1):
        self.embbeding_mat = g_embeddings

        # Placeholders for input, output and dropout
        self.input_x = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, sequence_length], name="input_x")
        self.input_x_lable = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, num_classes], name="input_y")

        self.input_y = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, sequence_length], name="input_x")
        self.input_y_lable = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, num_classes], name="input_y")

        self.zh = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, emd_dim], name="zh")

        self.dropout_keep_prob = dropout_keep_prob
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.sequence_length = sequence_length
        self.num_classes = num_classes

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.compat.v1.constant(0.0)

        with tf.compat.v1.variable_scope('discriminator'):
            # Embedding layer
            with tf.compat.v1.device('/cpu:0'), tf.compat.v1.name_scope("embedding"):
                self.W = tf.compat.v1.Variable(
                    tf.compat.v1.random_uniform([emd_dim, emd_dim], -1.0, 1.0),
                    name="W")
                # self.embedded_chars = tf.compat.v1.nn.embedding_lookup(self.W, self.input_x)
                # self.embedded_chars_expanded = tf.compat.v1.expand_dims(self.embedded_chars, -1)

            self.W_conv = list()
            self.b_conv = list()
            for filter_size, num_filter in zip(filter_sizes, num_filters):
                with tf.compat.v1.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, emd_dim, 1, num_filter]
                    W = tf.compat.v1.Variable(tf.compat.v1.truncated_normal(filter_shape, stddev=0.1), name="W")
                    self.W_conv.append(W)
                    b = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[num_filter]), name="b")
                    self.b_conv.append(b)

            num_filters_total = sum(self.num_filters)
            with tf.compat.v1.name_scope("output"):
                self.Wo = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([num_filters_total, self.num_classes], stddev=0.1), name="W")
                self.bo = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[self.num_classes]), name="b")

            # recon layer
            with tf.compat.v1.name_scope("recon"):
                self.Wzh = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([num_filters_total, 1], stddev=0.1), name="Wz")
                self.bzh = tf.compat.v1.Variable(tf.compat.v1.constant(0.0, shape=[1]), name="bz")

            input_xy = tf.compat.v1.concat([self.input_x, self.input_y], axis=0)
            self.input_label = tf.compat.v1.concat([self.input_x_lable, self.input_y_lable], axis=0)

            input_x = tf.compat.v1.nn.embedding_lookup(self.embbeding_mat, input_xy)  # batch_size x seq_length x g_emb_dim
            scores, ypred_for_auc, self.predictions = self.predict(input_x=input_x)

            self.predictions2 = self.predictions

            def compute_pairwise_distances(x, y):
                """Computes the squared pairwise Euclidean distances between x and y.
                Args:
                  x: a tensor of shape [num_x_samples, num_features]
                  y: a tensor of shape [num_y_samples, num_features]
                Returns:
                  a distance matrix of dimensions [num_x_samples, num_y_samples].
                Raises:
                  ValueError: if the inputs do no matched the specified dimensions.
                """

                if not len(x.get_shape()) == len(y.get_shape()) == 2:
                    raise ValueError('Both inputs should be matrices.')

                if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
                    raise ValueError('The number of features should be the same.')

                norm = lambda x: tf.compat.v1.reduce_sum(tf.compat.v1.square(x), 1)

                # By making the `inner' dimensions of the two matrices equal to 1 using
                # broadcasting then we are essentially substracting every pair of rows
                # of x and y.
                # x will be num_samples x num_features x 1,
                # and y will be 1 x num_features x num_samples (after broadcasting).
                # After the substraction we will get a
                # num_x_samples x num_features x num_y_samples matrix.
                # The resulting dist will be of shape num_y_samples x num_x_samples.
                # and thus we need to transpose it again.
                return tf.compat.v1.transpose(norm(tf.compat.v1.expand_dims(x, 2) - tf.compat.v1.transpose(y)))

            def gaussian_kernel_matrix(x, y, sigmas=None):
                r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.
                We create a sum of multiple gaussian kernels each having a width sigma_i.
                Args:
                  x: a tensor of shape [num_samples, num_features]
                  y: a tensor of shape [num_samples, num_features]
                  sigmas: a tensor of floats which denote the widths of each of the
                    gaussians in the kernel.
                Returns:
                  A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
                """
                if sigmas is None:
                    sigmas = [
                        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
                        1e3, 1e4, 1e5, 1e6
                    ]
                beta = 1. / (2. * (tf.compat.v1.expand_dims(sigmas, 1)))

                dist = compute_pairwise_distances(x, y)

                s = tf.compat.v1.matmul(beta, tf.compat.v1.reshape(dist, (1, -1)))

                return tf.compat.v1.reshape(tf.compat.v1.reduce_sum(tf.compat.v1.exp(-s), 0), tf.compat.v1.shape(dist))

            def calc_mmd(x, y):
                cost = tf.compat.v1.reduce_mean(gaussian_kernel_matrix(x, x))
                cost += tf.compat.v1.reduce_mean(gaussian_kernel_matrix(y, y))
                cost -= 2 * tf.compat.v1.reduce_mean(gaussian_kernel_matrix(x, y))

                # We do not allow the loss to become negative.
                cost = tf.compat.v1.where(cost > 0, cost, 0, name='value')

                return cost

            # CalculateMean cross-entropy loss
            with tf.compat.v1.name_scope("loss"):
                batch_num = tf.compat.v1.shape(scores)[0]
                pos_score = tf.compat.v1.slice(scores, begin=[0, 0], size=[batch_num, 1])
                pos_label = tf.compat.v1.slice(self.input_label, begin=[0, 0], size=[batch_num, 1])
                gan_loss = tf.compat.v1.log(tf.compat.v1.norm(pos_score - pos_label, ord=1))
                x_feature = self.feature(input_x=self.input_x, name='x')
                y_feature = self.feature(input_x=self.input_y, name='y')
                mmd_loss = calc_mmd(x_feature, y_feature)

                z_hat = tf.compat.v1.matmul(x_feature, self.Wzh)
                recon_loss = - tf.compat.v1.square(tf.compat.v1.norm(tf.compat.v1.subtract(z_hat, self.zh), axis=1))
                self.loss = tf.compat.v1.reduce_mean(gan_loss) + l2_reg_lambda * l2_loss + 0.1 * mmd_loss + 0.1 * recon_loss

        self.params = [param for param in tf.compat.v1.trainable_variables() if 'discriminator' in param.name]
        d_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars)

    def set_embbeding_mat(self, generator):
        raise NotImplementedError



    def feature(self, input_x, name = ''):
        if len(input_x.get_shape()) == 2:
            # incase input_x : batch_size x seq_length [tokens]
            input_x = tf.compat.v1.nn.embedding_lookup(self.embbeding_mat, input_x)
        # input_x:  batch_size x seq_length x g_emb_dim
        pooled_outputs = []
        index = -1
        embedded_chars = tf.compat.v1.scan(lambda a, x: tf.compat.v1.matmul(x, self.W), input_x)
        embedded_chars_expanded = tf.compat.v1.expand_dims(embedded_chars, -1)
        for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
            index += 1
            with tf.compat.v1.name_scope("conv-maxpool-%s-midterm" % filter_size):
                # Convolution Layer
                conv = tf.compat.v1.nn.conv2d(
                    embedded_chars_expanded,
                    self.W_conv[index],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.compat.v1.nn.relu(tf.compat.v1.nn.bias_add(conv, self.b_conv[index]), name="relu")
                # Maxpooling over the outputs
                pooled = tf.compat.v1.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = sum(self.num_filters)
        h_pool = tf.compat.v1.concat(pooled_outputs, 3)
        h_pool_flat = tf.compat.v1.reshape(h_pool, [-1, num_filters_total])
        return h_pool_flat


    def predict(self, input_x):
        # input_x:  batch_size x seq_length x g_emb_dim
        l2_loss = tf.compat.v1.constant(0.0)
        d_feature = self.feature(input_x)
        # Add dropout
        with tf.compat.v1.name_scope("dropout"):
            h_drop = tf.compat.v1.nn.dropout(d_feature, self.dropout_keep_prob)
        num_filters_total = sum(self.num_filters)
        # Final (unnormalized) scores and predictions
        with tf.compat.v1.name_scope("output"):
            l2_loss += tf.compat.v1.nn.l2_loss(self.Wo)
            l2_loss += tf.compat.v1.nn.l2_loss(self.bo)
            scores = tf.compat.v1.nn.xw_plus_b(h_drop, self.Wo, self.bo, name="scores")
            ypred_for_auc = tf.compat.v1.nn.softmax(scores)
            self.predictions = tf.compat.v1.argmax(scores, 1, name="predictions")

        return scores, ypred_for_auc, self.predictions
