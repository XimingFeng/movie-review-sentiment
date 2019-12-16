import tensorflow as tf
from src.popcorn import PopcornHelper

class lstm_classifier():

    def __init__(self, num_wds, embed_size, lstm_size, dense_size, class_num, learning_rate, root_dir, batch_size, verbose=True):
        self.num_wds = num_wds
        self.embed_size = embed_size
        self.lstm_size = lstm_size
        self.dense_size = dense_size
        self.class_num = class_num
        self.learning_rate = learning_rate
        self.root_dir = root_dir
        self.verbose = verbose
        self.train = None
        self.accuracy = None
        self.loss = None
        self.X = None
        self.y_true = None
        self.graph = None
        self.data_helper = None
        self.graph_init = None
        self.setup_data(batch_size)
        self.keep_prob = None
        self.wd_dict_size = len(self.data_helper.corpora_dict)
        self.initialize_graph()


    def initialize_graph(self):
        tf.reset_default_graph()

        self.graph = tf.Graph()
        with self.graph.as_default():

            with tf.name_scope("input"):
                self.X = tf.placeholder(dtype=tf.int32, shape=[None, self.num_wds], name="X")
                self.y_true = tf.placeholder(dtype=tf.int32, shape=[None, self.class_num], name="y_true")
                self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

            with tf.name_scope("embedding"):
                embed_lookup = tf.Variable(tf.random_normal(shape=[self.wd_dict_size, self.embed_size]))
                wd_embed = tf.nn.embedding_lookup(embed_lookup, self.X)

            with tf.name_scope("lstm_layer"):
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)
                lstm_cell_dropout = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
                outputs, states = tf.nn.dynamic_rnn(lstm_cell_dropout, wd_embed, dtype=tf.float32)
                rnn_final_out = outputs[:, -1]

            with tf.name_scope("feed_forward_layer"):
                full_conn1 = tf.layers.dense(rnn_final_out, self.dense_size[0], activation=tf.nn.relu)
                full1_batch_norm = tf.layers.batch_normalization(full_conn1, training=True)
                full_conn2 = tf.layers.dense(full1_batch_norm, self.dense_size[1], activation=tf.nn.relu)
                full2_batch_norm = tf.layers.batch_normalization(full_conn2, training=True)
                scores = tf.layers.dense(full2_batch_norm, self.class_num, activation=None, name="scores")

            with tf.name_scope("cross_entropy"):
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.y_true))

            with tf.name_scope("train"):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.train = optimizer.minimize(self.loss, name="train")

            with tf.name_scope("accuracy"):
                matches = tf.equal(tf.argmax(scores, axis=1), tf.argmax(self.y_true, axis=1))
                self.accuracy = tf.reduce_mean(tf.cast(matches, tf.float32), name="accuracy")

    def setup_data(self, batch_size):
        self.data_helper = PopcornHelper(self.root_dir, batch_size=batch_size)
        self.data_helper.setup_data()
        self.data_helper.pad_train_test(max_len=self.num_wds)
        self.data_helper.set_train_validation(val_size=0.25)

    def train_model(self, steps, epochs, verbose_every=10):

        X_val = self.data_helper.X_val
        y_val = self.data_helper.y_val
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epochs):
                X_train_batch, y_train_batch = self.data_helper.get_next_batch(i)
                for j in range(steps):
                    _, loss_val, accuracy_val = sess.run([self.train, self.loss, self.accuracy],
                                                         feed_dict={
                                                             self.X: X_train_batch,
                                                             self.y_true: y_train_batch,
                                                             self.keep_prob: 0.8})
                    if self.verbose:
                        print("Training loss: {}, accuracy: {}".format(str(loss_val), str(accuracy_val)))
                if i % verbose_every == 0:
                    acc = sess.run(self.accuracy, feed_dict={self.X: X_val, self.y_true: y_val, self.keep_prob: 1.0})
                    print("----------------- Step {}: validation accuracy {} ----------------".format(str(i), str(acc)))
                else:
                    print("-----------------------------------------------------------------")