import tensorflow as tf

class lstm_classifier():

    def __init__(self, num_wds, embed_size, lstm_size, wd_dict_size, dense_size, class_size):
        self.num_wds = num_wds
        self.embed_size = embed_size
        self.lstm_size = lstm_size
        self.dense_size = dense_size
        self.class_size = class_size
        self.wd_dict_size = wd_dict_size
        self.build_graph()

    def build_graph(self):

        tf.reset_default_graph()

        with tf.name_scope("input"):
            X = tf.placeholder(dtype=tf.int32, shape=[None, self.num_wds])

        with tf.name_scope("embedding"):
            embed_lookup = tf.Variable(tf.random_normal(shape=[self.wd_dict_size, self.embed_size]))
            wd_embed = tf.nn.embedding_lookup(embed_lookup, X)

        with tf.name_scope("lstm_layer"):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)
            outputs, states = tf.nn.dynamic_rnn(lstm_cell, wd_embed, dtype=tf.float32)
            rnn_final_out = outputs[:, -1, :]

        with tf.name_scope("feed_forward_layer"):
            full_conn1 = tf.layers.dense(rnn_final_out, self.dense_size, activation=tf.nn.relu)
            model_out = tf.layers.dense(full_conn1, self.class_size, activation=None)
            print(model_out)






