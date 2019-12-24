import tensorflow as tf
from src.popcorn import PopcornHelper
import os
import numpy as np

class lstm_classifier():

    def __init__(self, num_wds, embed_size, lstm_size, dense_size, class_num, learning_rate, root_dir, batch_size, verbose=True):
        self.num_wds = num_wds
        self.embed_size = embed_size
        self.lstm_size = lstm_size
        self.dense_size = dense_size
        self.class_num = class_num
        self.learning_rate = learning_rate
        self.root_dir = root_dir
        self.model_dir = self.root_dir + "/model/basic_lstm/"
        self.verbose = verbose
        self.train = None
        self.scores = None
        self.accuracy = None
        self.loss = None
        self.confidence = None
        self.X = None
        self.y_true = None
        self.is_training = None
        tf.reset_default_graph()
        self.graph = tf.get_default_graph()
        self.data_helper = None
        self.graph_init = None
        self.setup_data(batch_size)
        self.keep_prob = None
        self.wd_dict_size = len(self.data_helper.corpora_dict)
        self.sess = tf.Session()
        self.batch_size = batch_size
        if not os.path.exists(self.model_dir+"basic_lstm.meta"):
            print("No meta graph found. Start to build graph from scratch")
            self.initialize_graph()
            print("Graph built successfully!")
        else:
            self.restore_model()
            print("Graph built from meta file")

    def new_weight_variable(self, shape, name):
        initial = tf.truncated_normal_initializer(stddev=0.01)
        return tf.get_variable(dtype=tf.float32, shape=shape, initializer=initial, name=name + "_W")

    def new_bias_variable(self, shape, name):
        initial = tf.constant(0., shape=shape, dtype=tf.float32)
        return tf.get_variable(dtype=tf.float32, initializer=initial, name=name + "_b")

    def fully_conn_layer(self, input_X, output_dim, output_name, use_relu=True):
        input_dim = input_X.get_shape()[1]
        W = self.new_weight_variable([input_dim, output_dim], output_name)
        b = self.new_bias_variable([output_dim], output_name)
        affine_out = tf.matmul(input_X, W) + b
        layer_output = affine_out
        if use_relu:
            layer_output = tf.nn.relu(affine_out)
        return tf.identity(layer_output, name=output_name)

    def initialize_graph(self):
        with tf.name_scope("input"):
            self.X = tf.placeholder(dtype=tf.int32, shape=[None, self.num_wds], name="X")
            self.y_true = tf.placeholder(dtype=tf.int32, shape=[None, self.class_num], name="y_true")
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            self.is_training = tf.placeholder(tf.bool, name="is_training")

        with tf.name_scope("embedding"):
            embed_lookup = tf.Variable(tf.random_normal(shape=[self.wd_dict_size, self.embed_size]))
            wd_embed = tf.nn.embedding_lookup(embed_lookup, self.X)

        with tf.name_scope("lstm_layer"):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)
            lstm_cell_dropout = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
            outputs, states = tf.nn.dynamic_rnn(lstm_cell_dropout, wd_embed, dtype=tf.float32)
            rnn_final_out = outputs[:, -1]

        with tf.name_scope("feed_forward_layer"):
            full_conn1 = self.fully_conn_layer(rnn_final_out, self.dense_size[0], output_name="full_conn1")
            full1_batch_norm = tf.layers.batch_normalization(full_conn1, training=self.is_training)
            full_conn2 = self.fully_conn_layer(full1_batch_norm, self.dense_size[1], output_name="full_conn2")
            full2_batch_norm = tf.layers.batch_normalization(full_conn2, training=self.is_training)

        with tf.name_scope("scores_output"):
            self.scores = self.fully_conn_layer(full2_batch_norm, self.class_num, output_name="scores", use_relu=False)
            self.confidence = tf.nn.softmax(logits=self.scores, name="confidence")

        with tf.name_scope("cross_entropy"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y_true), name="loss")

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train = optimizer.minimize(self.loss, name="train")

        with tf.name_scope("accuracy"):
            matches = tf.equal(tf.argmax(self.scores, axis=1), tf.argmax(self.y_true, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(matches, tf.float32), name="accuracy")
        self.sess.run(tf.global_variables_initializer())
        tf.add_to_collection("train", self.train)

    def setup_data(self, batch_size):
        self.data_helper = PopcornHelper(self.root_dir, batch_size=batch_size)
        self.data_helper.setup_data()
        self.data_helper.pad_train_test(max_len=self.num_wds)
        self.data_helper.set_train_validation(val_size=0.25)

    def train_model(self, steps, epochs, validate_every=10):
        X_val = self.data_helper.X_val
        y_val = self.data_helper.y_val
        max_acc = 0.0
        for i in range(epochs):
            X_train_batch, y_train_batch = self.data_helper.get_next_batch(i)
            for j in range(steps):
                _, loss_val, accuracy_val, confidence_val = self.sess.run([self.train, self.loss, self.accuracy, self.confidence],
                                                     feed_dict={
                                                         self.X: X_train_batch,
                                                         self.y_true: y_train_batch,
                                                         self.keep_prob: 0.8,
                                                         self.is_training: True})
                if self.verbose:
                    print("Training loss: {}, accuracy: {}".format(str(loss_val), str(accuracy_val)))
            if i % validate_every == 0:
                acc = self.sess.run(self.accuracy, feed_dict={self.X: X_val,
                                                              self.y_true: y_val,
                                                              self.keep_prob: 1.0,
                                                              self.is_training: False})
                if acc > max_acc:
                    self.save_model(self.sess)
                    max_acc = acc
                    if self.verbose:
                        print("Better accuracy found. LSTM model saved")
                print("----------------- Step {}: validation accuracy {} ----------------".format(str(i), str(acc)))


    def save_model(self, sess):
        saver = tf.train.Saver(save_relative_paths=True)
        if not os.path.exists(self.model_dir + "basic_lstm.meta"):
            save_meta = True
        else: save_meta = False
        saver.save(sess, self.model_dir + "basic_lstm", write_meta_graph=save_meta)

    def restore_model(self):
        saver = tf.train.import_meta_graph(self.model_dir+"basic_lstm.meta")
        saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))
        self.graph = tf.get_default_graph()
        self.X = self.graph.get_tensor_by_name("input/X:0")
        self.y_true = self.graph.get_tensor_by_name("input/y_true:0")
        self.keep_prob = self.graph.get_tensor_by_name("input/keep_prob:0")
        self.is_training = self.graph.get_tensor_by_name("input/is_training:0")
        self.scores = self.graph.get_tensor_by_name("scores_output/scores: 0")
        self.confidence = self.graph.get_tensor_by_name("scores_output/confidence:0")
        self.loss = self.graph.get_tensor_by_name("cross_entropy/loss:0")
        self.accuracy = self.graph.get_tensor_by_name("accuracy/accuracy: 0")
        self.train = tf.get_collection("train")[0]

    def predict(self, raw_text):
        wd_list = self.data_helper.clean_sentence(raw_text)
        # print("cleaned words list: ", wd_list)
        wd_id_list = self.data_helper.corpora_dict.doc2idx(wd_list)
        wd_id_padded = self.data_helper.pad_sequence([wd_id_list], self.num_wds)
        confidence_val = self.sess.run( self.confidence,
                                        feed_dict={self.X: wd_id_padded,
                                                   self.keep_prob: 1.0,
                                                   self.is_training:False})
        print(confidence_val)
