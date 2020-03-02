from tensorflow.python.keras.models import Sequential, model_from_json
from tensorflow.python.keras.layers import Dense, LSTM, Embedding, BatchNormalization
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
import os
import json


class ModelLSTM:

    def __init__(self, model_dir, graph_config, verbose=True):
        """ Build a Keras sequential model

        The model will contain the following layers

        embeddings -> LSTM -> n * fully connected -> output (1 unit)

        The information of all the graph configuration is saved in

        ../../model/lstm_keras/graph_configs.json.

        They are in a list that look like [graph1_config, graph2_config, ...]. The list index of a graph is its id.

        * The actual graph is saved with the naming convention: graph_(graph id).json.
        * If the model has been trained, the weight with best result is saved with naming convention:
          (config id)_weights.h5

        The constructor will check if there is the same configuration in saved config as graph_config parameter.
        If yes, load from exist and load weight if there is trained weights.
        If no, build from scratch.

        :param str model_dir: directory for saved models the their config information
        :param dict graph_config: This is a dictionary that has the graph configuration information
          the key includes "vocab_size", "doc_len", "lstm_units", "embedding_size", "dense_units"
        :param bool verbose: whether print out the the verbose
        """
        self.verbose = verbose

        self.model_dir = model_dir
        self.graph_config_path = self.model_dir + "graph_configs.json"

        self.log_path = self.model_dir + "logs"
        self.model = None
        self.history = None
        self.graph_config = graph_config
        self.config_id, found_exist = self.get_graph_config_id(self.graph_config)
        if self.verbose:
            if found_exist:
                print("Found matching configuration with id: {}".format(self.config_id))
            else:
                print("No matching configuration, save new config with id: {}".format(self.config_id))
        self.weight_path = self.model_dir + "model_" + str(self.config_id) + "_weights.h5"

        if found_exist:
            self.model = self.load_graph(self.config_id)
            self.load_exist_weight()
        else:
            self.model = self.build_graph(self.graph_config, self.config_id)
        print(self.model.summary())
        self.log_path = self.model_dir + "log/" + str(self.config_id)

    def get_graph_config_id(self):
        config_id = 0
        found_existing = False
        config_list = []
        if os.path.exists(self.graph_config_path):
            with open(self.graph_config_path, 'r') as json_file:
                config_list = json.load(json_file)
            for i in range(len(config_list)):
                if self.graph_config == config_list[i]:
                    found_existing = True
                    config_id = i
                    break
        if not found_existing:
            config_list.append(self.graph_config)
            config_id = len(config_list) - 1
            with open(self.graph_config_path, 'w') as json_file:
                json.dump(config_list, json_file)
        return config_id, found_existing

    def build_graph(self, config_dict, config_id):
        lstm_units = config_dict["lstm_units"]
        embedding_size = config_dict["embedding_size"]
        dense_units = config_dict["dense_units"]
        vocab_size = config_dict["vocab_size"]
        doc_len = config_dict["doc_len"]
        model_layers = []
        model_layers.append(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=doc_len))
        model_layers.append(LSTM(units=lstm_units, input_shape=(doc_len, embedding_size), return_sequences=False))
        for i in dense_units:
            model_layers.append(Dense(units=i, activation="relu"))
            model_layers.append(BatchNormalization())
        model_layers.append(Dense(units=1, activation="sigmoid"))
        model = Sequential(model_layers)
        model_json = model.to_json()
        with open(self.model_dir + "graph_" + str(config_id) + ".json", 'w') as json_file:
            json_file.write(model_json)
        return model

    def load_graph(self, config_id):
        with open(self.model_dir + "graph_" + str(config_id) + ".json", 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        return model

    def load_exist_weight(self):
        for file in os.listdir(self.model_dir):
            if file == "model_" + str(self.config_id) + "_weights.h5":
                self.model.load_weights(self.weight_path)
                if self.verbose:
                    print("Found existing trained weights, loaded to model")
                break

    def train(self, train_generator, train_config):
        steps_p_epoch = train_config["steps_p_epoch"]
        eps = train_config["eps"]
        val_data = train_config["val_data"]
        val_freq = train_config["val_freq"]
        verbose = train_config["verbose"]
        lr = train_config["lr"]
        adam_optimizer = Adam(learning_rate=lr)
        self.model.compile(loss="binary_crossentropy", optimizer=adam_optimizer, metrics=["accuracy"])
        check_point = ModelCheckpoint(filepath=self.weight_path,
                                      monitor="val_acc",
                                      verbose=False,
                                      save_best_only=True,
                                      save_weights_only=True,
                                      mode="max")
        tensorboard = TensorBoard(self.log_path)
        history = self.model.fit_generator(generator=train_generator,
                                            steps_per_epoch=steps_p_epoch,
                                            epochs=eps,
                                            validation_data=val_data,
                                            validation_freq=val_freq,
                                            verbose=verbose,
                                            callbacks=[check_point, tensorboard])
        return history
