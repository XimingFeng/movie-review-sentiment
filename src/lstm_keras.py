from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from datetime import datetime


class ModelLSTM():
    def __init__(self, lstm_units, embedding_size, doc_len, dense_units, vocab_size, drop_rate, lr, root_dir):
        self.lstm_units = lstm_units
        self.embedding_size = embedding_size
        self.dense_units = dense_units
        self.vocab_size = vocab_size
        self.drop_rate = drop_rate
        self.learning_rate = lr
        self.doc_len = doc_len
        self.model = self.build_graph()
        self.train_history = None
        self.root_dir = root_dir
        self.model_dir = root_dir + "/model/basic_lstm/"


    def build_graph(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size, output_dim=self.embedding_size, input_length=self.doc_len))
        model.add(LSTM(units=self.lstm_units, return_sequences=False))
        for i in self.dense_units:
            model.add(Dense(units=i, activation="relu"))
            model.add(BatchNormalization())
        # model.add(Dropout(rate=self.drop_rate))
        model.add(Dense(units=1, activation="sigmoid"))
        Adam_optimier = Adam(learning_rate=self.learning_rate)
        model.compile(loss="binary_crossentropy", optimizer=Adam_optimier, metrics=["accuracy"])
        print(model.summary())
        return model

    def train(self, data_generator, batch_size, steps_per_epoch, epochs, val_generator):
        self.train_history = self.model.fit_generator(data_generator(batch_size),
                                                      validation_data=val_generator,
                                                      steps_per_epoch=steps_per_epoch,
                                                      epochs=epochs)

    def save_model(self):
        model_json = self.model.to_json()
        time_str = datetime.now().strftime("LSTM Keras %m-%d-%Y %H-%M")
        model_file_name = time_str + ".json"
        weights_file_name = time_str + ".h5"
        with open(self.model_dir + model_file_name, "w") as model_file:
            model_file.write(model_json)

        self.model.save_weights(self.model_dir + weights_file_name)

    def load_model(self, save_time):
        with open(self.model_dir + "LSTM Keras " + save_time + ".json", "r") as model_file:
            model_json = model_file.read()
        self.model = model_from_json(model_json)
        self.model.load_weights(self.model_dir + "LSTM Keras " + save_time + ".h5")
        Adam_optimier = Adam(learning_rate=self.learning_rate)
        self.model.compile(loss="binary_crossentropy", optimizer=Adam_optimier, metrics=["accuracy"])
        print(self.model.summary())





