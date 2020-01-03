from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam

class ModelLSTM():
    def __init__(self, lstm_units, embedding_size, doc_len, dense_units, vocab_size, drop_rate, lr):
        self.lstm_units = lstm_units
        self.embedding_size = embedding_size
        self.dense_units = dense_units
        self.vocab_size = vocab_size
        self.drop_rate = drop_rate
        self.learning_rate = lr
        self.doc_len = doc_len
        self.model = self.build_graph()

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

    def train(self, data_generator, batch_size, steps_per_epoch, epochs):
        self.model.fit_generator(data_generator(batch_size), steps_per_epoch=steps_per_epoch, epochs=epochs)

