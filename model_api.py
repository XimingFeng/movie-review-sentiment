import flask
import io
from src.preprocessing_helper import PreprocessingHelper
from src.lstm_keras import ModelLSTM
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.backend import set_session
import numpy as np

app = flask.Flask(__name__)


def load_model():
    print("* Loading Keras model and Flask starting server... please wait until server has fully started")
    model_config = {
        "embedding_size": 256,
        "lstm_units": 128,
        "dense_units": [256],
        "doc_len": 200,
        "vocab_size": 5000
    }
    model = ModelLSTM(root_dir="./", graph_config=model_config, verbose=True)
    global graph
    graph = tf.get_default_graph()
    return model


def load_data_preprocess_helper():
    prep_helper = PreprocessingHelper(root_dir="./", max_doc_len=200, vocab_size=50000)
    return prep_helper

def prepare_text(text):
    return [i for i in range(200)]


@app.route("/predict", methods=["POST"])
def predict():
    return_data = {
        "success": False
    }
    if flask.request.method == "POST":
        input_text = flask.request.form.get("input_text")
        return_data['success'] = True
        print("Received text: {}".format(input_text))
        cleaned_text = prep_helper.clean_docs([input_text])
        text_ids = tokenizer.texts_to_sequences(cleaned_text)
        text_ids_padded = pad_sequences(text_ids, maxlen=200, padding="post", truncating="post")
        text_ids_padded = np.array(text_ids_padded)
        print("Text ids padded: ", text_ids_padded)
        with graph.as_default():
            set_session(sess)
            confidence = lstm_helper.model.predict(text_ids_padded)[0, 0]
        print("Confidence: ", confidence)
        if confidence < 0.4:
            prediction = "Negative"
        elif confidence > 0.6:
            prediction = "Positive"
        else:
            prediction = "Neutral"
        return_data['prediction'] = prediction
        return_data['confidence'] = str(confidence)
    return flask.jsonify(return_data)


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    prep_helper = load_data_preprocess_helper()
    tokenizer = prep_helper.tokenizer
    global graph
    global sess
    sess = tf.Session()
    set_session(sess)
    graph = tf.get_default_graph()
    lstm_helper = load_model()
    app.run()

