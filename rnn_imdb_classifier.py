# inspired by https://www.tensorflow.org/tutorials/text/text_classification_rnn
import tensorflow_datasets as tfds
import tensorflow as tf

from helpers import console_logger, plot_graphs
from Models import RNNClassifier


LOGGER = console_logger("tensorflow", "DEBUG")
DEBUG = False

BUFFER_SIZE = 10000
BATCH_SIZE = 64

RNN_TYPE = "GRU"
LAYER_SIZES = (32, 16, 8)

OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=1e-4)
LOSS = tf.keras.losses.BinaryCrossentropy(from_logits=True)
METRICS = ['accuracy']

EPOCHS = 10
LOG_PERIOD = 100
LAYER_INDEX_NAME_DICT = None  # defaults to what is implemented in the RNNClassifier class


def load_imdb_dataset():
    LOGGER.info("Loading IMDB REVIEWS Dataset")
    dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
    return dataset['train'], dataset['test'], info


def sample_predict(sample_pred_text):
    encoded_sample_pred_text = encoder.encode(sample_pred_text)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))
    return predictions


def infer_sentiment(prediction: tf.float32):
    if prediction > 0.5:
        sentiment = "positive"
    elif prediction == 0.5:
        sentiment = "neutral"
    else:
        sentiment = "negative"
    return sentiment


if __name__ == '__main__':
    train_dataset, test_dataset, info = load_imdb_dataset()
    LOGGER.debug(f"info: {info}")

    LOGGER.info("Initializing english word-level encoder.")
    encoder = info.features['text'].encoder
    LOGGER.debug(f"Encoder vocab size: {encoder.vocab_size}")

    LOGGER.info("Testing encoder integrity.")
    original_string = "Decode this!"
    encoded_string = encoder.encode(original_string)
    decoded_string = encoder.decode(encoded_string)
    LOGGER.debug(f"original_string: {original_string}")
    LOGGER.debug(f"encoded_string: {encoded_string}")
    LOGGER.debug(f"decoded_string: {decoded_string}")
    assert original_string == decoded_string, "Original and decoded string don't match!"

    LOGGER.info("Preparing data for training.")
    train_dataset = (train_dataset
                     .shuffle(BUFFER_SIZE)
                     .padded_batch(BATCH_SIZE, padded_shapes=((None, ), tuple())))
    test_dataset = test_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), tuple()))

    if DEBUG:
        LOGGER.info("Take only part of the datasets for debugging.")
        train_dataset = train_dataset.take(20)
        test_dataset = test_dataset.take(10)

    LOGGER.info("Initializing GRU-RNN model.")
    model = RNNClassifier(encoder.vocab_size, RNN_TYPE, LAYER_SIZES, LOG_PERIOD, LAYER_INDEX_NAME_DICT)

    LOGGER.info("Compiling the model.")
    model.compile(OPTIMIZER, LOSS, METRICS)

    LOGGER.info("Training the model on IMDB REVIEWS training dataset.")
    history = model.fit(train_dataset, epochs=EPOCHS,
                        validation_data=test_dataset,
                        validation_steps=30)
    LOGGER.debug(f"history: {history}")
    LOGGER.debug(f"model: {model.summary()}")

    LOGGER.info("Evaluating the model on IMDB REVIEWS testing dataset.")
    metrics = model.evaluate(test_dataset)
    LOGGER.info(f"Test Loss: {metrics[0]}")
    LOGGER.info(f"Test Accuracy: {metrics[1]}")

    LOGGER.info("Predicting sample text with trained model.")
    sample_pred_text = ("""The movie was cool. The animation and the graphics were out of this world. I would 
    recommend this movie.""")
    prediction = sample_predict(sample_pred_text)
    LOGGER.info(f"Prediction: {prediction} | Sentiment: {infer_sentiment(prediction)}")



