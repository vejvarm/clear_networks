import tensorflow as tf

from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist

from helpers import console_logger
from Models import DenseClassifier

LOGGER = console_logger(__name__, "DEBUG")

TEST_SET_SIZE = 8000
BUFFER_SIZE = 10000
BATCH_SIZE = 64

HIDDEN_SIZE = 16
NUM_CLASSES = 10

OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=1e-4)
LOSS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
METRICS = ['accuracy']

EPOCHS = 5
LOG_PERIOD = 100
LAYER_INDEX_NAME_DICT = None  # defaults to what is implemented in the DenseClassifier class
LOG_PATH = "./log/dense_mnist"

if __name__ == '__main__':
    LOGGER.info("Import MNIST dataset.")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    image_shape = x_train.shape[1:]
    LOGGER.debug(f"image_shape: {image_shape}")

    LOGGER.info("Convert datasets into data.Dataset structures.")
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_testval = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    LOGGER.info("Split test set into test and validation set.")
    ds_test = ds_testval.take(TEST_SET_SIZE)
    ds_val = ds_testval.skip(TEST_SET_SIZE)

    LOGGER.info("Plot examples from the train dataset")
    for x, y in ds_train.take(2):
        plt.figure()
        plt.imshow(x)
        plt.title(y.numpy())

    plt.show()

    LOGGER.info("Shuffle and batch datasets.")
    ds_train = ds_train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    ds_test = ds_test.batch(BATCH_SIZE)
    ds_val = ds_val.batch(BATCH_SIZE)

    LOGGER.info("Initialize Dense classifier model.")
    model = DenseClassifier(image_shape, HIDDEN_SIZE, NUM_CLASSES, LOG_PERIOD, LAYER_INDEX_NAME_DICT, LOG_PATH)

    LOGGER.info("Compile model.")
    model.compile(OPTIMIZER, LOSS, METRICS)

    LOGGER.info("Train model on training data.")
    history = model.fit(ds_train, epochs=EPOCHS, validation_data=ds_test)
    LOGGER.debug(f"model: {model.summary()}")

    # model.w_and_g_writer.close()  # TODO: close the writer in the model class itself

    LOGGER.info("Evaluate model on validation data")
    metrics = model.evaluate(ds_val)

    LOGGER.debug(f"metrics: {metrics}")