from keras import layers, models
from keras.layers.core import Dense
import matplotlib.pyplot as plt


def compile_model(logger):
    """Create the network and compile it."""
    logger.info("Compiling model")
    model = models.Sequential()
    model.add(layers.Dense(64, activation="relu", input_shape=(10000,)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(13, activation="softmax"))
    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    logger.info(f"The network has {len(model.get_config()['layers'])} layers")
    logger.info(f"The network loss function is {model.loss}")
    return model


def evaluate_model(logger, model, results, xt_test, test_labels):
    """Evaluate on test data, and plot val accuracy."""
    test_loss, test_accuracy = model.evaluate(xt_test, test_labels)
    logger.info(f"On the test data the accuracy is {test_accuracy}")
    acc = results.history["accuracy"]
    val_acc = results.history["val_accuracy"]
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("TrainingvValidationAccuracy.png")
    logger.info("Training complete!")
