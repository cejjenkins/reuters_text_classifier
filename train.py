from reuters import load_data
from clean_data import setup_split_data, split_val, map_topics, tokenize_data
from classifier import compile_model, evaluate_model
import logging
import sys
import warnings

warnings.filterwarnings("ignore")


def get_logger(name="model"):
    """Set up logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


if __name__ == "__main__":
    logger = get_logger()
    logger.info("Starting the training.")
    df = load_data(logger, "data/reuters21578/")
    train_data, test_data = setup_split_data(logger, df)
    train_labels_encoded, test_labels_encoded = map_topics(
        logger, train_data, test_data
    )
    xt_train, xt_test, t = tokenize_data(logger, train_data, test_data)
    x_val, partial_x_train, y_val, partial_y_train = split_val(
        logger, xt_train, train_labels_encoded
    )
    clf = compile_model(logger)
    logger.info("Start training")
    results = clf.fit(
        partial_x_train, partial_y_train, epochs=20, validation_data=(x_val, y_val)
    )
    evaluate_model(logger, clf, results, xt_test, test_labels_encoded)
