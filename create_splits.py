import argparse
import os
import random
from pathlib import Path

import numpy as np

from utils import get_module_logger


def split(source, destination):
    """
    Takes data which is already split in train_and_validation, where ony every 10th frame is saved, 
    and test, where all frames are present in ordner to create a video in the end. Those data will 
    be symlinked two three new folder, train, val, and test, where the traina nd validation data 
    are split as well.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, to contain 3 sub folders:
            train / val / test
    """
    training_validation_data_fraction_for_traing = 0.8

    source_path = Path(source)
    training_validation_path = source_path / "training_and_validation"
    test_path = source_path / "test"
    train_test_records = [record for record in training_validation_path.glob("*.tfrecord")]
    test_records = [record for record in test_path.glob("*.tfrecord")]

    random.shuffle(train_test_records)
    split_index = int(len(train_test_records) * training_validation_data_fraction_for_traing + 0.5)
    train_records = train_test_records[:split_index]
    validation_records = train_test_records[split_index:]

    destination_path = Path(destination)
    dest_train_path = destination_path / "train"
    dest_val_path = destination_path / "val"
    dest_test_path = destination_path / "test"

    os.mkdir(dest_train_path)
    os.mkdir(dest_val_path)
    os.mkdir(dest_test_path)

    for record in train_records:
        os.symlink(record, dest_train_path / record.name)

    for record in validation_records:
        os.symlink(record, dest_val_path / record.name)

    for record in test_records:
        os.symlink(record, dest_test_path / record.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)
