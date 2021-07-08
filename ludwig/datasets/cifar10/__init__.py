import os
import struct
import pickle

from multiprocessing.pool import ThreadPool
import numpy as np
from skimage.io import imsave

from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
from ludwig.datasets.mixins.download import TarDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin

NUM_LABELS = 10


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=False):
    dataset = CIFAR10(cache_dir=cache_dir)
    return dataset.load(split=split)


class CIFAR10(CSVLoadMixin, TarDownloadMixin, BaseDataset):
    """ CIFAR10 Dataset """

    config: dict
    raw_temp_path: str
    raw_dataset_path: str
    processed_temp_path: str
    processed_dataset_path: str

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION):
        super().__init__(dataset_name="cifar10", cache_dir=cache_dir)

    def process_downloaded_dataset(self):
        """Read the training and test directories and write out
        a csv containing the training path and the label.
        """
        os.makedirs(self.processed_temp_path, exist_ok=True)
        for dataset in ["training", "testing"]:
            print(f'>>> create ludwig formatted {dataset} data')
            labels, data = self.read_source_dataset(dataset,
                                                    self.raw_dataset_path)
            self.write_output_dataset(labels, data,
                                      os.path.join(self.processed_temp_path,
                                                   dataset))
        self.output_training_and_test_data()
        os.rename(self.processed_temp_path, self.processed_dataset_path)
        print('>>> completed data preparation')

    def read_source_dataset(self, dataset="training", path="."):
        """Create a directory for training and test and extract all the images
        and labels to this destination.
        :args:
            dataset (str) : the label for the dataset
            path (str): the raw dataset path
        :returns:
            A tuple of the label for the image, the file array, the size and rows and columns for the image"""

        training_file_paths = ['cifar-10-batches-py/data_batch_1',
                               'cifar-10-batches-py/data_batch_2', 'cifar-10-batches-py/data_batch_3',
                               'cifar-10-batches-py/data_batch_4', 'cifar-10-batches-py/data_batch_5']
        test_file_paths = ['cifar-10-batches-py/test_batch']
        labels, data = [], []
        if dataset == "training":
            for file_path in training_file_paths:
                with open(os.path.join(path, file_path), 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                    labels.extend(dict[b'labels'])
                    data.extend(dict[b'data'])
        elif dataset == "testing":
            for file_path in test_file_paths:
                with open(os.path.join(path, file_path), 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                    labels.extend(dict[b'labels'])
                    data.extend(dict[b'data'])
        else:
            raise ValueError("dataset must be 'testing' or 'training'")

        data = np.vstack(data).reshape(-1, 3, 32, 32)
        data = data.transpose((0, 2, 3, 1))  # convert to HWC

        return labels, data

    def write_output_dataset(self, labels, data, output_dir):
        """Create output directories where we write out the images.
        :args:
            labels (str) : the labels for the image
            data (np.array) : the binary array corresponding to the image
            output_dir (str) : the output directory that we need to write to
            path (str): the raw dataset path
        :returns:
            A tuple of the label for the image, the file array, the size and rows and columns for the image"""
        # create child image output directories
        output_dirs = [
            os.path.join(output_dir, str(i))
            for i in range(NUM_LABELS)
        ]

        for output_dir in output_dirs:
            os.makedirs(output_dir, exist_ok=True)

        def write_processed_image(t):
            i, label = t
            output_filename = os.path.join(output_dirs[label], str(i) + ".png")
            imsave(output_filename, data[i])

        # write out image data
        tasks = list(enumerate(labels))
        pool = ThreadPool(NUM_LABELS)
        pool.map(write_processed_image, tasks)
        pool.close()
        pool.join()

    def output_training_and_test_data(self):
        """The final method where we create a training and test file by iterating through
        all the images and labels previously created.
        """
        with open(
                os.path.join(self.processed_temp_path, self.csv_filename),
                'w'
        ) as output_file:
            output_file.write('image_path,label,split\n')
            for name in ["training", "testing"]:
                split = 0 if name == 'training' else 2
                for i in range(NUM_LABELS):
                    img_path = os.path.join(
                        self.processed_temp_path,
                        '{}/{}'.format(name, i)
                    )
                    final_img_path = os.path.join(
                        self.processed_dataset_path,
                        '{}/{}'.format(name, i)
                    )
                    for file in os.listdir(img_path):
                        if file.endswith(".png"):
                            output_file.write('{},{},{}\n'.format(
                                os.path.join(final_img_path, file),
                                str(i),
                                split
                            ))

    @property
    def download_url(self):
        return self.config["download_url"]
