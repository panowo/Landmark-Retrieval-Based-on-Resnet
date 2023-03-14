import numpy as np
from collections import Counter
import os
import random
from shutil import move,copy
from ipdb import set_trace

def create_test_class(source, class_dir):
    '''

    :param source: Name of dataset directory
    :param class_dir: Name of class directory to split

    Move n (TEST_DATASET_SIZE %) from source/class_dir directory
    to source_test/class_dir
    :return:
    '''

    files = []
    test_dir = os.path.join(source + '_test', class_dir)

    # Check files im dataset
    for filename in os.listdir(os.path.join(source, class_dir)):
        file = os.path.join(source, class_dir, filename)
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    # Compute len and shuffle dataset
    training_length = int(len(files) * (1 - TEST_DATASET_SIZE))
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    testing_set = shuffled_set[:testing_length]
    train_set = shuffled_set[testing_length+1:]

    # move files from dataset to test_dataset
    for filename in testing_set:
        this_file = os.path.join(source, class_dir, filename)
        destination = os.path.join(test_dir, filename)
        move(this_file, destination)

def divide_class(source, class_dir):
    '''

    :param source: Name of dataset directory
    :param class_dir: Name of class directory to split

    Move n (TEST_DATASET_SIZE %) from source/class_dir directory
    to source_test/class_dir
    :return:
    '''

    files = []
    test_dir = os.path.join(source + '_test', class_dir)
    train_dir=os.path.join(source + '_train', class_dir)

    # Check files im dataset
    for filename in os.listdir(os.path.join(source, class_dir)):
        file = os.path.join(source, class_dir, filename)
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    # Compute len and shuffle dataset
    training_length = int(len(files) * (1 - TEST_DATASET_SIZE))
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    testing_set = shuffled_set[:testing_length]
    train_set = shuffled_set[testing_length:]

    # move files from dataset to test_dataset
    for filename in testing_set:
        this_file = os.path.join(source, class_dir, filename)
        destination = os.path.join(test_dir, filename)
        copy(this_file, destination)

    # move files from dataset to train_dataset
    for filename in train_set:
        this_file = os.path.join(source, class_dir, filename)
        destination = os.path.join(train_dir, filename)
        copy(this_file, destination)

def create_test_dataset(dataset_dir):
    '''
    :param dataset_dir: Path to dataset

    Creating a dataset_test, size of dataset_test is TEST_DATASET_SIZE %
    '''
    test_dir = dataset_dir + '_test'

    # Create dataset_test directoru
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # Move Images for each directory in dataset
    for class_dir in os.listdir(dataset_dir):
        if not os.path.exists(os.path.join(test_dir, class_dir)):
            os.mkdir(os.path.join(test_dir, class_dir))
        create_test_class(dataset_dir, class_dir)
    print('Create {}.'.format(test_dir))

def divide_dataset(dataset_dir):
    '''
    :param dataset_dir: Path to dataset

    Creating a dataset_test, size of dataset_test is TEST_DATASET_SIZE %
    '''
    test_dir = dataset_dir + '_test'
    train_dir=dataset_dir + '_train'

    # Create dataset_test directoru
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    # Move Images for each directory in dataset
    for class_dir in os.listdir(dataset_dir):
        if not os.path.exists(os.path.join(test_dir, class_dir)):
            os.mkdir(os.path.join(test_dir, class_dir))
        if not os.path.exists(os.path.join(train_dir, class_dir)):
            os.mkdir(os.path.join(train_dir, class_dir))
        divide_class(dataset_dir, class_dir)
    print('Create {}.'.format(test_dir))
    print('Create {}.'.format(train_dir))

def create_dataset_csv(dataset_dir):
    '''
    :param dataset_dir: Path to dataset

    Creating a dataset_test, size of dataset_test is TEST_DATASET_SIZE %
    '''
    file_name_csv = dataset_dir+'.csv'

    with open(file_name_csv, 'w', encoding='UTF-8') as file_temp:
            file_temp.write("image_path,class_image\n")
            for root, _, files in os.walk(dataset_dir, topdown=False):
                class_image = root.split(os.path.sep)[-1]
                for name in files:
                    if not name.endswith('.jpg'):
                        continue
                    image_path = os.path.join(root, name)
                    file_temp.write("{},{}\n".format(image_path, class_image))