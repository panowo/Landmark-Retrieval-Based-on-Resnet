import numpy as np
from collections import Counter
import os
import random
from shutil import move,copy
from ipdb import set_trace

TEST_DATASET_SIZE = .1

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
    val_dir=os.path.join(source + '_val', class_dir)
    index_dir=os.path.join(source + '_index', class_dir)

    # Check files im dataset
    for filename in os.listdir(os.path.join(source, class_dir)):
        file = os.path.join(source, class_dir, filename)
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    # Compute len and shuffle dataset
    training_length = int(len(files) * (1 - 4*TEST_DATASET_SIZE))
    testing_length = int(len(files) *TEST_DATASET_SIZE)
    val_length = int(len(files) *TEST_DATASET_SIZE)
    index_length = int(len(files) *2*TEST_DATASET_SIZE)

    shuffled_set = random.sample(files, len(files))
    testing_set = shuffled_set[:testing_length]
    val_set=shuffled_set[testing_length:testing_length+val_length]
    index_set=shuffled_set[2*testing_length:4*testing_length]
    train_set = shuffled_set[4*testing_length:]

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

    for filename in val_set:
        this_file = os.path.join(source, class_dir, filename)
        destination = os.path.join(val_dir, filename)
        copy(this_file, destination)

    for filename in index_set:
        this_file = os.path.join(source, class_dir, filename)
        destination = os.path.join(index_dir, filename)
        copy(this_file, destination)

def divide_index_class(source, class_dir):
    '''

    :param source: Name of dataset directory
    :param class_dir: Name of class directory to split

    Move n (TEST_DATASET_SIZE %) from source/class_dir directory
    to source_test/class_dir
    :return:
    '''

    files = []
    test_dir = os.path.join(source + '_test', class_dir)
    # train_dir=os.path.join(source + '_train', class_dir)
    # val_dir=os.path.join(source + '_val', class_dir)
    index_dir=os.path.join(source + '_index', class_dir)

    # Check files im dataset
    for filename in os.listdir(os.path.join(source, class_dir)):
        file = os.path.join(source, class_dir, filename)
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    # Compute len and shuffle dataset
    # training_length = int(len(files) * (1 - 4*TEST_DATASET_SIZE))
    testing_length = int(len(files) *7*TEST_DATASET_SIZE)
    # val_length = int(len(files) *TEST_DATASET_SIZE)
    index_length = int(len(files) *3*TEST_DATASET_SIZE)

    shuffled_set = random.sample(files, len(files))
    testing_set = shuffled_set[:testing_length]
    # val_set=shuffled_set[testing_length:testing_length+val_length]
    index_set=shuffled_set[testing_length:]
    # train_set = shuffled_set[4*testing_length:]

    # move files from dataset to test_dataset
    for filename in testing_set:
        this_file = os.path.join(source, class_dir, filename)
        destination = os.path.join(test_dir, filename)
        copy(this_file, destination)

    # # move files from dataset to train_dataset
    # for filename in train_set:
    #     this_file = os.path.join(source, class_dir, filename)
    #     destination = os.path.join(train_dir, filename)
    #     copy(this_file, destination)

    # for filename in val_set:
    #     this_file = os.path.join(source, class_dir, filename)
    #     destination = os.path.join(val_dir, filename)
    #     copy(this_file, destination)

    for filename in index_set:
        this_file = os.path.join(source, class_dir, filename)
        destination = os.path.join(index_dir, filename)
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
    val_dir=dataset_dir + '_val'
    index_dir=dataset_dir+ '_index'

    # Create dataset_test directoru
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    if not os.path.exists(val_dir):
        os.mkdir(val_dir)

    if not os.path.exists(index_dir):
        os.mkdir(index_dir)

    # Move Images for each directory in dataset
    for class_dir in os.listdir(dataset_dir):
        if not os.path.exists(os.path.join(test_dir, class_dir)):
            os.mkdir(os.path.join(test_dir, class_dir))
        if not os.path.exists(os.path.join(train_dir, class_dir)):
            os.mkdir(os.path.join(train_dir, class_dir))
        if not os.path.exists(os.path.join(val_dir, class_dir)):
            os.mkdir(os.path.join(val_dir, class_dir))
        if not os.path.exists(os.path.join(index_dir, class_dir)):
            os.mkdir(os.path.join(index_dir, class_dir))
        divide_class(dataset_dir, class_dir)
    print('Create {}.'.format(test_dir))
    print('Create {}.'.format(train_dir))

def divide_iindex_dataset(dataset_dir):
    '''
    :param dataset_dir: Path to dataset

    Creating a dataset_test, size of dataset_test is TEST_DATASET_SIZE %
    '''
    test_dir = dataset_dir + '_test'
    # train_dir=dataset_dir + '_train'
    # val_dir=dataset_dir + '_val'
    index_dir=dataset_dir+ '_index'

    # Create dataset_test directoru
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # if not os.path.exists(train_dir):
    #     os.mkdir(train_dir)

    # if not os.path.exists(val_dir):
    #     os.mkdir(val_dir)

    if not os.path.exists(index_dir):
        os.mkdir(index_dir)

    # Move Images for each directory in dataset
    for class_dir in os.listdir(dataset_dir):
        if not os.path.exists(os.path.join(test_dir, class_dir)):
            os.mkdir(os.path.join(test_dir, class_dir))
        # if not os.path.exists(os.path.join(train_dir, class_dir)):
        #     os.mkdir(os.path.join(train_dir, class_dir))
        # if not os.path.exists(os.path.join(val_dir, class_dir)):
        #     os.mkdir(os.path.join(val_dir, class_dir))
        if not os.path.exists(os.path.join(index_dir, class_dir)):
            os.mkdir(os.path.join(index_dir, class_dir))
        divide_index_class(dataset_dir, class_dir)
    # print('Create {}.'.format(test_dir))
    # print('Create {}.'.format(train_dir))

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