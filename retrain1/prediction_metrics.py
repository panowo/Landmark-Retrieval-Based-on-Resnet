import numpy as np
from collections import Counter
import os
import random
from shutil import move,copy
import matplotlib
matplotlib.use('agg')
#linux环境
import matplotlib.pyplot as plt
from ipdb import set_trace
from operator import itemgetter, attrgetter
from numpy import dot
from numpy.linalg import norm

DEPTH = 5 #选最相似的几个
CLASS_DEPTH = 3 #

def count_distance(vector_1, vector_2, distance_type):
    '''
    :param vector_1:
    :param vector_2:
    :param distance_type: 'L1' Manhattan distance or 'L2' Euclidean distance

    Counting a distance between two vectors
    '''
    # Manhattan distance
    if distance_type == 'L1':
        return np.sum(np.absolute(vector_1 - vector_2))

    # Euclidean distance
    elif distance_type == 'L2':
        return np.sum((vector_1 - vector_2)**2)

    # cosine similiarity
    elif distance_type == 'L3':
        # set_trace()
        return np.sum(1-dot(vector_1, vector_2)/(norm(vector_1)*norm(vector_2)))

def predict_class(query, vectors, distance, mode=0):
    '''
    :param query: Query Image dict with {'image_path': Path to image,
                                    'class_image': Class of an image,
                                    'feature_vector': Feature vector of image}
    :param vectors: List with query-like type dict of whole dataset
    :param distance: Type of distance
    :param mode:    0 - Print result of prediction
                    1 - Return result of prediction
                    2 - Return list of DEPTH predictions
                    3 - two_stage_rerank

    Computing distance between query image vector and all vectors in dataset.
    Sort result list. Take n(DEPTH) best distances and return most frequently class.
    '''
    result_list = []

    # Count distance between query and dataset vectors
    for temp_vector in vectors:
        distance_len = count_distance(temp_vector['feature_vector'], query['feature_vector'], distance)
        result_list.append({
            'class': temp_vector['class_image'],
            'distance': distance_len,
            'image_path': temp_vector['image_path']
        })

    set_trace()
    # Sorting list by distance
    result_list = sorted(result_list, key=lambda x: x['distance'])

    # Get best distance
    result_best_distance = min(result_list, key=lambda x: x['distance'])

    # Get most frequent class in (DEPTH) first classes
    res_count = Counter([x['class'] for x in result_list][:CLASS_DEPTH])
    res = min(res_count.items(), key=lambda x: (-x[1], x[0]))[0]

    if mode == 2:
        return result_list[:DEPTH]

    if mode == 1:
        return res

    # Check similarity with dataset
    if result_best_distance['distance'] < 0.5:
        print('\nPredicted class for query image: {}.'.format(res))
    else:
        print('\nQuery is not similar to any class.')

def predict_new_class(query, vectors, distance, mode):
    '''
    :param query: Query Image dict with {'image_path': Path to image,
                                    'class_image': Class of an image,
                                    'feature_vector': Feature vector of image}
    :param vectors: List with query-like type dict of whole dataset
    :param distance: Type of distance
    :param mode:    0 - Print result of prediction
                    1 - Return result of prediction
                    2 - Return list of DEPTH predictions

    Computing distance between query image vector and all vectors in dataset.
    Sort result list. Take n(DEPTH) best distances and return most frequently class.
    '''
    result_list = []

    # Count distance between query and dataset vectors
    for temp_vector in vectors:
        distance_len = count_distance(temp_vector['feature_vector'], query['feature_vector'], distance)
        result_list.append({
            'class': temp_vector['class_image'],
            'distance': distance_len,
            'image_path': temp_vector['image_path'],
            'priority': 0
        })

    # set_trace()
    # Sorting list by distance
    result_list = sorted(result_list, key=lambda x: x['distance'])

    # Get best distance
    result_best_distance = min(result_list, key=lambda x: x['distance'])

    # Get most frequent class in (DEPTH) first classes
    res_count = Counter([x['class'] for x in result_list][:CLASS_DEPTH])
    res = min(res_count.items(), key=lambda x: (-x[1], x[0]))[0]

    # rerank
    for result in result_list:
        if result['class']==res:
            result['priority']=1

    # set_trace()
    # result_list = sorted(result_list, key=lambda x: (x['distance'], x['priority']))
    result_list = sorted(result_list, key=lambda x: x['priority'],reverse=True)
    # set_trace()

    if mode == 2:
        return result_list[:DEPTH]

    if mode == 1:
        return res

    # Check similarity with dataset
    if result_best_distance['distance'] < 0.5:
        print('\nPredicted class for query image: {}.'.format(res))
    else:
        print('\nQuery is not similar to any class.')

def get_ap(label, results):
    '''
    :param label: True class of image
    :param results: DEPTH-samples of best distance predicted classes

    Counting an average precision
    :return:
    '''
    precision = []
    hit = 0

    # Calculate precision
    for i, result in enumerate(results):

        if result['class'] == label:
            hit += 1

        # if best distance not the same class like label
        if hit == 0:
            return 0.
        precision.append(hit / (i + 1.))

    return np.mean(precision)


def calculate_ap(query, vectors, distance):
    '''
    :param query: Query Image dict with {'image_path': Path to image,
                                    'class_image': Class of an image,
                                    'feature_vector': Feature vector of image}
    :param vectors: List with query-like type dict of whole dataset
    :param distance: distance: Type of distance

    Calculating an average precision for query
    :return:
    '''
    # set_trace()
    results = []

    # Create result list with distances between query and vectors in dataset
    for temp in vectors:
        if query['image_path'] == temp['image_path']:
            continue
        distance_len = count_distance(query['feature_vector'], temp['feature_vector'], distance)
        results.append({'class': temp['class_image'],
                        'distance': distance_len})

    # set_trace()

    # Sorting list by distance
    results = sorted(results, key=lambda x: x['distance'])

    # Take DEPTH samples
    results = results[:DEPTH]

    # Calculate AP
    ap = get_ap(query['class_image'], results)
    return ap

def calculate_new_ap(query, vectors, distance):
    '''
    :param query: Query Image dict with {'image_path': Path to image,
                                    'class_image': Class of an image,
                                    'feature_vector': Feature vector of image}
    :param vectors: List with query-like type dict of whole dataset
    :param distance: distance: Type of distance

    Calculating an average precision for query
    :return:
    '''
    # set_trace()
    results = []

    # Create result list with distances between query and vectors in dataset
    for temp in vectors:
        if query['image_path'] == temp['image_path']:
            continue
        distance_len = count_distance(query['feature_vector'], temp['feature_vector'], distance)
        results.append({'class': temp['class_image'],
                        'distance': distance_len,'priority': 0})

    # set_trace()

    # Sorting list by distance
    results = sorted(results, key=lambda x: x['distance'])

    # rerank
    # Get best distance
    result_best_distance = min(results, key=lambda x: x['distance'])

    # Get most frequent class in (DEPTH) first classes
    res_count = Counter([x['class'] for x in results][:CLASS_DEPTH])
    res = min(res_count.items(), key=lambda x: (-x[1], x[0]))[0]

    # rerank
    for result in results:
        if result['class']==res:
            result['priority']=1

    # set_trace()
    # result_list = sorted(result_list, key=lambda x: (x['distance'], x['priority']))
    results = sorted(results, key=lambda x: x['priority'],reverse=True)
    

    # Take DEPTH samples
    results = results[:DEPTH]

    # Calculate AP
    ap = get_ap(query['class_image'], results)
    return ap

def calculate_map_metric(dataset, testdataset, FeatureExtractor, distance='L1'):
    '''
    :param dataset: Object of DataSet with images and classes
    :param FeatureExtractor: Object of FeatureExtractor
    :param distance: Type of distance


    Compute all Images from dataset into feature vectors.
    Calculating an Average Precision for each image in class.
    Calculating Mean Average Precision for each class and Mean MAP
    '''

    # Get list of labels
    labels = dataset.get_labels()
    ret = {c: [] for c in labels}
    mean_ap = []

    # Create object to create feature vectors from photos
    extractor = FeatureExtractor(dataset.dataset_name)

    # Create vectors
    vectors = extractor.feature_vectors(dataset)

    test_extractor = FeatureExtractor(testdataset.dataset_name)

    test_vectors = test_extractor.feature_vectors(testdataset)

    # Calculate Average Precision for each image and store it for each class
    # set_trace()
    for temp_vector in test_vectors:
        ap = calculate_ap(temp_vector, vectors, distance)
        ret[temp_vector['class_image']].append(ap)

    # Calculate MAP and MMAP
    print('\nMAP based on first {} vectors'.format(DEPTH))
    for class_temp, ap_temp in ret.items():
        map = np.mean(ap_temp)
        mean_ap.append(map)
        print("Class: {}, AP: {}".format(class_temp, round(map, 2)))
    print("MAP: ", np.mean(mean_ap))

def calculate_new_map_metric(dataset, testdataset,FeatureExtractor, distance='L1'):
    '''
    :param dataset: Object of DataSet with images and classes
    :param FeatureExtractor: Object of FeatureExtractor
    :param distance: Type of distance


    Compute all Images from dataset into feature vectors.
    Calculating an Average Precision for each image in class.
    Calculating Mean Average Precision for each class and Mean MAP
    '''

    # Get list of labels
    labels = dataset.get_labels()
    ret = {c: [] for c in labels}
    mean_ap = []

    # Create object to create feature vectors from photos
    extractor = FeatureExtractor(dataset.dataset_name)

    # Create vectors
    vectors = extractor.feature_vectors(dataset)

    test_extractor = FeatureExtractor(testdataset.dataset_name)

    test_vectors = test_extractor.feature_vectors(testdataset)

    # Calculate Average Precision for each image and store it for each class
    # set_trace()
    for temp_vector in test_vectors:
        ap = calculate_new_ap(temp_vector, vectors, distance)
        ret[temp_vector['class_image']].append(ap)

    # Calculate MAP and MMAP
    print('\nMAP based on first {} vectors'.format(DEPTH))
    for class_temp, ap_temp in ret.items():
        map = np.mean(ap_temp)
        mean_ap.append(map)
        print("Class: {}, MAP: {}".format(class_temp, round(map, 2)))
    print("MMAP: ", np.mean(mean_ap))

def calculate_map_vlad_metric(dataset, testdataset,vlad_class, distance):
    '''
    :param dataset: Object of DataSet with images and classes
    :param FeatureExtractor: Object of FeatureExtractor
    :param distance: Type of distance


    Compute all Images from dataset into feature vectors.
    Calculating an Average Precision for each image in class.
    Calculating Mean Average Precision for each class and Mean MAP
    '''

    # Get list of labels
    labels = dataset.get_labels()
    ret = {c: [] for c in labels}
    mean_ap = []

    _, vlad_vectors = vlad_class.get_clusters_vlad_descriptors(dataset)
    _, test_vlad_vectors = vlad_class.get_clusters_vlad_descriptors(testdataset)

    # Calculate Average Precision for each image and store it for each class
    for temp_vector in test_vlad_vectors:
        ap = calculate_ap(temp_vector, vlad_vectors, distance)
        ret[temp_vector['class_image']].append(ap)

    # Calculate MAP and MMAP
    print('\nMAP based on first {} vectors'.format(DEPTH))
    for class_temp, ap_temp in ret.items():
        map = np.mean(ap_temp)
        mean_ap.append(map)
        print("Class: {}, MAP: {}".format(class_temp, round(map, 2)))
    print("MMAP: ", np.mean(mean_ap))

def calculate_new_map_vlad_metric(dataset, vlad_class, distance):
    '''
    :param dataset: Object of DataSet with images and classes
    :param FeatureExtractor: Object of FeatureExtractor
    :param distance: Type of distance


    Compute all Images from dataset into feature vectors.
    Calculating an Average Precision for each image in class.
    Calculating Mean Average Precision for each class and Mean MAP
    '''

    # Get list of labels
    labels = dataset.get_labels()
    ret = {c: [] for c in labels}
    mean_ap = []

    _, vlad_vectors = vlad_class.get_clusters_vlad_descriptors(dataset)

    # Calculate Average Precision for each image and store it for each class
    for temp_vector in vlad_vectors:
        ap = calculate_new_ap(temp_vector, vlad_vectors, distance)
        ret[temp_vector['class_image']].append(ap)

    # Calculate MAP and MMAP
    print('\nMAP based on first {} vectors'.format(DEPTH))
    for class_temp, ap_temp in ret.items():
        map = np.mean(ap_temp)
        mean_ap.append(map)
        print("Class: {}, MAP: {}".format(class_temp, round(map, 2)))
    print("MMAP: ", np.mean(mean_ap))

def calculate_accuracy_metric_vlad(dataset, test_dataset, vlad_class, distance):
    '''
    :param dataset: Object of main DataSet with images and classes
    :param test_dataset: Object of test DataSet with images and classes
    :param FeatureExtractor: Object of FeatureExtractor
    :param distance: Type of distance

    Compute all Images from dataset into feature vectors.
    Going through test_dataset directory and make prediction for every image.
    When calculating an accuracy for each class
    '''

    kmeans_clusters, vlad_descriptors = vlad_class.get_clusters_vlad_descriptors(dataset)
    # Create object to create feature vectors from photos

    all_result = []
    mean_acc=0
    num=0

    # For each image in dataset_test make predictions and check accuracy for each class
    for class_path in os.listdir(test_dataset):
        right_pred = 0
        for query_path in os.listdir(os.path.join(test_dataset, class_path)):

            # Create feature vector
            class_pred = vlad_class.get_prediction(kmeans_clusters, vlad_descriptors, distance,
                        os.path.join(test_dataset, class_path, query_path), 1)

            if class_pred == class_path:
                    right_pred += 1
                    mean_acc +=1
        accuracy_class = right_pred / len(os.listdir(os.path.join(test_dataset, class_path)))
        num=num+len(os.listdir(os.path.join(test_dataset, class_path)))
        all_result.append({'accuracy': accuracy_class,
                           'class': class_path})
    print('\n')
    for temp_list in all_result:
        print('Class: {}, accuracy: {}.'.format(temp_list['class'], round(temp_list['accuracy'], 2)))
    print('Average accuracy: ', mean_acc/num)


def calculate_accuracy_metric(dataset, test_dataset, FeatureExtractor, distance):
    '''
    :param dataset: Object of main DataSet with images and classes
    :param test_dataset: Object of test DataSet with images and classes
    :param FeatureExtractor: Object of FeatureExtractor
    :param distance: Type of distance

    Compute all Images from dataset into feature vectors.
    Going through test_dataset directory and make prediction for every image.
    When calculating an accuracy for each class
    '''

    # Create object to create feature vectors from photos
    extractor = FeatureExtractor(dataset.dataset_name)

    # Create vectors
    vectors = extractor.feature_vectors(dataset)

    all_result = []
    # set_trace()
    mean_acc=0
    num=0

    # For each image in dataset_test make predictions and check accuracy for each class
    for class_path in os.listdir(test_dataset):
        right_pred = 0
        for query_path in os.listdir(os.path.join(test_dataset, class_path)):
            # try:
                # Create feature vector
                query = extractor.compute_query_vector(os.path.join(test_dataset, class_path, query_path))

            # Make prediction and check it
                class_pred = predict_class(query, vectors, distance, 1)
                if class_pred == class_path:
                    right_pred += 1
                    mean_acc +=1
            # except:
                # print(os.path.join(test_dataset, class_path, query_path))
        accuracy_class = right_pred / len(os.listdir(os.path.join(test_dataset, class_path)))
        num=num+len(os.listdir(os.path.join(test_dataset, class_path)))
        all_result.append({'accuracy': accuracy_class,
                           'class': class_path})
    print('\n')
    
    for temp_list in all_result:
        print('Class: {}, accuracy: {}.'.format(temp_list['class'], round(temp_list['accuracy'], 2)))
    # set_trace()
    print('Average accuracy: ', mean_acc/num)

def calculate_new_accuracy_metric(dataset, test_dataset, FeatureExtractor, distance):
    '''
    :param dataset: Object of main DataSet with images and classes
    :param test_dataset: Object of test DataSet with images and classes
    :param FeatureExtractor: Object of FeatureExtractor
    :param distance: Type of distance

    Compute all Images from dataset into feature vectors.
    Going through test_dataset directory and make prediction for every image.
    When calculating an accuracy for each class
    '''

    # Create object to create feature vectors from photos
    extractor = FeatureExtractor(dataset.dataset_name)

    # Create vectors
    vectors = extractor.feature_vectors(dataset)

    all_result = []
    set_trace()
    mean_acc=0
    num=0

    # For each image in dataset_test make predictions and check accuracy for each class
    for class_path in os.listdir(test_dataset):
        right_pred = 0
        for query_path in os.listdir(os.path.join(test_dataset, class_path)):
            try:
                # Create feature vector
                query = extractor.compute_query_vector(os.path.join(test_dataset, class_path, query_path))

            # Make prediction and check it
                class_pred = predict_new_class(query, vectors, distance,1)
                if class_pred == class_path:
                    right_pred += 1
                    mean_acc +=1
            except:
                print(os.path.join(test_dataset, class_path, query_path))
        accuracy_class = right_pred / len(os.listdir(os.path.join(test_dataset, class_path)))
        num=num+len(os.listdir(os.path.join(test_dataset, class_path)))
        all_result.append({'accuracy': accuracy_class,
                           'class': class_path})
    print('\n')
    
    for temp_list in all_result:
        print('Class: {}, accuracy: {}.'.format(temp_list['class'], round(temp_list['accuracy'], 2)))
    # set_trace()
    print('Average accuracy: ', mean_acc/num)

def print_nearest_photo(query, vectors, distance):
    '''
    :param query: Query Image dict with {'image_path': Path to image,
                                    'class_image': Class of an image,
                                    'feature_vector': Feature vector of image}
    :param vectors: List with query-like type dict of whole dataset
    :param distance: Type of distance

    Calculate nearest photos and prints nearest DEPTH samples
    '''
    results = predict_class(query, vectors, distance, mode=2)

    plt.imshow(np.array(plt.imread(query['image_path']), dtype=int))
    plt.title('Query Image')
    plt.show()

    for i, vectors in enumerate(results):
        image_path = vectors['image_path']
        image = np.array(plt.imread(image_path), dtype=int)
        plt.title('Result {}'.format(i))
        plt.imshow(image)
        plt.show()

def save_nearest_photo(query, vectors, distance, method):
    '''
    :param query: Query Image dict with {'image_path': Path to image,
                                    'class_image': Class of an image,
                                    'feature_vector': Feature vector of image}
    :param vectors: List with query-like type dict of whole dataset
    :param distance: Type of distance

    Calculate nearest photos and prints nearest DEPTH samples
    '''
    if method=='one':
        results = predict_class(query, vectors, distance, mode=2)
    else:
        results = predict_new_class(query, vectors, distance, mode=2)
    # set_trace()

    output_name=os.path.splitext(os.path.split(query['image_path'])[-1])[0]
    output_path=os.path.join('output',output_name)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    plt.imshow(np.array(plt.imread(query['image_path']), dtype=int))
    plt.title('Query Image')
    plt.axis('off')
    plt.savefig(os.path.join(output_path,'Query.png'))#保存图片
    # plt.show()
    with open(os.path.join(output_path,'result.txt'), 'a', encoding='utf8') as file:
        for i, vectors in enumerate(results):
            image_path = vectors['image_path']
            print("the "+str(i)+" of the query:"+image_path)
            file.write("the "+str(i)+" of the query:"+image_path+'\n')
            image = np.array(plt.imread(image_path), dtype=int)
            plt.imshow(image)
            plt.axis('off')
            plt.savefig(os.path.join(output_path,'Result {}.png'.format(i)))#保存图片


def save_three_nearest_photo(query, vectors, distance):
    '''
    :param query: Query Image dict with {'image_path': Path to image,
                                    'class_image': Class of an image,
                                    'feature_vector': Feature vector of image}
    :param vectors: List with query-like type dict of whole dataset
    :param distance: Type of distance

    Calculate nearest photos and prints nearest DEPTH samples
    '''
    results = predict_class(query, vectors, distance, mode=2)

    plt.figure(figsize=(5, 2))
    plt.subplot(1,4,1)
    plt.imshow(np.array(plt.imread(query['image_path']), dtype=int))
    plt.title('Query Image')
    plt.axis('off')
    # plt.show()
    # plt.savefig('result.png')#保存图片

    for i, vectors in enumerate(results):
        image_path = vectors['image_path']
        print("the "+str(i)+" of the query:"+image_path)
        plt.subplot(1,4,i+2)
        image = np.array(plt.imread(image_path), dtype=int)
        plt.axis('off')
        plt.title('Result {}'.format(i))
        # plt.scatter(x, y, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
        plt.imshow(image)
        # plt.savefig('{}.png'.format(i))#保存图片
        # plt.show()
    
    plt.savefig('result.png',bbox_inches='tight', dpi=300)#保存图片



def print_nearest_photo_vlad(vlad_class, vlad_dataset, query):
    '''
    :param query: Query Image dict with {'image_path': Path to image,
                                    'class_image': Class of an image,
                                    'feature_vector': Feature vector of image}
    :param vectors: List with query-like type dict of whole dataset
    :param distance: Type of distance

    Calculate nearest photos and prints nearest DEPTH samples
    '''
    kmeans_clusters, vlad_descriptors = vlad_class.get_clusters_vlad_descriptors(vlad_dataset)
    results = vlad_class.get_prediction(kmeans_clusters, vlad_descriptors, None,mode=2)

    plt.imshow(np.array(plt.imread(query), dtype=int))
    plt.title('Query Image')
    plt.show()

    for i, vectors in enumerate(results):
        image_path = vectors['image_path']
        image = np.array(plt.imread(image_path), dtype=int)
        plt.title('Result {}'.format(i))
        plt.imshow(image)
        plt.show()

def save_nearest_photo_vlad(vlad_class, vlad_dataset, query,method):
    '''
    :param query: Query Image dict with {'image_path': Path to image,
                                    'class_image': Class of an image,
                                    'feature_vector': Feature vector of image}
    :param vectors: List with query-like type dict of whole dataset
    :param distance: Type of distance

    Calculate nearest photos and prints nearest DEPTH samples
    '''
    kmeans_clusters, vlad_descriptors = vlad_class.get_clusters_vlad_descriptors(vlad_dataset)
    if method =='one':
        results = vlad_class.get_prediction(kmeans_clusters, vlad_descriptors, None,mode=2)
    else:
        results = vlad_class.get_new_prediction(kmeans_clusters, vlad_descriptors, None,mode=2)

    output_name=os.path.splitext(os.path.split(query)[-1])[0]
    output_path=os.path.join('output',output_name)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    plt.imshow(np.array(plt.imread(query), dtype=int))
    plt.title('Query Image')
    # plt.show()
    plt.axis('off')
    plt.savefig(os.path.join(output_path,'Query.png'))#保存图片

    with open(os.path.join(output_path,'result.txt'), 'a', encoding='utf8') as file:
        for i, vectors in enumerate(results):
            image_path = vectors['image_path']
            print("the "+str(i)+" of the query:"+image_path)
            file.write("the "+str(i)+" of the query:"+image_path+'\n')
            image = np.array(plt.imread(image_path), dtype=int)
            plt.title('Result {}'.format(i))
            plt.imshow(image)
            plt.axis('off')
            # plt.show()
            plt.savefig(os.path.join(output_path,'Result {}.png'.format(i)))#保存图片
