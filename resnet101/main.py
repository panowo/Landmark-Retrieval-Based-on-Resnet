import torch
import argparse
from model import FeatureExtractor
from prediction_metrics import *
from dataset import DataSet
from vlad import VladPrediction
from preprocess import *
import time
from tqdm import tqdm

USE_GPU = torch.cuda.is_available()
MODEL_URL_RESNET_152 = 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'

parser = argparse.ArgumentParser(description='Image Classification based on ResNet pre-trained model and L1 norm')
parser.add_argument('--dataset', type=str, help='Path to your dataset', required=True)
parser.add_argument('--test_dataset', type=str, help='Path to your test dataset, work if mode: accuracy')
parser.add_argument('--query', type=str, help='Path to your query: example.jpg')
parser.add_argument('--model', type=str, help='vlad: Predict class using VLAD + SIFT \n'
                                                'resnet: Pre-trained ResNet')
parser.add_argument('--mode', type=str, help='train: Extract feature vectors from given dataset, \n'
                                                'predict: Predict a class of query \n'
                                                'create_test_dataset: Creating a testing dataset from dataset\n'
                                                'divide_dataset: Dividing dataset into train and test from dataset\n'
                                                'create_dataset_csv: Creating csv for dataset\n'
                                                'cbir: Print 3 most similar photo for query'
                                                'vlad: Predict class using VLAD + SIFT')
parser.add_argument('--metric', type=str, help='map: Calculate an average precision of dataset\n'
                                                'accuracy: Calculate accuracy of prediction on testing dataset\n')
parser.add_argument('--distance', type=str, help='L1: Manhattan Distance \n'
                                                'L2: Euclidean distance\n', required=False, default='L2')
parser.add_argument('--method', type=str, help='one: one_stage \n'
                                                'two: two_stage\n', required=False, default='one')
parser.add_argument('--DEPTH', type=int, help='20: query_length \n', required=False, default=10)
args = parser.parse_args()

def print_defenitions():
    """
    Print mode of execution, distance type and available GPU
    """
    print('Model type: {}'.format(args.model))
    print('Mode type: {}.'.format(args.mode))
    print('Metric type: {}'.format(args.metric))
    print('Distance type: {}.'.format(args.distance))
    print('Method type: {}.'.format(args.method))
    if USE_GPU:
        print('GPU available: True.')
    else:
        print('GPU available: False.')

if __name__ == "__main__":
    print_defenitions()

    dataset = DataSet(args.dataset)
    vlad_dataset = DataSet(args.dataset)
    # set_trace()

    if args.mode == 'train':
        if args.model == 'resnet':
            feat_ex = FeatureExtractor(dataset.dataset_name)
            feat_ex.feature_vectors(dataset)
        elif args.model == 'vlad':
            vlad_class = VladPrediction(vlad_dataset, args.dataset, args.query)
            vlad_vectors = vlad_class.vlad_train(vlad_dataset, args.distance)

    elif args.mode == 'predict':
        if args.model == 'resnet':
            extractor = FeatureExtractor(dataset.dataset_name)
            vectors = extractor.feature_vectors(dataset)
            start_t=time.time()
            predict_class(extractor.compute_query_vector(args.query), vectors, args.distance)
            end_t=time.time()
            cost=end_t-start_t
            print(cost)
        elif args.model == 'vlad':
            vlad_class = VladPrediction(vlad_dataset, args.dataset, args.query)
            vlad_vectors = vlad_class.vlad_prediction(vlad_dataset, args.distance,args.method)

    elif args.metric == 'map':
        test_dataset=DataSet(args.test_dataset)
        if args.model == 'resnet':
            if args.method =='one':
                calculate_map_metric(dataset, test_dataset,FeatureExtractor, args.distance)
            else:
                calculate_new_map_metric(dataset, test_dataset,FeatureExtractor, args.distance)
        elif args.model == 'vlad':
            vlad_class = VladPrediction(vlad_dataset, args.dataset, args.query)
            if args.method =='one':
                calculate_map_vlad_metric(vlad_dataset, vlad_class,args.distance)
            else:
                calculate_new_map_vlad_metric(vlad_dataset, vlad_class,args.distance)

    elif args.metric == 'accuracy':
        if args.model == 'resnet':
            if args.method =='one':
                calculate_accuracy_metric(dataset, args.test_dataset, FeatureExtractor, args.distance)
            else:
                calculate_new_accuracy_metric(dataset, args.test_dataset, FeatureExtractor, args.distance)
        elif args.model == 'vlad':
            vlad_class = VladPrediction(vlad_dataset, args.dataset, args.query)
            if args.method =='one':
                calculate_accuracy_metric_vlad(dataset, args.test_dataset, vlad_class, args.distance)
            else:
                calculate_new_accuracy_metric_vlad(dataset, args.test_dataset, vlad_class, args.distance)

    elif args.mode == 'cbir':
        if args.model == 'resnet':
            extractor = FeatureExtractor(dataset.dataset_name)
            vectors = extractor.feature_vectors(dataset)
            start_t=time.time()
            save_nearest_photo(extractor.compute_query_vector(args.query), vectors, args.distance, args.method)
            end_t=time.time()
            cost=end_t-start_t
            print(cost)
            # print_nearest_photo(extractor.compute_query_vector(args.query), vectors, args.distance)
        elif args.model == 'vlad':
            vlad_class = VladPrediction(vlad_dataset, args.dataset, args.query)
            vlad_vectors = save_nearest_photo_vlad(vlad_class, vlad_dataset, args.query,args.method)
            # vlad_vectors = print_nearest_photo_vlad(vlad_class, vlad_dataset, args.query)

    elif args.mode == 'time':
        if args.model == 'resnet':
            extractor = FeatureExtractor(dataset.dataset_name)
            vectors = extractor.feature_vectors(dataset)
            start_t=time.time()
            for i in tqdm(range(100)):
                predict_class(extractor.compute_query_vector(args.query), vectors, args.distance,mode=1)
            end_t=time.time()
            cost=end_t-start_t
            print(cost)
        elif args.model == 'vlad':
            vlad_class = VladPrediction(vlad_dataset, args.dataset, args.query)
            vlad_vectors = vlad_class.vlad_prediction(vlad_dataset, args.distance,args.method)

    elif args.mode == 'create_test_dataset':
        create_test_dataset(dataset.dataset_dir)

    elif args.mode == 'divide_dataset':
        divide_dataset(dataset.dataset_dir)

    elif args.mode == 'divide_index_dataset':
        divide_iindex_dataset(dataset.dataset_dir)

    elif args.mode == 'create_dataset_csv':
        create_dataset_csv(dataset.dataset_dir)

    else:
        print('Wrong flags.')
