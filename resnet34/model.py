import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.autograd import Variable
from torchvision.models.resnet import Bottleneck, ResNet, BasicBlock
import torch.utils.model_zoo as model_zoo
from six.moves import cPickle
import torch
from ipdb import set_trace
import torchvision.models as models
import matplotlib
matplotlib.use('agg')
from tqdm import tqdm


USE_GPU = torch.cuda.is_available()
MODEL_URL_RESNET_152 = 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
MODEL_URL_RESNET_34 = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'

class ResidualNet(ResNet):
    def __init__(self):
        super().__init__(BasicBlock,[3,4,6,3])
        self.load_state_dict(model_zoo.load_url(MODEL_URL_RESNET_34))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class FeatureExtractor(object):
    def __init__(self, dataset):
        self.model_type = 'vectors_for_dataset_{}_resnet.cpickle'.format(dataset)
        self.metadata_dir = 'metadata'
        self.vectors_dir = 'vectors'
        self.res_model = ResidualNet()
        # self.res_model= models.__dict__['resnet152'](pretrained=True)
        if USE_GPU:
            self.res_model = self.res_model.cuda()

    def load_feature_vectors(self):
        '''
        Load a Python instance from .cpickle file
        vectors = vectors type of {'image_path': Path to image,
                                    'class_image': Class of an image,
                                    'feature_vector': Feature vector of image}
        '''
        print("Computed feature vectors of dataset: True.")
        vectors = cPickle.load(open(os.path.join(self.vectors_dir, self.model_type), "rb", True))
        return vectors

    def preprocces_image(self, image_path):
        '''
        :param image_path: Path to image file

        Preprocess image
        '''

        image = plt.imread(image_path,1)

        image= image[:,:,0:3] 
        # plt.imshow(image)
        # plt.savefig('test.png')#保存图片


        # set_trace()

        # Resize for big-scale images
        image = cv2.resize(image, (600, 600))

        # Change filters (600, 600, 3) -> (3, 600, 600), and normalize
        image = np.transpose(image, (2, 0, 1)) / 255.

        # Add dimension (3, 600, 600) -> (1, 3, 600, 600)
        image = np.expand_dims(image, axis=0)

        # Change to variables to compute gradients
        inputs = Variable(torch.from_numpy(image).cuda().float()) if USE_GPU \
            else Variable(torch.from_numpy(image).float())
        return inputs

    def compute_input(self, input):
        '''
        :param input: preprocessed image

        Create a feature vector from input
        '''
        # set_trace()
        # Apply layers to input
        feature_vector = self.res_model(input)

        # Covert to numpy array
        feature_vector = feature_vector.data.cpu().numpy().flatten()

        # Normalize
        feature_vector /= np.sum(feature_vector)
        return feature_vector

    def compute_query_vector(self, image_path):
        '''
        :param image_path: Path to query image

        Create a feature vector from image
        '''
        self.res_model.eval()
        image_query = self.preprocces_image(image_path)
        result_query = {'feature_vector': self.compute_input(image_query),
                        'image_path': image_path}
        return result_query

    def compute_feature_vectors(self, dataset):
        '''
        Computing feature vector for images in dataset
        vectors = vectors type of {'image_path': Path to image,
                                    'class_image': Class of an image,
                                    'feature_vector': Feature vector of image}
        '''
        print("Computed feature vectors of dataset: False. \nComputing...")
        self.res_model.eval()
        # Check directory for metadata
        if not os.path.exists(self.metadata_dir):
            os.makedirs(self.metadata_dir)
        vectors = []

        # Create list of vectors
        df_data = dataset.get_data()
        # set_trace()
        for image_temp in tqdm(df_data.itertuples()):
            try:
                # print(image_temp.image_path)
                # temp="metadata/taibei/艋舺龙山寺/0000050.jpg"
                # input = self.preprocces_image(temp)
                input = self.preprocces_image(image_temp.image_path)
                vectors.append({'image_path': image_temp.image_path,
                                'class_image': image_temp.class_image,
                                'feature_vector': self.compute_input(input)})
            except:
                print(image_temp.image_path)
                pass
        print('Complete.')
        return vectors

    def feature_vectors(self, database):
        '''
        :param database: Object of Database

        If we have computed feature vectors in .cpickle file to our dataset we load them,
        on the contrary we need to compute vectors.
        '''
        # set_trace()
        if os.path.exists((os.path.join(self.vectors_dir, self.model_type))):
            vectors = self.load_feature_vectors()
        else:
            vectors = self.compute_feature_vectors(database)
            # print(os.path.join(self.metadata_dir, self.model_type))
            cPickle.dump(vectors, open(os.path.join(self.vectors_dir, self.model_type), "wb", True))
        return vectors