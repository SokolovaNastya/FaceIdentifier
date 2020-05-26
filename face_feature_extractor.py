import os.path
import os
import time
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd

from sklearn import preprocessing
from scipy import misc
from insightface import InsightFaceModel

np.random.seed(123)

DATASET_PATH = '/home/datasets/images/MS1M/raw/'

img_extensions=['.jpg','.jpeg','.png']
def is_image(path):
    _, file_extension = os.path.splitext(path)
    return file_extension.lower() in img_extensions

def get_files(db_dir):
    return [[d,os.path.join(d,f)] for d in next(os.walk(db_dir))[1] for f in next(os.walk(os.path.join(db_dir,d)))[2] if not f.startswith(".") and is_image(f)]
    
#def get_files(db_dir):
#    return [[db_dir,os.path.join(db_dir,f)] for f in next(os.walk(db_dir))[2] if not f.startswith(".") and is_image(f)]

def load_graph(frozen_graph_filename, prefix=''):
    with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name=prefix)
    return graph


class TensorFlowInference:
    def __init__(self, frozen_graph_filename, input_tensor, output_tensor, learning_phase_tensor=None, convert2BGR=True,
                 imageNetUtilsMean=True, additional_input_value=0):
        graph = load_graph(frozen_graph_filename, '')
        print([n.name for n in graph.as_graph_def().node if 'input' in n.name])

        graph_op_list = list(graph.get_operations())
        print([n.name for n in graph_op_list if 'keras_learning' in n.name])

        self.tf_sess = tf.Session(graph=graph)

        self.tf_input_image = graph.get_tensor_by_name(input_tensor)
        print('tf_input_image=', self.tf_input_image)
        self.tf_output_features = graph.get_tensor_by_name(output_tensor)
        print('tf_output_features=', self.tf_output_features)
        self.tf_learning_phase = graph.get_tensor_by_name(learning_phase_tensor) if learning_phase_tensor else None;
        print('tf_learning_phase=', self.tf_learning_phase)
        if self.tf_input_image.shape.dims is None:
            w = h = 160
        else:
            _, w, h, _ = self.tf_input_image.shape
        self.w, self.h = int(w), int(h)
        print('input w,h', self.w, self.h, ' output shape:', self.tf_output_features.shape)

        self.convert2BGR = convert2BGR
        self.imageNetUtilsMean = imageNetUtilsMean
        self.additional_input_value = additional_input_value

    def preprocess_image(self, img_filepath, crop_center):
        if crop_center:
            orig_w, orig_h = 250, 250
            img = misc.imread(img_filepath, mode='RGB')
            img = misc.imresize(img, (orig_w, orig_h), interp='bilinear')
            w1, h1 = 128, 128
            dw = (orig_w - w1) // 2
            dh = (orig_h - h1) // 2
            img = img[dh:-dh, dw:-dw]
        else:
            img = misc.imread(img_filepath, mode='RGB')

        x = misc.imresize(img, (self.w, self.h), interp='bilinear').astype(float)

        if self.convert2BGR:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
            # Zero-center by mean pixel
            if self.imageNetUtilsMean:  # imagenet.utils caffe
                x[..., 0] -= 103.939
                x[..., 1] -= 116.779
                x[..., 2] -= 123.68
            else:  # vggface-2
                x[..., 0] -= 91.4953
                x[..., 1] -= 103.8827
                x[..., 2] -= 131.0912
        else:
            x /= 127.5
            x -= 1.
        return x

    def extract_features(self, img_filepath, crop_center=False):
        print(img_filepath)
        x = self.preprocess_image(img_filepath, crop_center)
        x = np.expand_dims(x, axis=0)
        feed_dict = {self.tf_input_image: x}
        if self.tf_learning_phase is not None:
            feed_dict[self.tf_learning_phase] = self.additional_input_value
        preds = self.tf_sess.run(self.tf_output_features, feed_dict=feed_dict).reshape(-1)
        return preds

    def close_session(self):
        self.tf_sess.close()


def extract_mxnet_features(model,img_filepath):
    print(img_filepath)
    img = cv2.imread(img_filepath)
    embeddings = model.get_feature(img)
    if embeddings is None:
        print(img_filepath)
    return embeddings


if __name__ == '__main__':

    # VGGFace2     -  0
    # InsightFace  -  1
	# MobileNet2   -  2
	# FaceNet      -  3
    model_name = 1
    features_file = 'tmp.npy'

    if model_name == 0:
        features_file = 'vggface2_feature.txt'
    elif model_name == 1:
        features_file = 'insightface_feature.txt'
    if model_name == 2:
        features_file = 'mobileNet2_feature.txt'
    elif model_name == 3:
        features_file = 'facenet_feature.txt'

    if not os.path.exists(features_file):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        df = pd.read_csv('classes_vggface2_ms1m_TheWorst_TheBest.csv')
        df['id'] = df['id'].astype(str)
        
        files = df['id'].values.tolist()
        
        #dirs_and_files = np.array(get_files(DATASET_PATH))
        #dirs = dirs_and_files[:, 0]
        #files = dirs_and_files[:, 1]

        if model_name != 1:
            if model_name == 0: # VGGFace2
				tfInference = TensorFlowInference('models/vggface2/vgg2_resnet.pb', input_tensor='input:0',
												output_tensor='pool5_7x7_s1:0', convert2BGR=True, imageNetUtilsMean=False)
			elif model_name == 2:# MobileNet
				tfInference=TensorFlowInference('models/mobilenet2/vgg2_mobilenet.pb',input_tensor='input_1:0',
												output_tensor='reshape_1/Reshape:0',
												learning_phase_tensor='conv1_bn/keras_learning_phase:0',
												convert2BGR=True, imageNetUtilsMean=True)
			elif model_name == 3: #Facenet
				tfInference = TensorFlowInference(
					'models/facenet_inceptionresnet/20180402-114759.pb',
					input_tensor='input:0', output_tensor='embeddings:0', learning_phase_tensor='phase_train:0',
					convert2BGR=False)								

            with open(features_file, 'a') as file:
                for filepath in files:
                    file.write(filepath)
                    file.write(',')
                    arr = tfInference.extract_features(os.path.join(DATASET_PATH, filepath), crop_center=False)
                    for el in arr:
                        file.write(str(el))
                        file.write(" ")
                    file.write('\n')

            tfInference.close_session()
        else:# InsightFace
            cnn_model = InsightFaceModel()
            
            with open(features_file, 'a') as file:
                for filepath in files:
                    file.write(filepath)
                    file.write(',')
                    arr = extract_mxnet_features(cnn_model, os.path.join(DATASET_PATH, filepath))
                    for el in arr:
                        file.write(str(el))
                        file.write(" ")
                    file.write('\n')