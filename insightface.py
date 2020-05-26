from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import mxnet as mx
import cv2
import sklearn
import face_preprocess


def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx,:,:])

class InsightFaceModel:
    def __init__(self):
        image_size = 112
        self.image_size = '{},{}'.format(image_size,image_size)
        self.flip = 0
        model_path = '/home/student/asokolova/models/model-r100-ii/model,0'
        _vec = model_path.split(',')
        prefix = _vec[0]
        epoch = int(_vec[1])
        print('loading',prefix, epoch)
        #ctx = mx.gpu(0)
        ctx = mx.cpu()
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
        model.bind(data_shapes=[('data', (1, 3, image_size, image_size))])
        model.set_params(arg_params, aux_params)
        self.model = model


    def get_feature(self, face_img):
        nimg = face_preprocess.preprocess(face_img, None, None, image_size=self.image_size)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2,0,1))
        embedding = None
        for flipid in [0,1]:
            if flipid==1:
                if self.flip==0:
                    break
                do_flip(aligned)
            input_blob = np.expand_dims(aligned, axis=0)
            data = mx.nd.array(input_blob)
            db = mx.io.DataBatch(data=(data,))
            self.model.forward(db, is_train=False)
            _embedding = self.model.get_outputs()[0].asnumpy()
            if embedding is None:
                embedding = _embedding
            else:
                embedding += _embedding

        embedding = sklearn.preprocessing.normalize(embedding).flatten()
        return embedding