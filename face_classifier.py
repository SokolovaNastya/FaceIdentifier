import numpy as np
import pickle
import pandas as pd
import sys
import math

from keras.models import load_model
from sklearn import preprocessing, model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from scipy import spatial
from keras.preprocessing.image import image
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


if __name__ == '__main__':
    features_file = 'vggface2_gen_feature.txt'
    model = load_model('my_model_vggface2.h5')
    print(model.summary())

    X_val = []
    y_l = []

    start_time = datetime.now()

    file = open(features_file, 'r')
    for line in file:
        line_split = line.split(',')
        n_tmp = line_split[0].split('/')[1]
        name = n_tmp.split('_')[0]

        line_X = line_split[1].split(' ')
        X_tmp = []
        features_number = len(line_X) - 1
        for el in line_X[:features_number]:
            X_tmp.append(float(el))
        
        X_tmp_tmp = X_tmp
        pred = model.predict_proba(np.reshape(X_tmp_tmp, (1,features_number)))[0][0]
        
        if any(el in line_split[0] for el in bad_arr):
            print("bad")
        else:
            if pred > 0.5:
                y_l.append(name)
                X_val.append(X_tmp)


    X = np.array(X_val)
    X_norm = preprocessing.normalize(X, norm='l2')

    y = np.array(y_l)

    classifier = KNeighborsClassifier(1)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_norm, y, test_size=0.5, random_state=42)
    
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)
    print('train classes:', len(np.unique(y_train)))
    classifier.fit(X_train, y_train)
    y_test_pred = classifier.predict(X_test)
    acc = 100.0 * (y_test == y_test_pred).sum() / len(y_test)
    print('acc=', acc)
    
    finish_time = datetime.now()
    time_total = finish_time - start_time
    print('Time: ', time_total)

    classifier_results_file = 'KNeighborsClassifier_vggface2_all.npy'
    np.savez(classifier_results_file, x=X_test, y=y_test, z=y_test_pred)

    filename = 'KNeighborsClassifier_vggface2_all.sav'
    pickle.dump(classifier, open(filename, 'wb'))
