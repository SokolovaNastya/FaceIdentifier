import numpy as np
import pandas as pd
import keras

from sklearn import preprocessing, model_selection
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


BATCH_SIZE = 16

if __name__ == '__main__':
    df = pd.read_csv('classes_vggface2_vggface2_ms1m_TheWorst_TheBest_Gen.csv')
    df['Good_or_bad'] = df['Good_or_bad'].astype(str)

    df_map = df.set_index('id')['Good_or_bad'].to_dict()

    X_val = []
    y_l = []
    cnt = 0

    with open('features.txt', 'r') as f:
        for line in f:
            line_split = line.split(',')
            
            #n_tmp = line_split[0].split('/')[6]
            n_tmp = line_split[0].replace('.jpg', '')
            n_tmp = n_tmp.replace('.png', '')
            n_tmp = n_tmp.replace('/', '_')
            n_tmp = n_tmp.replace('-', '_')
            
            if n_tmp in df_map:
                y_l.append(df_map[n_tmp])
          
                line_X = line_split[1].split(' ')
                X_tmp = []
                features_number = len(line_X) - 1
                for el in line_X[:features_number]:
                    X_tmp.append(float(el))
                X_val.append(X_tmp)
                print(cnt)
                cnt = cnt + 1
            else:
                print(n_tmp)

    X = np.array(X_val)
    X_norm = preprocessing.normalize(X, norm='l2')

    y = np.array(y_l)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, init='uniform'))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=15, verbose=2, shuffle=True, validation_data=(X_test, y_test))
    
    print(history.history['accuracy'])
    
    y_pred = model.predict(X_test).round().astype(int)
    
    cnt = 0
    for ind in range(len(y_test)):
        if int(y_test[ind]) == y_pred[ind][0]:
            cnt = cnt + 1
    
    print(cnt)
    print(len(y_test))
    acc = cnt / len(y_test)
    print(acc)
    
    model.save('my_model.h5')