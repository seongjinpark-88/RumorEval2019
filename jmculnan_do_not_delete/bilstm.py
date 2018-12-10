"""
Build a Bi-directional LSTM to train on the data.
Input is assumed to be pre-processed.
"""

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.layers import TimeDistributed, Masking
from keras import optimizers
from keras import regularizers
from sklearn.metrics import f1_score, classification_report
import math
from keras.utils.np_utils import to_categorical
from hyperopt import STATUS_OK
import json

#path to data
trainxpath      = 'saved_dataRumEval2019_npy_files/train/train_array.npy'
task1trainypath = 'saved_dataRumEval2019_npy_files/train/fold_stance_labels.npy'
testxpath       = 'saved_dataRumEval2019_npy_files/dev/train_array.npy'
task1testypath  = 'saved_dataRumEval2019_npy_files/dev/fold_stance_labels.npy'
tweet_idspath   = 'saved_dataRumEval2019_npy_files/dev/tweet_ids.npy'


def load_data():
    """
    Loads dataset and prepares it for use with the task
    """
    x_train = np.load(trainxpath) #load training input
    x_test  = np.load(testxpath)  #load testing input
    y_t = np.load(task1trainypath)#load training labels
    #take training labels and transform them to categorical
    y_train = []
    for i in range(len(y_t)):
        y_train.append(to_categorical(y_t[i], num_classes=4))
    y_train = np.asarray(y_train)
    #load testing labels
    y_test  = np.load(task1testypath)
    #shuffle the data by applying the same permutation to x and y
    p = np.random.permutation(len(x_train))
    x_train, y_train = x_train[p], y_train[p]
    return x_train,y_train,x_test,y_test #return all data

def prep_cv(xset,k=5):
    """
    Prepare for cross-validation
    xset: dataset to be used for training/dev
    k:    number of folds in CV
    """
    #split data into folds for cv
    splitspot = math.floor(len(xset)/k)
    return splitspot #return index for splitting into folds

#create bi-LSTM
def BiLSTM_model_stance(x_train, y_train, x_test, model_name='bilstm_stance.h5',
        num_lstm_units=100,num_lstm_layers=2, num_dense_layers=1, num_dense_units=200,
        num_epochs=50,learn_rate=1e-3,mb_size=64,l2reg=3e-4):
    """
    Create a bi-directional LSTM model and run it on the dataset
    x_train: a set of training data
    y_train: a set of training gold labels
    x_test: a set of testing data
    model_name: name model is saved under. must have extension .h5
    other parameters: hyperparameters. alter these to tune
    """
    #create model
    model = Sequential()
    num_features = x_train.shape[2]
    #zero pad the data to length of maximum thread
    model.add(Masking(mask_value=0., input_shape=(None, num_features)))
    #add biLSTM layers
    for nl in range(num_lstm_layers-1):
        model.add(Bidirectional(LSTM(num_lstm_units, kernel_initializer='glorot_normal',
                       dropout=0.2, recurrent_dropout=0.2,
                       return_sequences=True)))
    model.add(Bidirectional(LSTM(num_lstm_units, kernel_initializer='glorot_normal',
                   dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    #add dense layers
    model.add(TimeDistributed(Dense(num_dense_units, activation='relu')))
    for nl in range(num_dense_layers-1):
        model.add(TimeDistributed(Dense(num_dense_units, activation='relu')))
    model.add(Dropout(0.5))
    #add final layer with number of units equal to number of prediction classes
    model.add(TimeDistributed(Dense(4, activation='softmax',
                              activity_regularizer=regularizers.l2(l2reg))))
    #use an adaptive learning rate
    adam = optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999,
                           epsilon=1e-08, decay=0.0)
    #compile the model
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    #fit the model to the training data
    model.fit(x_train, y_train,
              batch_size=mb_size,
              epochs=num_epochs, shuffle=True, class_weight=None)
    #get probabilities of each class
    pred_probabilities = model.predict(x_test, batch_size=mb_size)
    confidence = np.max(pred_probabilities, axis=2)
    #get class predictions for the data set
    Y_pred = model.predict_classes(x_test, batch_size=mb_size)
    model.save(model_name) #save the model so it may be loaded for testing
    return Y_pred, confidence

#run the model with cross-validation
def run_models_with_cv(xtrain,ytrain,splitspot,k=5, model_name='bilstm_stance.h5',
        num_lstm_units=100,num_lstm_layers=2, num_dense_layers=1, num_dense_units=200,
        num_epochs=50,learn_rate=1e-3,mb_size=64,l2reg=3e-4):
    """
    Used to run data through cross-validation. May be used to tune hyperparameters
    prior to testing.
    xtrain: a set of training data
    ytrain: a set of training gold labels
    splitspot: number indicating the index for splitting folds
    """
    total_pred  = [] #prepare to collect all predictions
    total_testy = []
    #iterate over folds
    for i in range(k):
        #get test partition
        x_test = xtrain[splitspot*i:splitspot*(i+1)]
        y_test = ytrain[splitspot*i:splitspot*(i+1)]
        #make copy of dataset and remove test partition
        x_train = np.copy(xtrain[:])
        x_train = np.delete(x_train, np.arange(splitspot*i,splitspot*(i+1)),0)
        y_train = np.copy(ytrain[:])
        y_train = np.delete(y_train, np.arange(splitspot*i,splitspot*(i+1)),0)
        #call the model on training partitions
        #adapted from objective_functions.py
        y_pred, confidence = BiLSTM_model_stance(x_train,y_train,x_test,model_name,
                            num_lstm_units,num_lstm_layers,num_dense_layers,num_dense_units,
                            num_epochs,learn_rate,mb_size,l2reg)
        fids_test = []
        ids_test = np.load(tweet_idspath)
        for i in ids_test:
            fids_test.extend(i)
        fy_pred = y_pred.flatten()
        fy_test = y_test.flatten()
        uniqtwid, uindices2 = np.unique(fids_test, return_index=True)
        uniqtwid = uniqtwid.tolist()
        uindices2 = uindices2.tolist()
        uniq_dev_prediction = [int(fy_pred[i]) for i in uindices2]
        uniq_dev_label = [int(fy_test[i]) for i in uindices2]
        total_pred.extend(uniq_dev_prediction)
        total_testy.extend(uniq_dev_label)
    #get macro and micro f1 scores on all the data after CV is completed
    print(classification_report(total_testy, total_pred, labels=None,
                target_names=None, sample_weight=None, digits=3))
    mactest_F = f1_score(total_pred, total_testy, average='macro')
    mictest_F = f1_score(total_pred, total_testy, average='micro')
    print('macro F1 equals: ',mactest_F)
    print('micro F1 equals: ',mictest_F)
    output = {'loss': 1-mactest_F,
              'macrof1': mactest_F,
              'microf1': mictest_F,
              'status': STATUS_OK,
              'attachments': {'ID':uniqtwid,
                              'Predictions':total_pred,
                              'Actual':total_testy,
                              'Labels':uniq_dev_label}}
    return output

###run all the above code
def run_bilstm(saved_info='answer',k=5,model_name='bilstm_stance.h5',
        num_lstm_units=100,num_lstm_layers=2, num_dense_layers=1, num_dense_units=200,
        num_epochs=50,learn_rate=1e-3,mb_size=64,l2reg=3e-4):
    x_train,y_train,_,_ = load_data()
    splitspot = prep_cv(x_train)
    answer = run_models_with_cv(x_train,y_train,splitspot,k,model_name,num_lstm_units,
                num_lstm_layers,num_dense_layers,num_dense_units,num_epochs,learn_rate,
                mb_size,l2reg)
    with open(saved_info + ".json", 'w') as f:
        json.dump(answer, f)

def test_created_model(modelfile,ytest,mb_size=64):
    model = load_model(modelfile)
    testx = np.load(testxpath)
    pred_probabilities = model.predict(testx,mb_size)
    confidence = np.max(pred_probabilities, axis=2)
    y_pred = model.predict_classes(testx, mb_size)
    y_pred = y_pred.tolist()
    with open('testpredshape.json','w') as tf:
        json.dump(y_pred,tf)
    y_pred = np.asarray(y_pred)

    y_test = []
    for i in range(len(ytest)):
        y_test.append(to_categorical(ytest[i], num_classes=4))
    y_test = np.asarray(y_test)
    ids_test = np.load(tweet_idspath)
    #testing ways to get f1 accurately
    better_preds = []
    better_testy = []
    all_ids      = []
    for i in range(len(ids_test)):
        for j in range(len(ids_test[i])):
            if ids_test[i][j] not in all_ids:
                all_ids.append(ids_test[i][j])
                better_preds.append(y_pred[i][j])
                better_testy.append(ytest[i][j])
            else:
                continue
    mactest_F = f1_score(better_preds, better_testy, average='macro')
    mictest_F = f1_score(better_preds, better_testy, average='micro')
    print(classification_report(better_testy, better_preds, labels=None,
                target_names=None, sample_weight=None, digits=3))
    print('microf1: ', mictest_F)
    print('macrof1: ', mactest_F)
    answerdict = {}
    output = answerdict
    with open("test_output.json", 'w') as f:
        json.dump(output, f)
    return output

####uncomment the code below in order to run the model
test_task1 = np.load(task1testypath)
test_created_model('culnan_bilstm.h5', test_task1)
#run_bilstm('bilstmoutput')
