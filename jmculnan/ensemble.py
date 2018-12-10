###ensemble models
import json
import numpy as np
from sklearn.metrics import f1_score, classification_report
from keras.models import load_model
from collections import Counter
import pickle

#take relevant files
x_testpath     = 'saved_dataRumEval2019_npy_files/dev/train_array.npy'
y_testpath     = 'saved_dataRumEval2019_npy_files/dev/fold_stance_labels.npy'
flatx_testpath = 'saved_dataRumEval2019_npy_files/dev/flat_dev_x.npy' #yiyun zhao
flaty_testpath = 'saved_dataRumEval2019_npy_files/dev/flat_dev_y.npy' #yiyun zhao
otherx_testpath= 'saved_dataRumEval2019_npy_files/dev/dev_x_preceding_thread.npy' #masha alexeeva
othery_testpath= 'saved_dataRumEval2019_npy_files/dev/dev_y_preceding_thread.npy' #masha alexeeva
tweet_idspath  = 'saved_dataRumEval2019_npy_files/dev/tweet_ids.npy'

#load relevant files
x_test     = np.load(x_testpath)
y_test     = np.load(y_testpath)
flatx_test = np.load(flatx_testpath)
flaty_test = np.load(othery_testpath)
otherx_test= np.load(otherx_testpath)
cnnx_test  = np.reshape(flatx_test, (np.shape(flatx_test)[0], np.shape(flatx_test)[1], 1)) #seongjin park


def ensemble(bilstm,svm,cnn,ffnn,x_test, flatx_test, y_test,
            flaty_test, batch_size=64):
    """
    Run an ensemble model using the models we have trained
    bilstm: a trained lstm model (saved as .h5)
    svm: a trained svm model (saved as .h5)
    cnn: a trained cnn model (this model is saved in a pickle)
    ffnn: a feedforward neural network model (saved as .h5)
    """
    ###load models
    bilstm_model = load_model(bilstm)
    svm_model    = pickle.load(open(svm, 'rb'))
    cnn_model    = load_model(cnn)
    ffnn_model   = load_model(ffnn)
    ###predict output for each model; models were created differently
    ##so formatting needs to be different
    bilstm_pred  = bilstm_model.predict_classes(x_test, batch_size)
    svm_pred     = svm_model.predict(otherx_test)
    cnn_predprob = cnn_model.predict(cnnx_test,batch_size)
    cnn_pred     = cnn_predprob.argmax(axis=1)
    ffnn_pred    = ffnn_model.predict_classes(flatx_test, batch_size)
    all_preds    = []

    #alter bilstm output to make compatible with other data
    ids_test = np.load(tweet_idspath)
    bilstm_preds = []
    all_ids      = []
    for i in range(len(ids_test)):
        for j in range(len(ids_test[i])):
            if ids_test[i][j] not in all_ids:
                all_ids.append(ids_test[i][j])
                bilstm_preds.append(bilstm_pred[i][j])
            else:
                continue
    bilstm_pred = bilstm_preds
    np.asarray(bilstm_pred)

    #ensure data is the same size
    if len(bilstm_pred) == len(svm_pred) and len(svm_pred) == len(ffnn_pred) and \
            len(ffnn_pred) == len(cnn_pred):
        ###put all predictions together
        for i in range(len(bilstm_pred)):
            all_preds.append([bilstm_pred[i], svm_pred[i], ffnn_pred[i], cnn_pred[i]])
        all_preds   = np.asarray(all_preds)
        final_preds = np.zeros(len(bilstm_pred)) #create vector to hold ensemble predictions
        #get class predictions
        for i in range(len(all_preds)):
            if 2 in all_preds[i]:
                final_preds[i] = 2 #hardcode to capture predictions of deny
            elif int(all_preds[i][0]) == 3  or int(all_preds[i][2]) == 3:
                final_preds[i] = 3 #hardcode to capture query predictions
            else:
                holder = Counter(all_preds[i]) #create dict for each set of preds
                held = [(v,k) for k,v in holder.items()] #return to list
                held = sorted(held, reverse=True) #sort by frequency
                if len(held) > 1: #only if multiple predictions are made
                    if held[0][0] == held[1][0]:
                        if held[0][1] == 0:
                            final_preds[i] = held[1][1] #don't select 0 class
                        elif held[1][1] == 0:
                            final_preds[i] = held[0][1] #don't select 0 class
                    else: #otherwise, select majority voted
                        final_preds[i] = held[0][1]
                else: #otherwise, select majority voted
                    final_preds[i] = held[0][1]
        #get a report of the classification by class
        print(classification_report(flaty_test, final_preds, labels=None,
                    target_names=None, sample_weight=None, digits=3))
        #calculate overall f1 scores
        mactest_F = f1_score(final_preds, flaty_test, average='macro')
        mictest_F = f1_score(final_preds, flaty_test, average='micro')
        print('microf1: ', mictest_F) #print these
        print('macrof1: ', mactest_F)
    else: #if data are not the same size, print error
        print("The data structures are not of the same size.\n \
             Bilstm predictions are of length: %d\n \
             SVM predictions are of length: %d\n \
             CNN predictions are of length: %d\n
             FFNN predictions are of length: %d" % (len(bilstm_pred), len(svm_pred),
             len(cnn_pred),len(ffnn_pred)))

#test this
ensemble('culnan_bilstm.h5','alexeeva_svm1.pickle','park_cnn.h5','zhao_ffnn.h5',x_test,flatx_test,y_test,flaty_test)
