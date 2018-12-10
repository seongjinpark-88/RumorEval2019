import numpy as np
import os
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split

from keras.utils.np_utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF


# load the flatten training data
flat_train_x = np.load("flat_train_x.npy")
flat_train_y = np.load("flat_train_y.npy")

# load the flatten testing data
flat_dev_x = np.load("flat_dev_x.npy")
flat_dev_y = np.load("flat_dev_y.npy")


def deep_model(x_trn, y_trn):
    model = Sequential()
    model.add(Dense(units=100,activation="relu", input_dim = x_trn.shape[1]))
    model.add(Dense(units=70,activation="relu"))
    model.add(Dense(units=30,activation="relu"))
    model.add(Dense(units=4, activation = 'softmax'))
    model.summary()
    model.compile(optimizer = "adam",loss="categorical_crossentropy",metrics=['accuracy'])
    model.fit(x_trn,y_trn,batch_size=64,epochs=400,shuffle=True)
    return model

def wide_model(x_trn, y_trn):
    model = Sequential()
    model.add(Dense(units=128,activation="relu", input_dim = x_trn.shape[1])) 
    model.add(Dense(units=4, activation = 'softmax'))
    model.summary()
    model.compile(optimizer = "adam",loss="categorical_crossentropy",metrics=['accuracy'])
    model.fit(x_trn,y_trn,batch_size=64,epochs=400,shuffle=True)
    return model

def evaluate(x,y,md):
	prediction = md.predict(x)
	F1_1 = f1_score(np.argmax(prediction, 1),np.argmax(y, 1),average='macro')
	F1_2 = f1_score(np.argmax(prediction, 1),np.argmax(y, 1),average=None)
	accuracy = model.evaluate(x,y, verbose=0)[0]
	return F1_1, F2_2,accuracy


# split the test 
x1,x2,y1,y2 = train_test_split(flat_train_x,flat_train_y, test_size=0.3, random_state=4)

m_deep = deep_model(x1,y1,x2,y2)
m_wide = wide_model(x1,y1,x2,y2)

deep_dev = evaluate(x2,y2,m_deep)
deep_test = evaluate(flat_dev_x,flat_dev_y,m_deep)


wide_dev = evaluate(x2,y2,m_wide)
wide_test = evaluate(flat_dev_x,flat_dev_y,m_wide)

print("deep models evaluation on dev: ", "macro F1: ", deep_dev[0], "F1: ", deep_dev[1], "accuracy: ", deep_dev[2])
print("deep models evaluation on test: ", "macro F1: ", deep_test[0], "F1: ", deep_test[1], "accuracy: ", deep_test[2])

print("wide models evaluation on dev: ", "macro F1: ", wide_dev[0], "F1: ", wide_dev[1], "accuracy: ", wide_dev[2])
print("wide models evaluation on test: ", "macro F1: ", wide_test[0], "F1: ", wide_test[1], "accuracy: ", wide_test[2])



