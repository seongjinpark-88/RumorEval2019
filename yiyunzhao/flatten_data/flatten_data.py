'''
@ author: Yiyun Zhao
Function: the current script is used to flatten the branchy data used in the paper
the original model has a branchy structure e.g.; [source1, comment1, comment2] [source1 , comment3]
the current script is used to flatten the data structure to [source, comment1, comment2, comment3] by singling out each unique post in the chain


source files: 
saved_dataRumEval2019_npy_files/train/train_array.npy
saved_dataRumEval2019_npy_files/train/fold_stance_labels.npy
saved_dataRumEval2019_npy_files/dev/train_array.npy
saved_dataRumEval2019_npy_files/dev/fold_stance_labels.npy
saved_dataRumEval2019_npy_files/dev/tweet_ids.npy
saved_dataRumEval2019_npy_files/train/tweet_ids.npy


outputfiles:
flat_train_x.npy
flat_train_y.npy
flat_dev_x.npy
flat_dev_y.npy

'''

cwd = os.getcwd()
x_train = np.load(os.path.join(cwd,'saved_dataRumEval2019_npy_files/train/train_array.npy'))
y_train = np.load(os.path.join(cwd,'saved_dataRumEval2019_npy_files/train/fold_stance_labels.npy'))
x_test = np.load(os.path.join(cwd,'saved_dataRumEval2019_npy_files/dev/train_array.npy'))
y_test = np.load(os.path.join(cwd,'saved_dataRumEval2019_npy_files/dev/fold_stance_labels.npy'))
ids_test = np.load(os.path.join(cwd,'saved_dataRumEval2019_npy_files/dev/tweet_ids.npy'))
ids_train = np.load(os.path.join(cwd,'saved_dataRumEval2019_npy_files/train/tweet_ids.npy'))

# flatten the training data
DATA_DIC = {}
Xs =[]
Ys =[]
for t1, thread in enumerate(ids_train):
    for p1, post in enumerate(thread):
        if post not in DATA_DIC:
            DATA_DIC[post] = [x_train[t1][p1],y_train[t1][p1]]
            Xs.append(x_train[t1][p1])
            Ys.append(y_train[t1][p1])

np.save("flat_train_x", np.array(Xs))
np.save("flat_train_y", to_categorical(Ys))



# flatten the test data
DEV_DATA_DIC = {}
DEV_Xs =[]
DEV_Ys =[]
for t1, thread in enumerate(ids_test):
    for p1, post in enumerate(thread):
        if post not in DEV_DATA_DIC:
            DEV_DATA_DIC[post] = [x_test[t1][p1],y_test[t1][p1]]
            DEV_Xs.append(x_test[t1][p1])
            DEV_Ys.append(y_test[t1][p1])


np.save("flat_dev_x", np.array(DEV_Xs))
np.save("flat_dev_y", to_categorical(dev_Ys))