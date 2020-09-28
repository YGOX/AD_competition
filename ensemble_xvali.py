import argparse
import os
import h5py
import pandas as pd

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import (Activation, Conv3D, MaxPooling3D, BatchNormalization, GlobalAveragePooling3D, GlobalMaxPooling3D, Dropout)
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils, multi_gpu_model
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.python.ops import array_ops
from keras import backend as K
import tensorflow as tf
from tensorflow.python.ops import array_ops
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from callback import F1Callback
from keras_radam import RAdam

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def plot_history(history, result_dir, foldid):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'fold{}_accuracy.png'.format(foldid)))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'fold{}_loss.png'.format(foldid)))
    plt.close()


def save_history(history, result_dir, foldid):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'fold{}_result.txt'.format(foldid)), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))


def loaddata(data_dir, filename, csv_name, nb_classes):

    with h5py.File(data_dir+filename, 'r') as f:
        # List all groups
        a_group_key = list(f.keys())[0]
        # Get the data
        data = list(f[a_group_key])
    X= np.asarray(data, 'float32')
    label = pd.read_csv(data_dir + csv_name)
    Y = label['label'].tolist()
    #_ , c = np.unique(Y, return_counts=True)
    return X, np_utils.to_categorical(Y, nb_classes), np.asarray(Y)


def load_testdata(data_dir, filename):

    with h5py.File(data_dir+filename, 'r') as f:
        # List all groups
        a_group_key = list(f.keys())[0]
        # Get the data
        data = list(f[a_group_key])
    X= np.asarray(data, 'float32')

    return X


def ensemble_predictions(members, testX):
    # make predictions
    yhats = [model.predict(testX) for model in members]
    yhats = np.array(yhats)
    # sum across ensemble members
    summed = np.sum(yhats, axis=0)
    # argmax across classes
    result = np.argmax(summed, axis=1)
    return result

def focal_loss(classes_num, gamma=2., e=0.1):
    # classes_num contains sample number of each classes
    def focal_loss_fixed(target_tensor, prediction_tensor):
        '''
        prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor
        '''
        #1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
        one_minus_p = array_ops.where(tf.greater(target_tensor,zeros), target_tensor - prediction_tensor, zeros)
        FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))

        #2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [ total_num / ff for ff in classes_num ]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ ff/sum_ for ff in classes_w_t1 ]   #scale
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)

        #3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_mean(balanced_fl)

        #4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1-e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(prediction_tensor)/nb_classes, prediction_tensor)

        return fianal_loss
    return focal_loss_fixed


def evaluate_model(trainX, trainy, valX, valy, batch_size, n_epochs, out_path, n_class, foldid, cls_cont):
    # define model
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(
        trainX.shape[1:]), border_mode='same', data_format="channels_first"))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), border_mode='same', data_format="channels_first"))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), border_mode='same', data_format="channels_first"))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same', data_format="channels_first"))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same', data_format="channels_first"))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), border_mode='same', data_format="channels_first"))

    model.add(Conv3D(128, kernel_size=(3, 3, 3), border_mode='same', data_format="channels_first"))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv3D(128, kernel_size=(3, 3, 3), border_mode='same', data_format="channels_first"))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), border_mode='same', data_format="channels_first"))

    model.add(Conv3D(256, kernel_size=(3, 3, 3), border_mode='same', data_format="channels_first"))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv3D(256, kernel_size=(3, 3, 3), border_mode='same', data_format="channels_first"))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), border_mode='same', data_format="channels_first"))

    model.add(Conv3D(n_class, kernel_size=(1, 1, 1), border_mode='same', data_format="channels_first"))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    model.add(GlobalAveragePooling3D(data_format='channels_first'))
    model.add(Activation('softmax'))

    model.compile(loss=focal_loss(classes_num=cls_cont), optimizer=Adam(), metrics=['accuracy', f1_m, precision_m, recall_m])
    model.summary()
    # fit model

    model_json = model.to_json()
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    with open(os.path.join(out_path, 'fold{}.json'.format(foldid)), 'w') as json_file:
        json_file.write(model_json)

    model_name = os.path.dirname(os.getcwd()) + '/AD/' + os.path.join(out_path, 'fold{}.hd5'.format(foldid))
    checkpoint_fixed_name = ModelCheckpoint(model_name,
                                            monitor='val_loss', verbose=1, save_best_only=True,
                                            save_weights_only=True, mode='auto', period=1)

    EarlyStop = EarlyStopping(monitor='val_loss', patience=40)

    callbacks = [checkpoint_fixed_name, EarlyStop]
    #train_f1_m = F1Callback(trainX, trainy)
    #val_f1_m = F1Callback(valX, valy, early_stopping_patience=40,
    #                     plateau_patience=np.inf, reduction_rate=0.5, stage='val', fold_n=foldid)

    history = model.fit(trainX, trainy, validation_data=(valX, valy), batch_size=batch_size, callbacks=callbacks, epochs=n_epochs, verbose=1, shuffle=True)
    # evaluate the model
    loss, acc, f1_s, pre, rec= model.evaluate(valX, valy, verbose=0)
    #model.save_weights(os.path.join(out_path, 'fold{}.hd5'.format(foldid)))

    plot_history(history, out_path, foldid)
    save_history(history, out_path, foldid)

    return model, loss, acc, f1_s, pre, rec


def main():
    parser = argparse.ArgumentParser(
        description='ensemble 3D convolution for AD recognition')
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--nclass', type=int, default=3)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.getcwd()), "AD/AD_data/")

    X, Y_one_hot, Y= loaddata(data_dir,'train_pre_data.h5','train_pre_label.csv', args.nclass)
    print('X_shape:{}\nY_shape:{}'.format(X.shape, Y_one_hot.shape))
    A = load_testdata(data_dir, 'testa.h5')
    B = load_testdata(data_dir, 'testb.h5')
    print('A_shape:{}'.format(A.shape))
    print('B_shape:{}'.format(B.shape))

    n_folds = 10
    kfold = KFold(n_folds, True, 1)
    valloss, valacc, members, valf1, valpre, valrec = list(), list(), list(),list(), list(), list()
    fold_idx= 1
    for train_ix, val_ix in kfold.split(X):
        # select samples
        trainX, trainy = X[train_ix], Y_one_hot[train_ix]
        valX, valy = X[val_ix], Y_one_hot[val_ix]
        _, cls_count= np.unique(Y[train_ix], return_counts=True)
        # evaluate model
        model, val_loss, val_acc, val_f1, val_pre, val_rec = evaluate_model(trainX, trainy, valX, valy, batch_size=args.batch, n_epochs=args.epoch, out_path=args.output, n_class= args.nclass, foldid=fold_idx, cls_cont=cls_count)
        print('>%.3f' % val_acc)
        valacc.append(val_acc)
        valloss.append(valloss)
        members.append(model)
        valf1.append(val_f1)
        valpre.append(val_pre)
        valrec.append(val_rec)
        fold_idx += 1
    print('Estimated accuracy %.3f (%.3f)' % (np.mean(valacc), np.std(valacc)))
    print('Estimated recall %.3f (%.3f)' % (np.mean(valrec), np.std(valrec)))
    print('Estimated precision %.3f (%.3f)' % (np.mean(valpre), np.std(valpre)))
    print('Estimated f1 %.3f (%.3f)' % (np.mean(valf1), np.std(valf1)))

    test_a= ensemble_predictions(members, A)
    test_b= ensemble_predictions(members, B)

    testa_id= ['testa_{}'.format(i) for i in range(A.shape[0])]
    testb_id= ['testb_{}'.format(i) for i in range(B.shape[0])]

    df = pd.DataFrame({'testa_id': testa_id + testb_id, 'label': list(test_a) + list(test_b)})
    df.to_csv(os.path.join(args.output, 'test_result.csv'), index=False)


if __name__ == '__main__':
    main()
