import argparse
import os
import h5py
import pandas as pd

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import (Activation, Conv3D, MaxPooling3D, BatchNormalization, GlobalAveragePooling3D, GlobalMaxPooling3D, PReLU)
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils, multi_gpu_model
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split


def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
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

    return X, np_utils.to_categorical(Y, nb_classes)


def main():
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for AD recognition')
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--nclass', type=int, default=3)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.getcwd()), "AD/AD_data/")

    X, Y= loaddata(data_dir,'train_pre_data.h5','train_pre_label.csv', args.nclass)
    print('X_shape:{}\nY_shape:{}'.format(X.shape, Y.shape))

    # Define model
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(
        X.shape[1:]), border_mode='same', data_format="channels_first"))
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

    model.add(Conv3D(args.nclass, kernel_size=(1, 1, 1), border_mode='same', data_format="channels_first"))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    model.add(GlobalAveragePooling3D(data_format='channels_first'))
    model.add(Activation('softmax'))

    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
    model.summary()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=43)

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=args.batch,
                        epochs=args.epoch, verbose=1, shuffle=True)
    model.evaluate(X_test, Y_test, verbose=0)
    model_json = model.to_json()
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    with open(os.path.join(args.output, 'AD_3dcnnmodel.json'), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(args.output, 'AD_3dcnnmodel.hd5'))

    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', acc)
    plot_history(history, args.output)
    save_history(history, args.output)


if __name__ == '__main__':
    main()
