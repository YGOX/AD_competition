import os, glob
import matplotlib
matplotlib.use('AGG')
import numpy as np
import keras
from keras.callbacks import Callback
from keras.utils import np_utils
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

class F1Callback(Callback):
    def __init__(self, X, Y, early_stopping_patience=40,
                 plateau_patience=20, reduction_rate=0.1,
                 stage='train', fold_n=0):
        super(Callback, self).__init__()
        self.X = X
        self.Y = Y
        self.history = []
        self.early_stopping_patience = early_stopping_patience
        self.plateau_patience = plateau_patience
        self.reduction_rate = reduction_rate
        self.class_names = ['0', '1', '2']
        self.history = [[] for _ in range(len(self.class_names) + 1)]
        self.fold_n = fold_n
        self.stage = stage
        self.best_f1_m = -float('inf')
        self.checkpoints_path = str(self.fold_n) + '/'
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)

    def compute_f1(self, y_true, y_pred):
        f1_m = 0
        print(f"\n{'#' * 30}\n")
        for class_i in range(len(self.class_names)):
            f1_class = f1_score(y_true[:, class_i], y_pred[:, class_i])
            f1_m += f1_class / len(self.class_names)
            print(f"F1 SCORE {self.class_names[class_i]}, {self.stage}: {f1_class:.3f}\n")
            self.history[class_i].append(f1_class)
        print(f"\n{'#' * 20}\n MACRO F1 SCORE, {self.stage}: {f1_m:.3f}\n{'#' * 20}\n")
        self.history[-1].append(f1_m)
        return f1_m

    def is_patience_lost(self, patience):
        if len(self.history[-1]) > patience:
            best_performance = max(self.history[-1][-(patience + 1):-1])
            return best_performance == self.history[-1][-(patience + 1)] and best_performance >= self.history[-1][-1]

    def early_stopping_check(self, f1_m):
        if self.is_patience_lost(self.early_stopping_patience):
            self.model.stop_training = True

    def model_checkpoint(self, f1_m, epoch):
        if f1_m > self.best_f1_m:
            # remove previous checkpoints to save space
            for checkpoint in glob.glob(os.path.join(self.checkpoints_path, 'cls_epoch_*')):
                os.remove(checkpoint)
            self.best_f1_m = f1_m
            self.model.save(os.path.join(self.checkpoints_path, f'cls_epoch_{epoch}_val_f1_{round(f1_m, 5)}.h5'))
            print(f"\n{'#' * 20}\nSaved new checkpoint\n{'#' * 20}\n")

    def reduce_lr_on_plateau(self):
        if self.is_patience_lost(self.plateau_patience):
            new_lr = float(keras.backend.get_value(self.model.optimizer.lr)) * self.reduction_rate
            keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print(f"\n{'#' * 20}\nReduced learning rate to {new_lr}.\n{'#' * 20}\n")

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X)
        y_pred = np_utils.to_categorical(np.argmax(y_pred, axis=1), 3)
        y_true = self.Y
        # estimate F1 SCORE
        f1_m = self.compute_f1(y_true, y_pred)

        if self.stage == 'val':
            # early stop after early_stopping_patience epochs of no improvement
            self.early_stopping_check(f1_m)

            # save a model with the best MACRO F1 SCORE in validation
            self.model_checkpoint(f1_m, epoch)

            # reduce learning rate on MACRO F1 SCORE plateau
            self.reduce_lr_on_plateau()

    def get_f1_history(self):
        return self.history