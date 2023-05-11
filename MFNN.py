# Tensorflow / Keras
from tensorflow import keras  # for building Neural Networks

# for creating a linear stack of layers for our Neural Network
from keras.models import Sequential
from keras import Input  # for instantiating a keras tensor

# for creating regular densely-connected NN layer.
from keras.layers import Dense

import pandas as pd  # for data manipulation
import numpy as np  # for data manipulation

import sklearn  # for model evaluation

# for splitting the data into train and test samples
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report  # for model evaluation metrics
from imblearn.over_sampling import SMOTE  # for oversampling

from collections import Counter
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from MachineLearning import naive_bayes, xgboost, rdforest
from collections import defaultdict


class MFNN:
    def __init__(self, config_file, data, target):
        """
        Initialize the neural network

        Parameters:
            config_file: path to the json configuration file (look at MFNN_config foler)
            data: pandas dataframe
            target: a string of target variable (e.g "PHQ9_binary")
        """
        config = json.load(open(config_file, "r"))
        self.name = config["name"]
        self.hyperparams = config["hyperparams"]

        self.data = data

        self.target = target
        self.model_config = config["model"]
        self.figures = config["figure"]
        self.model = self.build()

    def build(self):
        """Builds the Keras model based"""
        model = Sequential(name=self.name)  # Model
        # Input Layer - need to specify the shape of inputs
        model.add(Input(shape=(self.data.shape[1] - 1,), name="Input-Layer"))
        # Hidden Layer
        hiddens = self.model_config["hidden"]
        for hidden_layer in hiddens:
            model.add(
                Dense(
                    hidden_layer["units"],
                    activation=hidden_layer["activation"],
                    name=hidden_layer["name"],
                )
            )
        # Output Layer
        output = self.model_config["output"]
        model.add(
            Dense(output["units"], activation=output["activation"], name="Output-Layer")
        )

        learning_rate = self.hyperparams["learning_rate"]
        momentum = self.hyperparams["momentum"]
        decay = learning_rate / self.hyperparams["epochs"]
        optimizers = keras.optimizers.SGD(
            learning_rate=learning_rate, decay=decay, momentum=momentum
        )
        # Compile the model
        model.compile(
            optimizer=optimizers,  # self.hyperparams["optimizer"],
            loss=self.hyperparams["loss"],
            metrics=self.hyperparams["metrics"],
            # default=None, Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs.
            loss_weights=None,
            # default=None, List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
            weighted_metrics=None,
        )

        print(model.summary())
        return model

    def train(self):
        kf = KFold(n_splits=self.hyperparams["fold"])
        auc_maxes = {
            "Neural Network": [None, None, 0],
            "Naive Bayes": [None, None, 0],
            "XGBoost": [None, None, 0],
            "Random Forest": [None, None, 0],
        }

        # Convert target classes to categorical ones
        for i, (train_index, test_index) in enumerate(kf.split(self.data)):
            # if i > 0:
            #     break
            print("Fold {}".format(i + 1))
            Y = self.data[self.target]
            X = self.data.drop([self.target], axis=1)

            # oversampling on training data
            oversample = SMOTE(sampling_strategy=self.hyperparams["smote"])
            X_train, Y_train = X.iloc[train_index], Y.iloc[train_index]
            X_train, Y_train = oversample.fit_resample(X_train, Y_train)

            X_test, Y_test = X.iloc[test_index], Y.iloc[test_index]
            fit = self.model.fit(
                X_train,
                Y_train,
                batch_size=self.hyperparams["batch_size"],
                epochs=self.hyperparams["epochs"],
                verbose=0,
                shuffle=True,
                validation_data=(X_test, Y_test),
            )
            prediction = self.model.predict(X.iloc[test_index])
            y_pred = (prediction > 0.5).astype("int32")

            self.display_pred(y_pred)

            print("Actual:")
            print(Y.iloc[test_index].value_counts())
            print(classification_report(Y.iloc[test_index], y_pred))

            fpr, tpr, _ = roc_curve(Y_test, prediction)
            nn_auc = auc(fpr, tpr)
            if auc_maxes["Neural Network"][2] < nn_auc:
                auc_maxes["Neural Network"] = [fpr, tpr, nn_auc]

            self.plot_loss(fit, self.figures["loss_figure"])

            print("Naive Bayes:")
            nb_fpr, nb_tpr, nb_auc = naive_bayes(X_train, X_test, Y_train, Y_test)
            if auc_maxes["Naive Bayes"][2] < nb_auc:
                auc_maxes["Naive Bayes"] = [nb_fpr, nb_tpr, nb_auc]

            print("Random Forest:")
            rdf_fpr, rdf_tpr, rdf_auc = rdforest(X_train, X_test, Y_train, Y_test)
            if auc_maxes["Random Forest"][2] < rdf_auc:
                auc_maxes["Random Forest"] = [rdf_fpr, rdf_tpr, rdf_auc]

            print("XGBoost")
            xgb_fpr, xgb_tpr, xgb_auc = xgboost(X_train, X_test, Y_train, Y_test)
            if auc_maxes["XGBoost"][2] < rdf_auc:
                auc_maxes["XGBoost"] = [xgb_fpr, xgb_tpr, xgb_auc]

        self.plot_auc(auc_maxes, self.figures["auc_figure"])

    def plot_loss(self, fit, file):
        plt.figure(1)
        plt.plot(fit.history["loss"])
        plt.plot(fit.history["val_loss"], linestyle="dashed")
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(loc="upper right")
        if file is None:
            file = "loss.png"
        plt.savefig(file)

    def plot_auc(self, auc_maxes, file):
        plt.figure(2)
        for k, v in auc_maxes.items():
            plt.plot(v[0], v[1], label="ROC curve {} (area = {:.2f})".format(k, v[2]))
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        if file is None:
            file = "auc.png"
        plt.savefig(file)

    def display_pred(self, pred):
        print("Prediction:")
        unique, counts = np.unique(pred, return_counts=True)
        print(dict(zip(unique, counts)))
