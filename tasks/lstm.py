import d6tflow, luigi
from  tasks.preprocessing import TaskRuleProcessor, TaskVocabCreator, TaskPrepareXY, TaskTrainTestSplit
import logging

from keras.models import Sequential, Model, load_model, save_model
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.metrics import Recall, Precision
from collections import Counter

#Workaround for keras objects being not pickable
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

# Hotfix function
def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__

@d6tflow.inherits(TaskTrainTestSplit)
class TaskTrainLstm(d6tflow.tasks.TaskPickle):
    embedding_vecor_length = luigi.IntParameter(default=32)
    epochs = luigi.IntParameter(default=3)
    batch_size = luigi.IntParameter(default=64)
    num_lstm_cells = luigi.IntParameter(default=100)
    dropout_emb_lstm = luigi.FloatParameter(default=0.0)
    dropout_lstm_dense = luigi.FloatParameter(default=0.0)


    def requires(self):
        return self.clone(TaskTrainTestSplit)

    def run(self):
        print(f"###Running {type(self).__name__}")

        X_train = self.input()["X_train"].load()
        y_train = self.input()["y_train"].load()
        X_test = self.input()["X_test"].load()
        y_test = self.input()["y_test"].load()

        train_counter = Counter(y_train)
        test_counter = Counter(y_test)
        print(f"Feature Distribution: Train: {train_counter[1] *100/ len(y_train)}%, Test: {test_counter[1] *100/ len(y_test)}%")
        print(f"length Train: {len(X_train)}, length Test: {len(X_test)}")

        #WORKAROUND for keras models not being pickleable
        make_keras_picklable()

        if self.encode_type:
            max_length = self.window_size *2
        else:
            max_length = self.window_size
        num_top_words = self.max_vocab_size + 2 #since we normaly use 0..1000 for normal tokens and 1001 for unknown token, it needs +2?

        recall_metric = Recall()
        model = Sequential()
        model.add(Embedding(num_top_words, self.embedding_vecor_length, input_length=max_length))
        model.add(Dropout(self.dropout_emb_lstm))
        model.add(LSTM(self.num_lstm_cells))
        model.add(Dropout(self.dropout_lstm_dense))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', recall_metric, Precision()])
        print(model.summary())
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size)

        self.save(model)

from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score
import numpy as np
from utils.plotter import confusion_matrix, evaluate_model
from utils.plotter import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

@d6tflow.inherits(TaskTrainLstm, TaskTrainTestSplit)
class TaskEvaluateLstm(d6tflow.tasks.TaskPqPandas):

    def requires(self):
        return{"model": self.clone(TaskTrainLstm), "data": self.clone(TaskTrainTestSplit)}

    def run(self):
        print(f"###Running {type(self).__name__}")


        model = self.input()["model"].load()
        X_train = self.input()["data"]["X_train"].load()
        y_train = self.input()["data"]["y_train"].load()
        X_test = self.input()["data"]["X_test"].load()
        y_test = self.input()["data"]["y_test"].load()
        print(f"Length Train: {len(X_train)}, length Test {len(X_test)}")

        # Training predictions (to demonstrate overfitting)
        train_rf_predictions = (model.predict(X_train) > 0.5).astype("int32")
        train_rf_probs = model.predict(X_train)

        # Testing predictions (to determine performance)
        rf_predictions = (model.predict(X_test) > 0.5).astype("int32")
        rf_probs = model.predict(X_test)

        # Plot formatting
        plt.style.use('fivethirtyeight')
        plt.rcParams['font.size'] = 18


        evaluate_model(rf_predictions, rf_probs, y_test,  train_rf_predictions, train_rf_probs, y_train)


        # Confusion matrix
        cm = confusion_matrix(y_test, rf_predictions)
        plot_confusion_matrix(cm, classes = ['0', '1'],
                            title = 'Confusion Matrix', normalize=True)
        
        # save test result
        evaluation_results = pd.DataFrame(zip(X_test, y_test, rf_predictions), columns=["x", "ground_truth", "predicted"])
        self.save(evaluation_results)