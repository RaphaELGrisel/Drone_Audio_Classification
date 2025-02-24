import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score,recall_score, f1_score

class Evaluate():
    def __init__(self,model,test_ds,class_names=None):
        self.model = model
        self.test_ds = test_ds
        self.class_names=class_names

    def accuracy(self):
        self.test_ds = self.test_ds.cache().prefetch(tf.data.AUTOTUNE)
        results = self.model.evaluate(self.test_ds, return_dict=True)
        print(f"Model accuracy on test dataset: {results['accuracy']}")
        y_pred = self.model.predict(self.test_ds)
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.concat(list(self.test_ds.map(lambda s,lab: lab)), axis=0)

        acc = accuracy_score(y_true,y_pred)
        print(f"Model accuracy on test dataset : {acc}")
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")

        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")

        return y_true, y_pred, precision, recall, f1
    
    def conf_matrix(self,y_true,y_pred):
        confusion_matrix = tf.math.confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10,8))
        sns.heatmap(confusion_matrix,
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.xlabel("Prediction")
        plt.ylabel("Label")
        plt.show()


    