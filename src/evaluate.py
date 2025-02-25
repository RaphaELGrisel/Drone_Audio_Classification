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
        print("New confusion matrix")
        print("New confusion matrix")

        # Calcul de la matrice de confusion
        confu = confusion_matrix(y_true, y_pred)

        # Normalisation optionnelle pour obtenir des pourcentages
        confu_norm = (confu.astype("float") / confu.sum(axis=1)[:, np.newaxis])*100

        # Création de la figure
        plt.figure(figsize=(10, 8))
        
        # Heatmap avec normalisation
        sns.heatmap(confu_norm, 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names, 
                    annot=True, 
                    fmt=".2f",  # Affichage en pourcentage
                    cmap="Blues", 
                    linewidths=0.5, 
                    linecolor="gray",
                    cbar=True)

        # Labels et titre
        plt.xlabel("Prediction")
        plt.ylabel("Label")
        plt.title(f"Matrice de confusion - Accuracy : {np.trace(confu) / np.sum(confu):.2%}")

        # Affichage
        plt.show()

        