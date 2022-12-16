#!/bin/sh
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
#import wandb
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import time


class SVM():
    def __init__(self, path_content, data=None):
        self.path_content= path_content
        self.data= pd.DataFrame()

    def load_patient_features(self):
        FeaturesData_path_files= os.listdir(self.path_content)
        if '.DS_Store' in FeaturesData_path_files:
            FeaturesData_path_files.remove('.DS_Store')

        for features_path_patient in FeaturesData_path_files:
            path_csv= self.path_content+features_path_patient

            data = pd.read_csv(self.path_content+features_path_patient)

            if not data.empty:
                if self.data.empty: #primera iter
                    self.data= data

                else: #concatenamos todos los datos en un pandas
                    self.data= pd.concat([self.data, data], sort= False)
            else:
                print(f'Empty file: {features_path_patient}\n')

    def train_model(self):
        start_time = time.time()

        a = self.data[["mean", "std", "median", "kurtosis", "skewness", "entropy", "min_value", "max_value"]]
        X = a.to_numpy()
        X[~np.isfinite(X)] = 0
        y = self.data["label"].to_numpy()

        # k-fold cross validation
        clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
        score = cross_val_score(clf, X, y, cv=5, scoring='f1').mean()
        print(f"\nModel F1 socre = {score:0.4f}\n")

        self.model = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # fit the clf
        self.model.fit(self.X_train, self.y_train)

        self.time_train= (time.time() - start_time)


    def test_model(self, path_file_txt):
        start_time = time.time()

        self.ground_truth = self.y_test
        self.predictions = self.model.predict(self.X_test)

        self.time_test= (time.time() - start_time)

        file = open(path_file_txt, "w")
        file.write('------ RESULTATS del model svm ------')
        file.write(f'\n time train execution: {round(self.time_train, 3)} s')
        file.write(f'\n time test execution: {round(self.time_test, 3)} s')
        file.write(f'\n accuracy score: {accuracy_score(self.ground_truth, self.predictions)}')
        file.close()


    def generate_confusion_matrix(self, path_folder):
            #Creem la matriu de Confusió
            skplt.metrics.plot_confusion_matrix(self.ground_truth, self.predictions, normalize=True)
            plt.savefig(path_folder+'Confusion_Matrix.png') #save curves

    def generate_roc_curve(self, path_folder):
        #Creem la roc curve
        fpr, tpr, _ = roc_curve(self.ground_truth, self.predictions)
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.savefig(path_folder+'Roc_Curve.png')


    def generate_precision_recall(self, path_folder):
        #Creem la Precision-Recall
        prec, recall, _ = precision_recall_curve(self.ground_truth, self.predictions)
        pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
        plt.savefig(path_folder+'Precision_Recall.png') #save curves


path_features= '/home/mapiv05/epileptic-detection/data/features_data/'

path_results_svm= '/home/mapiv05/epileptic-detection/data/results_svm/dim_windows/'
name_file_results= 'results.txt'

Svm= SVM(path_features)
Svm.load_patient_features()
Svm.train_model()
Svm.test_model(path_results_svm+name_file_results)

Svm.generate_confusion_matrix(path_results_svm)
Svm.generate_roc_curve(path_results_svm)
Svm.generate_precision_recall(path_results_svm)

