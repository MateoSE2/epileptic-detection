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
    def __init__(self, path_features_training, path_features_testing, dim):
        self.path_features_training= path_features_training
        self.path_features_testing= path_features_testing
        self.dim = dim

    def data_balance(self, data, tipus):
        print(f'\nDATA BALANCE del {tipus} de la dimensió {dim}')
        print(f'# windows data: {data.shape[0]}')
        data_non_seizure= data[data['label']== 0]
        data_seizure= data[data['label']== 1]

        count_non_seizure = data_non_seizure.shape[0]
        count_seizure = data_seizure.shape[0]

        print(f'  - # non seizure data: {count_non_seizure}')
        print(f'  - # seizure data: {count_seizure}')

        rows = abs(count_non_seizure - count_seizure)
        print(f'Difference: {rows}')

        if count_non_seizure > count_seizure:
            data_non_seizure= data_non_seizure.iloc[rows:]

        elif count_non_seizure < count_seizure:
            data_seizure= data_seizure.iloc[rows:]

        data_balancejada= pd.concat([data_non_seizure, data_seizure])
        count_new_non_seizure= (data_balancejada[data_balancejada['label']== 0]).shape[0]
        count_new_seizure= (data_balancejada[data_balancejada['label']== 1]).shape[0]
        
        print(f'# final data: {data_balancejada.shape[0]}')
        print(f'  - # non seizure windows: {count_new_non_seizure}')
        print(f'  - # seizure windows: {count_new_seizure}\n')

        return data_balancejada


    def load_patient_features(self):
        data_training = pd.read_csv(self.path_features_training)
        data_balance_training= self.data_balance(data_training, 'training')
        X_train= data_balance_training[["mean", "std", "median", "kurtosis", "skewness", "entropy", "min_value", "max_value"]]


        self.X_train = X_train.to_numpy()
        self.X_train[~np.isfinite(self.X_train)] = 0

        self.y_train= data_balance_training["label"].to_numpy()
        

        data_testing= pd.read_csv(self.path_features_testing)
        data_balance_testing= self.data_balance(data_testing,'testing')
        X_test= data_balance_testing[["mean", "std", "median", "kurtosis", "skewness", "entropy", "min_value", "max_value"]]
        self.X_test = X_test.to_numpy()
        self.X_test[~np.isfinite(self.X_test)] = 0
        
        self.y_test= data_balance_testing["label"].to_numpy()


    def train_model(self):
        start_time = time.time()

        self.model = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
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

dimensions= ['pacient', 'recording', 'periode']

for dim in dimensions:
    path_features= f'/ghome/mapiv05/epileptic-detection/data/features_100k/{dim}/'
    path_features_training= path_features+'training_features.csv'
    path_features_testing= path_features+'testing_features.csv'

    path_results_svm= f'/ghome/mapiv05/epileptic-detection/data/results_svm/{dim}/'
    name_file_results= 'results.txt'

    Svm= SVM(path_features_training, path_features_testing, dim)
    Svm.load_patient_features()
    Svm.train_model()
    Svm.test_model(path_results_svm+name_file_results)

    Svm.generate_confusion_matrix(path_results_svm)
    Svm.generate_roc_curve(path_results_svm)
    Svm.generate_precision_recall(path_results_svm)


