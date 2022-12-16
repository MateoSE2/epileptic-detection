import numpy as np
import os
import pandas as pd
import scipy.stats as stats


class FeaturesPatient():
    def __init__(self, path_metadata, directorio_data, df_features=None, list_seizure_atacks=None):
        self.metadata= pd.read_csv(path_metadata)
        data= np.load(directorio_data)
        self.data= list(data['arr_0'])
        self.df_features= df_features
        self.list_seizure_atacks= list_seizure_atacks


    def calculate_window_features(self, window_values, channels=21):
        for c in range(channels):
            col_canal= window_values[:,c]

            mean = np.mean(col_canal)
            std = np.std(col_canal)
            kurtosis = stats.kurtosis(col_canal)
            skewness = stats.skew(col_canal)
            median = np.median(col_canal)
            entropy= stats.entropy(col_canal)
            min_value= np.min(col_canal)
            max_value= np.max(col_canal)
            power_energy= np.sum(np.power((col_canal),2))

        return mean, std, median, kurtosis, skewness, entropy, min_value, max_value, power_energy

    def load_non_seizure_windows(self):
        df_features = pd.DataFrame()

        non_seizure_metadata= self.metadata[self.metadata['label']== 0] #Que no haya ataque
        print('# non seizure windows:', non_seizure_metadata.shape[0])

        nrows= non_seizure_metadata.shape[0]
        label= 0 #no hay ataque

        for index in range(nrows):
            metadata_non_seizure_window= non_seizure_metadata.iloc[index]
            id_non_seizure_window= metadata_non_seizure_window['id']

            data_non_seizure_window= self.data[id_non_seizure_window]
            
            mean, std, median, kurtosis, skewness, entropy, min_value, max_value, power_energy= self.calculate_window_features(data_non_seizure_window)

            row = {"mean": mean, "std":std, "median":median, "kurtosis":kurtosis, "skewness":skewness, "entropy":entropy, "min_value":min_value, "max_value":max_value, "power_energy":power_energy, "label":label}

            df_features = df_features.append(row, ignore_index=True)


        self.df_features= df_features #label = 0

    def load_seizure_periods(self):
        seizure_metadata= self.metadata[self.metadata['label']== 1] #Que haya ataque


        nrows= seizure_metadata.shape[0]
        id_prev= None

        list_seizure_atacks = []

        for index in range(nrows):
            metadata_seizure_window= seizure_metadata.iloc[index]
            id_seizure_window= metadata_seizure_window['id']

            data_seizure_window= self.data[id_seizure_window]

            if id_prev== None:
                data_window_prev = data_seizure_window
                id_prev = id_seizure_window

            elif id_prev == (id_seizure_window - 1): #if nrows=40 -> (5120, 21)
                data_window_prev= np.vstack((data_window_prev, data_seizure_window))
                id_prev = id_seizure_window

            elif id_prev!= (id_seizure_window - 1):
                list_seizure_atacks.append(data_window_prev)
                data_window_prev = data_seizure_window
                id_prev = id_seizure_window
        

            if index == (nrows-1):
                list_seizure_atacks.append(data_window_prev)

        self.list_seizure_atacks= list_seizure_atacks

    def Windows_DataAumentation(self, k=1):
        index_inicial = 0
        index_final = 128
        label = 1

        number_windows= 0

        for seizure_period in self.list_seizure_atacks:
            while index_final < seizure_period.shape[0]:
                windows_data_aumentation= seizure_period[index_inicial: index_final, :]

                mean, std, median, kurtosis, skewness, entropy, min_value, max_value, power_energy= self.calculate_window_features(windows_data_aumentation)

                row = {"mean": mean, "std":std, "median":median, "kurtosis":kurtosis, "skewness":skewness, "entropy":entropy, "min_value":min_value, "max_value":max_value, "power_energy":power_energy, "label":label}

                self.df_features = self.df_features.append(row, ignore_index=True)


                index_inicial += k
                index_final += k
                number_windows+= 1


        return number_windows

    def save_features_data(self, filename, folder):
        """
        Guarda les windows i les metadades al folder

        :param folder: ruta del directori on guardarem la info
        :type folder: str
        """

        # if necessary create folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # save metadata as csv
        self.df_features.to_csv(os.path.join(folder, filename), index=False)
            

path_metadata= '/home/mapiv05/epileptic-detection/data/metadata/'
directorio_data= '/home/mapiv05/epileptic-detection/data/windows_data/'

path_features= '/home/mapiv05/epileptic-detection/data/features_data/'
path_file_txt= '/home/mapiv05/epileptic-detection/data/logs_features/'


metadata_path_files= os.listdir(path_metadata)
if '.DS_Store' in metadata_path_files:
    metadata_path_files.remove('.DS_Store')

data_path_files= os.listdir(directorio_data)
if '.DS_Store' in data_path_files:
    data_path_files.remove('.DS_Store')


for count, patient_data in enumerate(data_path_files):
    name_patient= patient_data.split('_')[0]

    file = open(path_file_txt+f'log_{name_patient}.txt', "w")
    file.write(f'--------------- Log Features patient {name_patient} ---------------\n')
    file.write('------------------ Start Execution ------------------  \n\n')

    patient_metadata= name_patient+'_raw_eeg_128.csv' #Debería existir

    if patient_metadata in metadata_path_files: #comprueba que existe
        file.write(f'Exists data and metadata of patient {name_patient}\n')

        features_patient= FeaturesPatient(path_metadata+patient_metadata, directorio_data+patient_data)
        features_patient.load_non_seizure_windows()

        file.write(f'   - # non seizure windows: {features_patient.df_features.shape[0]}\n')


        list_seizure_atacks= features_patient.load_seizure_periods()
        number_windows= features_patient.Windows_DataAumentation(1)
        file.write(f'   - # seizure windows with Data Augmentation: {number_windows}\n')

        features_patient.save_features_data(f"features_data_{name_patient}.csv", path_features)
        file.write(f'   - dim of final pandas: {features_patient.df_features.shape}\n')

    else:
        file.write(f"ERROR: doesn't create features of patient {name_patient}\n")

    file.write('\n------------------  End Execution ------------------ ')
    file.close()

