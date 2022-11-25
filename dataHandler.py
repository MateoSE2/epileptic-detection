import os
import pandas as pd
import numpy as np
from tqdm import tqdm

class DataHandler():
    def __init__(self):
        self.labels_df = None
        self.windows_matrix = None

    def read_labels_data(self, filepath):
        '''
        Llegeix les dades de les etiquetes (Excel) i les guarda com a atribut de l'objecte DataHandler com a pandas dataframe (self.labels_df).
        :param filepath: ruta del Excel
        :type filepath: str
        '''

        # read labels excel file
        pd_tmp= pd.read_excel(filepath) # conda install openpyxl
        
        # change PatID from chb01 to 1
        pd_tmp["PatID"] = pd_tmp["PatID"].apply(lambda x: int(x[3:5]))

        # multiply every value by 128 to get the index in the raw data
        pd_tmp["seizure_start"] = pd_tmp["seizure_start"].apply(lambda x: x*128)
        pd_tmp["seizure_end"] = pd_tmp["seizure_end"].apply(lambda x: x*128)

        self.labels_df = pd_tmp

    def read_raw_data(self, filepath):
        '''
        Llegeix les dades raw del filepath, elimina les últimes 3 columnes i ho carrega en un pandas dataframe.
        Les dades raw corresponen a un sol pacient amb tots els seus recordings concatenats.

        :param filepath: ruta del arxiu parquet
        :type filepath: str
        :return: pandas de los datos del paciente, pacient_name, llista de recordings
        :rtype: pandas dataframe, str, list
        '''
        pd_pacient = pd.read_parquet(filepath, engine='pyarrow') # conda install pyarrow

        # extract pacient name
        pacient_id = int(pd_pacient["filename"][0].split("_")[0][3:5])

        # extract recordings
        list_recordings = list(set(pd_pacient["filename"]))
        list_recordings = [int(recording.split("_")[1][:-4]) for recording in list_recordings]

        return  pd_pacient


    def generate_windows(self, pd_pacient):
        """
        Separa el senyal del pacient en diferents recordings.
        Per cada recording, el separa per periodes (normal o atac (0/1)).
        També per recording el separa per finestres de k segons (128 dades per segon).
        A mesura que es creen les finestres s'etiqueten amb les metadades (label, pacient, index_inicial, periode i recording).
    
        Finalment, les metadades en un pd dataframe de tal manera que cada fila sigui una finestra amb totes les seves metadades.

        :param pd_pacient: pd amb les dades dels recordings del pacient
        :type pd_pacient: pandas dataframe
        :param pacient_name: nom del pacient
        :type pacient_name: str
        :param list_recordings: llista els noms dels recordings
        :type list_recordings: list
        """

        # split dataframe by recordings
        recordings = []
        unique_recordings = list(set(pd_pacient["filename"]))
        # list_recordings = [int(recording.split("_")[1][:-4]) for recording in list_recordings]
        for recording in unique_recordings:
            recordings.append(pd_pacient[pd_pacient["filename"] == recording])
            print("Recording: ", recording)
            print("Recording shape: ", recordings[-1].shape)

        # remove last 3 columns
        recordings = [recording.iloc[:, :-3] for recording in recordings]

        pat_id = int(pd_pacient["filename"][0].split("_")[0][3:5])
        
        # slice labels by pacient
        labels_pacient = self.labels_df[self.labels_df["PatID"] == pat_id]
        periods = []
        seconds_discard = 30

        for i,(recording,labels) in enumerate(zip(recordings,labels_pacient.iterrows())):
            # turn recording to numpy array
            recording = recording.to_numpy()

            if labels[1]["type"] == "seizure":
                # split by seizure
                start_seizure = labels[1]["seizure_start"]
                end_seizure = labels[1]["seizure_end"]
                
                # slice preseizure, seizure and postseizure
                periods.append((0,i,recording[:start_seizure-(seconds_discard*128)]))
                periods.append((1,i,recording[start_seizure:end_seizure]))
                periods.append((0,i,recording[end_seizure+(seconds_discard*128):]))
            else:
                periods.append((0,i,recording))

        a = 0

        # create windows by periods
        WINDOW_SIZE = 128
        windows_array = np.empty((0,WINDOW_SIZE))
        metadata = pd.DataFrame(columns=["id","label","pacient","index_inicial","periode","recording"])
        window_id = 0

        for n_period,period in tqdm(enumerate(periods), total=len(periods), desc="Generating windows"):
            # turn period to numpy array
            label = period[0]
            recording = period[1]
            period = period[2]


            # slice by windows
            for i in range(0,period.shape[0],WINDOW_SIZE):
                if i+WINDOW_SIZE <= period.shape[0]:
                    window = period[i:i+WINDOW_SIZE]
                    windows_array = np.vstack((windows_array,window.T))
                    row = pd.DataFrame([[window_id,label,pat_id,i,n_period,recording]],columns=["id","label","pacient","index_inicial","periode","recording"])
                    metadata = pd.concat([metadata,row],ignore_index=True)
                    window_id += 1

            break # debugging purposes
        
        return windows_array, metadata

    def save_data(self,folder,filename,window_array,metadata):
        """
        Guarda les windows i les metadades al folder

        :param folder: ruta del directori on guardarem la info
        :type folder: str
        """

        # if necessary create folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        

        # save windows as compressed numpy array
        np.savez_compressed(os.path.join(folder, "windows.npz"), window_array)

        # save metadata as csv
        metadata.to_csv(os.path.join(folder,filename), index=False)
        

if __name__ == "__main__":
    # Test DataHandler
    dh = DataHandler()
    dh.read_labels_data("df_annotation_full.xlsx")
    pd_pacient = dh.read_raw_data("chb01_raw_eeg_128_full.parquet")
    windows_array,metadata = dh.generate_windows(pd_pacient)

    # Test save data
    dh.save_data("data","test_windows.csv",windows_array,metadata)

