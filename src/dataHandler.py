import os
import pandas as pd
import numpy as np
from tqdm import tqdm

class DataHandler():
    def __init__(self, raw_data_folder, window_data_folder, metadata_folder, second_window_size = 1):
        """
        Constructor de la classe DataHandler
        """
        self.labels_df = None
        self.data_per_sec = 128
        self.WINDOW_SIZE = second_window_size * self.data_per_sec
        self.raw_data_folder = raw_data_folder
        self.window_data_folder = window_data_folder
        self.metadata_folder = metadata_folder

    def read_labels_data(self):
        '''
        Llegeix les dades de les etiquetes (Excel) i les guarda com a atribut de l'objecte DataHandler com a pandas dataframe (self.labels_df).
        :param filepath: ruta del Excel
        :type filepath: str
        '''

        # read labels excel file
        pd_tmp= pd.read_excel(os.path.join(self.raw_data_folder, "df_annotation_full.xlsx"))
        
        # change PatID from chb01 to 1
        pd_tmp["PatID"] = pd_tmp["PatID"].apply(lambda x: int(x[3:5]))

        # multiply every value by 128 to get the index in the raw data
        pd_tmp["seizure_start"] = pd_tmp["seizure_start"].apply(lambda x: x*self.data_per_sec)
        pd_tmp["seizure_end"] = pd_tmp["seizure_end"].apply(lambda x: x*self.data_per_sec)

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
        print("Splitting data by recordings...")
        # list_recordings = [int(recording.split("_")[1][:-4]) for recording in list_recordings]
        for recording in tqdm(unique_recordings, total=len(unique_recordings)):
            recordings.append(pd_pacient[pd_pacient["filename"] == recording])
            #print("Recording: ", recording)
            #print("Recording shape: ", recordings[-1].shape)

        # remove last 3 columns
        recordings = [recording.iloc[:, :-3] for recording in recordings]

        pat_id = int(pd_pacient["filename"][0].split("_")[0][3:5])
        
        # slice labels by pacient
        labels_pacient = self.labels_df[self.labels_df["PatID"] == pat_id]
        periods = []
        seconds_discard = 30

        print(f"Generating windows for patient {pat_id}.\nWhich has {len(recordings)} recordings: {unique_recordings}")
        print("Splitting data by periods...")
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

        windows = []
        #windows_array = np.empty((0,WINDOW_SIZE))
        metadata = pd.DataFrame(columns=["id","label","pacient","index_inicial","periode","recording"])
        window_id = 0
        num_periods = 5
        n_period = 0

        print("Generating windows...")
        for n_period,period in tqdm(enumerate(periods), total=len(periods), desc="Generating windows"):
            # turn period to numpy array
            label = period[0]
            recording = period[1]
            period = period[2]

            # slice by windows
            for i in range(0,period.shape[0],self.WINDOW_SIZE):
                if i+self.WINDOW_SIZE <= period.shape[0]:
                    window = period[i:i+self.WINDOW_SIZE]
                    windows.append(window)
                    row = pd.DataFrame([[window_id,label,pat_id,i,n_period,recording]],columns=["id","label","pacient","index_inicial","periode","recording"])
                    metadata = pd.concat([metadata,row],ignore_index=True)
                    window_id += 1

            # debugging purposes
            n_period += 1
            if n_period == num_periods:
                break

        # turn windows to numpy array
        windows_array = np.array(windows)
        
        return windows_array, metadata

    def save_window_data(self,filename,window_array):
        """
        Guarda les windows i les metadades al folder

        :param folder: ruta del directori on guardarem la info
        :type folder: str
        """

        # if necessary create folder
        if not os.path.exists(self.window_data_folder):
            os.makedirs(self.window_data_folder)
        
        # save windows as compressed numpy array
        np.savez_compressed(os.path.join(self.window_data_folder, filename), window_array)
    
    def save_metadata(self,filename,metadata):
        """
        Guarda les windows i les metadades al folder

        :param folder: ruta del directori on guardarem la info
        :type folder: str
        """

        # if necessary create folder
        if not os.path.exists(self.metadata_folder):
            os.makedirs(self.metadata_folder)
        
        # save metadata as csv
        metadata.to_csv(os.path.join(self.metadata_folder,filename), index=False)

    def preprocess_data(self):
        """
        Preprocessa les dades raw del folder .
        Guarda les finestres i les metadades en el folder indicat.

        :param raw_data_folder: ruta del directori amb les dades raw
        :type raw_data_folder: str
        :param window_data_folder: ruta del directori on guardarem les finestres
        :type window_data_folder: str
        :param metadata_folder: ruta del directori on guardarem les metadades
        :type metadata_folder: str
        """

        # Read labels file
        self.read_labels_data()

        # get list of files ending with .parquet

        list_files = os.listdir(self.raw_data_folder)
        list_files = [file for file in list_files if file.endswith(".parquet")]
        
        print("Preprocessing data from folder: ", self.raw_data_folder)

        
        # iterate over files
        for file in list_files:
            print("Preprocessing file: ", file)

            # read raw data
            pd_pacient = self.read_raw_data(os.path.join(self.raw_data_folder, file))

            # generate windows
            windows_array, metadata = self.generate_windows(pd_pacient)

            print("Saving windows and metadata...")

            # save data
            self.save_window_data(file[:-8] + ".npz", windows_array)

            # save metadata
            self.save_metadata(file[:-8] + ".csv", metadata)
            
        

if __name__ == "__main__":
    # Test DataHandler
    dh = DataHandler()
    dh.read_labels_data("df_annotation_full.xlsx")
    pd_pacient = dh.read_raw_data("chb01_raw_eeg_128_full.parquet")
    windows_array,metadata = dh.generate_windows(pd_pacient)

    # Test save data
    dh.save_data("data","test_windows.csv",windows_array,metadata)

