import numpy as np
import os
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split


class CreateMetadataDim():
    def __init__(self, path_metadata = None, path_data = None, path_metadata_with_split = None, path_data_with_split = None, dim = None):
        self.path_metadata= path_metadata
        self.path_data= path_data
        self.path_metadata_with_split= path_metadata_with_split
        self.path_data_with_split= path_data_with_split
        self.dim= dim

    def load_all_metadata(self):
        metadata_path_files= os.listdir(self.path_metadata)
        if '.DS_Store' in metadata_path_files:
            metadata_path_files.remove('.DS_Store')

        for count, patient_metadata in enumerate(metadata_path_files):
            print(f'Exists file: {patient_metadata}\n')
            path_patient_metadata= path_metadata+patient_metadata

            metadata_patient = pd.read_csv(path_patient_metadata)

            if not metadata_patient.empty:
                if count== 0:
                    self.df_metadata= metadata_patient

                else: #concatenamos todos los datos en un pandas
                    self.df_metadata= pd.concat([self.df_metadata, metadata_patient], sort= False)
            else:
                print(f'Empty file: {patient_metadata}\n')

    def group_by(self, df = None): 

        def GenerateList(grouped_df):
            columns= list(grouped_df.columns.values)[:-1]
            dicc= {}

            for col in columns:
                single_column = grouped_df[col].tolist()
                dicc[col]= single_column

            if self.dim== 'pacient':
                return list(set(dicc['pacient']))

            elif self.dim== 'recording':
                return list(zip(dicc['pacient'], dicc['recording']))

            elif self.dim== 'periode':
                return list(zip(dicc['pacient'], dicc['recording'], dicc['periode']))


        dimension= {'pacient': ['pacient'], 'recording': ['pacient', 'recording'], 'periode': ['pacient', 'recording', 'periode']}

        if df is None:
            metadata = self.df_metadata
        else:
            metadata = df
            self.df_metadata = df

        grouped_df = metadata.groupby(dimension[self.dim]
                                ).size().reset_index(name="count")

        self.valors = GenerateList(grouped_df)

    def split_valors(self, test_size = 0.2):
        self.training_values, self.testing_values = train_test_split(self.valors, test_size=test_size)
        
        print(f' train: {self.training_values} \n test:  {self.testing_values}')

    def get_train_test(self, tipus='training'):

        if tipus== 'training': values_list= self.training_values
        elif tipus== 'testing': values_list= self.testing_values

        if self.dim== 'recording':
            df_list = []
            for pacient,recording in values_list:
                metadata_patient = self.df_metadata[(self.df_metadata['pacient']== pacient) & (self.df_metadata['recording']== recording)]
                df_list.append(metadata_patient)
        elif self.dim== 'periode':
            df_list = []
            for pacient,recording,periode in values_list:
                metadata_patient = self.df_metadata[(self.df_metadata['pacient']== pacient) & (self.df_metadata['recording']== recording) & (self.df_metadata['periode']== periode)]
                df_list.append(metadata_patient)
        elif self.dim== 'pacient':
            df_list = []
            for pacient in values_list:
                metadata_patient = self.df_metadata[(self.df_metadata['pacient']== pacient)]
                df_list.append(metadata_patient)
        
        metadata = pd.concat(df_list, sort= False)

        return metadata
            



    def load_url_data(self, tipus='training', only_metadata=False):

        if tipus== 'training': values_list= self.training_values
        elif tipus== 'testing': values_list= self.testing_values

        #chb02_raw_eeg_128.npz
        begging_name_file= 'chb'
        end_name_file_data= '_raw_eeg_128.npz'
        end_name_file_metadata= '_raw_eeg_128.csv'

        if self.dim== 'pacient':
            pacients= values_list

        else:
            pacients= []
            for value in values_list:
                pacients.append(value[0])
            pacients= list(set(pacients))
                
        #chb02_raw_eeg_128.csv
        self.pacients_url={}
        for pacient in pacients:
            pacient_url={}
            name_pacient= str(pacient)
            if len(name_pacient)==1 : name_pacient= f'0{name_pacient}'

            path_metadata= self.path_metadata+begging_name_file+name_pacient+end_name_file_metadata
            pacient_url['metadata'] = path_metadata

            path_data= self.path_data+begging_name_file+name_pacient+end_name_file_data
            pacient_url['data'] = path_data

            self.pacients_url[pacient]= pacient_url


    def load_data_metadata(self, tipus='training'):

        if tipus== 'training': values_list= self.training_values
        elif tipus== 'testing': values_list= self.testing_values

        metadata_prev = pd.DataFrame()
        data_prev= []
        for count, pacient_values in enumerate(values_list):
            if count== 0:
                pacient_name_prev= None

            if self.dim== 'pacient':
                pacient_name= pacient_values

            elif self.dim== 'recording':
                pacient_name= pacient_values[0]
                recording= pacient_values[1]
            
            elif self.dim== 'periode':
                pacient_name= pacient_values[0]
                recording= pacient_values[1]
                periode= pacient_values[2]


            if pacient_name != pacient_name_prev:
                #Start load metadata and data of pacient_name   
                pacient_metadata_df= pd.read_csv(self.pacients_url[pacient_name]['metadata'])
                
                pacient_data= np.load(self.pacients_url[pacient_name]['data'])
                pacient_data= pacient_data['arr_0']

            if self.dim== 'pacient':
                filtro= (pacient_metadata_df.pacient == pacient_name)

            elif self.dim== 'recording':
                filtro= (pacient_metadata_df.pacient == pacient_name) & (pacient_metadata_df.recording == recording)
            
            elif self.dim== 'periode':
                filtro= (pacient_metadata_df.pacient == pacient_name) & (pacient_metadata_df.recording == recording) & (pacient_metadata_df.periode == periode)


            metadata_with_filter= pacient_metadata_df[filtro]
            
            nrows= metadata_with_filter.shape[0]

            list_seizure_atacks = []


            for index in range(nrows):
                metadata_window= metadata_with_filter.iloc[index]
                id_window= metadata_window['id']
                data_window= list(pacient_data[int(id_window)])

                data_prev.append(data_window)
                metadata_window['id']= index #li assignem la seva nova id en el nou pandas
                metadata_prev = metadata_prev.append(metadata_window, ignore_index=True)

            pacient_name_prev= pacient_name

        
        data_prev = np.array(data_prev)

        print(f'\nShape of {tipus} metadata {metadata_prev.shape}')
        self.save_metadata(metadata_prev, tipus)

        print(f'\nShape of {tipus} data {data_prev.shape} \n')
        self.save_data(data_prev, tipus)


        
    def save_data(self, data, tipus, filename= 'data.npz'):
        """
        Guarda les windows al folder
        """
        
        # save windows as compressed numpy array
        print(f'----- Saving data with dimension {self.dim}-----')
        name_filename= f'{tipus}_{filename}'
        np.savez_compressed(os.path.join(self.path_data_with_split, name_filename), data)
    
    def save_metadata(self, metadata, tipus, filename= 'metadata.csv'):
        """
        Guarda les metadades al folder
        """
        
        # save metadata as csv
        print(f'----- Saving metadata with dimension {self.dim}-----')
        name_filename= f'{tipus}_{filename}'
        metadata.to_csv(os.path.join(self.path_metadata_with_split, name_filename), index=False)
        
if __name__ == '__main__':
    path_metadata= '/home/mapiv05/epileptic-detection/data/prova/metadata/'
    path_data= '/home/mapiv05/epileptic-detection/data/prova/data/'
    path_split= '/home/mapiv05/epileptic-detection/data/split_train-test/'


    dimensions= ['pacient', 'recording', 'periode']


    for dim in dimensions:
        path_metadata_with_split= path_split + dim+"/metadata/"
        path_data_with_split= path_split+ dim +"/data/"

        m= CreateMetadataDim(path_metadata, path_data, path_metadata_with_split, path_data_with_split, dim)
        m.load_all_metadata()
        m.group_by()
        m.split_valors()

        m.load_url_data('training')
        m.load_data_metadata('training')

        m.load_url_data('testing')
        m.load_data_metadata('testing')
