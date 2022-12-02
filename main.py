# main script, run this to start the epileptic seizure detection system
# script arguments:
# mode: "preprocessing" or "classic_ml" or "deep_learning"
# raw_data_folder: folder containing the raw data files
# window_folder: folder to store the windowed data files
# metadata_folder: folder to store the metadata files

import os
import sys
from src.dataHandler import DataHandler

if __name__ == '__main__':

    # get the script arguments
    mode = sys.argv[1]
    raw_data_folder = sys.argv[2]
    window_folder = sys.argv[3]
    metadata_folder = sys.argv[4]

    # for testing
    #mode = "preprocessing"
    #raw_data_folder = "data/raw_example"
    #window_folder = "data/windows_data"
    #metadata_folder = "data/metadata"

    if mode == "preprocessing":
        # run the preprocessing
        data_handler = DataHandler(raw_data_folder, window_folder, metadata_folder, second_window_size=1)
        data_handler.preprocess_data(excepting = ["chb01_raw_eeg_128.npz", "chb22_raw_eeg_128.npz", "chb22_raw_eeg_128.npz"])
        #data_handler.read_raw_data(os.path.join(raw_data_folder, "chb02_raw_eeg_128.parquet"))
    elif mode == "classic_ml":
        # run the classic machine learning
        raise NotImplementedError
    elif mode == "deep_learning":
        # run the deep learning
        raise NotImplementedError
