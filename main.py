# main script, run this to start the epileptic seizure detection system
# script arguments:
# mode: "preprocessing" or "classic_ml" or "deep_learning"
# raw_data_folder: folder containing the raw data files
# window_folder: folder to store the windowed data files
# metadata_folder: folder to store the metadata files

import sys
from src.dataHandler import DataHandler

if __name__ == '__main__':

    # get the script arguments
    # mode = sys.argv[1]
    # raw_data_folder = sys.argv[2]
    # window_folder = sys.argv[3]
    # metadata_folder = sys.argv[4]

    # for testing
    mode = "preprocessing"
    raw_data_folder = "data/raw_example"
    window_folder = "data/windowed"
    metadata_folder = "data/metadata"

    if mode == "preprocessing":
        # run the preprocessing
        data_handler = DataHandler()
        data_handler.preprocess_data(raw_data_folder, window_folder, metadata_folder)




    