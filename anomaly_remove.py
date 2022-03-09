import os
import datetime
from os import makedirs
from os.path import join, dirname, realpath

import pandas as pd
import torch
import torch.nn as nn
import multiprocessing
from multiprocessing import Pool
from functools import partial

from util import ip_lst, timestr_to_timestamp, filter_dest_port, filter_ip, header, parallelize_dataframe, \
    convert_timestamp, compute_tensor
    

def get_detected(detected_dir):
    detected_traffic = None
    for root, _, files in os.walk(detected_dir):
        total = len(files)
        i = 0
        print(root)
        for file in files:
            if file.endswith(".csv"):
                df = pd.read_csv(join(root, file)).set_index("Unnamed: 0")
                if df.shape[0] == 0:
                    continue
                if detected_traffic is None:
                    detected_traffic = df.index
                else:
                    detected_traffic = detected_traffic.append(df[~df.index.isin(detected_traffic)].index)
                i += 1
                print(f"{i / total}")
        break
    return detected_traffic
     

def clear_csv(target_path):
    with open(target_path, "w") as f:
        f.write("index," + ",".join(header) + "\n")


def remove_detected(original_path, target_path, detected_traffic):
    i = 0
    for df in pd.read_csv(original_path, names=header, usecols=header, chunksize=10000000):
        df[~df.index.isin(detected_traffic)].to_csv(target_path, mode="a", index=True, header=False)
        i += 1
        print(i)


def main(original_path, target_path, detected_dir):
    clear_csv(target_path)
    remove_detected(original_path, target_path, get_detected(detected_dir))
    

if __name__ == "__main__":
    file_dir_name = dirname(realpath(__file__))
    main(join(file_dir_name, "july.week5.csv"),
         join(file_dir_name, "july_week5_1min_mse4_port2.5_bidir_20.0_1hr_flag6_numBytes12_"
                             "hardcode_dest_UDP_removed.csv"),
         join(file_dir_name, "anomaly_time_series_july_week5_1min_mse4_port2.5_bidir_20.0_"
                             "1hr_flag6_numBytes12_hardcode_dest_UDP_merged"))
