import sys
import time
from functools import partial
from datetime import datetime
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from os import makedirs, walk
from os.path import join, dirname, realpath, exists
from shutil import rmtree

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

from util import compute_tensor, filter_ip, filter_bytes, dest_ip_lst
from util import parallelize_dataframe, convert_timestamp, get_model_type, get_index, header
from model import AutoEncoder


def _detect(shm_name, shape, dtype, anomaly_time_series, **kwargs):
    # get parameters
    time_stamp, mse_err, name, (src_ip_idx, dest_ip_idx, bytes_idx) = anomaly_time_series
    time_interval = kwargs.get("time_interval", 60)
    traffic_threshold = kwargs.get("traffic_threshold", 0.2)
    print(datetime.utcfromtimestamp(time_stamp), src_ip_idx, dest_ip_idx, bytes_idx)

    # restore df from the shared memory
    shm = SharedMemory(shm_name)
    np_array = np.recarray(shape=shape, dtype=dtype, buf=shm.buf)
    df = pd.DataFrame.from_records(np_array, index='index')

    start_time = time.time()  # start timer
    df_t = df[df["timestamp"].between(time_stamp, time_stamp + time_interval - 1)]
    tmp = filter_ip(df_t, ip_idx=src_ip_idx, src=True)
    tmp = filter_ip(tmp, ip_idx=dest_ip_idx, src=False)
    tmp = filter_bytes(tmp, bytes_idx)

    # calculate the entropy for src ip, dest ip, and dest port
    src_ip_len = len(tmp["src_IP"].unique())
    dest_ip_len = len(tmp["dest_IP"].unique())
    bytes_len = len(tmp["number_of_bytes"].unique())

    # apply filter again based only on the lowest and second lowest entropy
    if max(src_ip_len, dest_ip_len, bytes_len) == src_ip_len:
        df_t = filter_ip(df_t, ip_idx=dest_ip_idx, src=False)  # filter dest ip
        df_t = filter_bytes(df_t, bytes_idx)  # filter number of bytes
    elif max(src_ip_len, dest_ip_len, bytes_len) == dest_ip_len:
        df_t = filter_ip(df_t, ip_idx=src_ip_idx, src=True)  # filter src ip
        df_t = filter_bytes(df_t, bytes_idx)  # filter number of bytes
    else:
        df_t = filter_ip(df_t, ip_idx=src_ip_idx, src=True)  # filter src ip
        df_t = filter_ip(df_t, ip_idx=dest_ip_idx, src=False)  # filter dest ip
    
    # re-filter with only the top values
    tmp = [(src_ip_len, "src_IP"), (dest_ip_len, "dest_IP"), (bytes_len, "number_of_bytes")]
    entropy_tuple = [x[1] for x in sorted(tmp)[:2]]
    for entropy in entropy_tuple:
        value_counts = df_t[entropy].value_counts(normalize=True)  # get value count
        # only care about those above the threshold
        df_t = df_t[df_t[entropy].isin(value_counts[value_counts >= traffic_threshold].index)]

    print(datetime.utcfromtimestamp(time_stamp), src_ip_idx, dest_ip_idx, bytes_idx,
          "apply initial filter time:", time.time() - start_time)

    if df_t.shape[0] == 0:  # print some warnings
        print(f"Error! {time_stamp} {mse_err} with {src_ip_idx} {dest_ip_idx} {bytes_idx}, has 0 row filtered.")
        with open("err_report.txt", "a") as f:
            f.write(f"{time_stamp} {mse_err} ({src_ip_idx}, {dest_ip_idx}, {bytes_idx})\n")
    else:
        tensor = compute_tensor(
            time_stamp, df.drop(df_t.index), "anomaly_detection_tmp",
            return_tensor=True, time_interval=time_interval
        )
        torch.save(tensor, join(
            "anomaly_detection_tmp",
            f'tensor{time_stamp}_{mse_err}_{src_ip_idx}_{dest_ip_idx}_{bytes_idx}.pt'
        ))

    # release the shared memory
    shm.close()


def detect(df, anomaly_list, cpu_count, **kwargs):
    # convert time string to utc timestamp
    df = parallelize_dataframe(df, convert_timestamp, cpu_count)
    min_time = int(df["timestamp"].min())
    max_time = int(df["timestamp"].max())

    df = df[(df["dest_port"] == kwargs["port_number"]) | (df["src_port"] == kwargs["port_number"])]
    # create shared memory df
    np_array = df.to_records()
    shape, dtype = np_array.shape, np_array.dtype
    shm = SharedMemory(create=True, size=np_array.nbytes)
    shm_np_array = np.recarray(shape=shape, dtype=dtype, buf=shm.buf)
    np.copyto(shm_np_array, np_array)

    # serialize the anomaly list to make it easier to parallelize
    anomaly_list_serialized = []
    for x in anomaly_list:
        if min_time <= x[0] <= max_time - (kwargs["time_interval"] // 2):
            for y in x[3]:
                anomaly_list_serialized.append((x[0], x[1], x[2], y))

    # parallelize here
    with Pool(cpu_count) as pool:
        pool.map(
            partial(_detect, shm.name, shape, dtype, **kwargs),
            anomaly_list_serialized
        )

    # release the shared memory
    shm.close()
    shm.unlink()


def main(file_dir, week_str, time_interval, port_number, mse_threshold, time_series_threshold):
    # hyper parameter definition
    tensor_dirs = [f"{week_str}_tensors_all_{time_interval // 60}min_port{port_number}"]
    save_dir = "june_week_2to4_{}_"f"{time_interval // 60}min_port{port_number}_model"
    random_seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_traffic_threshold = 0.5
    port_dimension_threshold = 0.025
    flag_entropy = 6
    num_bytes_entropy = 12
    anomaly_save_dir = f"anomaly_time_series_{week_str}_{time_interval // 60}min_mse{mse_threshold}_abs{time_series_threshold}"
    anomaly_save_dir += f"_port{port_dimension_threshold * 100}_bidir_{total_traffic_threshold * 100}_1hr"
    anomaly_save_dir += f"_flag{flag_entropy}_numBytes{num_bytes_entropy}_hardcode_dest_UDP"
    anomaly_list_file = anomaly_save_dir.replace("anomaly_time_series", "anomaly_list") + ".txt"
    cpu_count = mp.cpu_count()

    # all anomaly time series stores here
    # format: [(timestamp, reconstruction MSE error, model name, [(src_ip, dest_ip, number_of_bytes), ...]), ...]
    if exists(anomaly_list_file):
        anomaly_list = eval(open(anomaly_list_file, "r").read())
    else:
        anomaly_list = []
        for name in ["weekday_morning", "weekday_evening", "weekend_morning", "weekend_evening"]:
            print("Processing", name)
            # Fix the randomness
            torch.manual_seed(random_seed)

            model = AutoEncoder()  # initialize model
            model.to(device)
            # load model
            save_dict = torch.load(join(save_dir.format(name), "best_model.pt"), map_location=device)
            model.load_state_dict(save_dict["model"])
            model.eval()
            # Iterate all tensor folders and get all the anomaly time series
            for tensor_dir in tensor_dirs:
                print("Starting tensors in", tensor_dir)
                for root, _, files in walk(tensor_dir):
                    total_file = len(files)
                    for i, file in enumerate(sorted(files, key=lambda x: int(x[6:-3]))):
                        time_stamp = int(file[6:-3])
                        if get_model_type(datetime.utcfromtimestamp(time_stamp)) == name:
                            with torch.no_grad():
                                tensor = torch.load(join(root, file)).to(device)
                                recon = model(tensor.unsqueeze(0).unsqueeze(0))[0, 0]
                                # calculate overall mse error
                                mse_err = float(torch.sqrt(torch.mean((recon - tensor).pow(2))))
                                if mse_err > mse_threshold:  # based on threshold select the tensor
                                    abs_error = abs(tensor - recon)
                                    abs_error_list = []
                                    for err_idx in torch.argsort(abs_error.flatten(), dim=-1, descending=True):
                                        first_idx, second_idx, third_idx = get_index(err_idx)
                                        if abs_error[first_idx, second_idx, third_idx].item() > time_series_threshold:
                                            if tensor[first_idx, second_idx, third_idx].item() > 0:
                                                abs_error_list.append((first_idx.item(), second_idx.item(), third_idx.item()))
                                        else:
                                            break
                                    if len(abs_error_list) > 0:
                                        print(f"added {len(abs_error_list)} anomaly")
                                        anomaly_list.append((time_stamp, mse_err, name, abs_error_list))
                        if i % 100 == 0:
                            print("processed", i + 1, "out of", total_file)
                print("Finished process tensors in", tensor_dir)
            print("Finished", name)
        print("anomaly_list is ready")
        anomaly_list.sort()
        with open(anomaly_list_file, "w") as f:
            f.write(str(anomaly_list))
        return
    print("In total: ", sum([len(x[3]) for x in anomaly_list]), "anomalous time series.")
    detection_dict = {
        "traffic_threshold": total_traffic_threshold,
        "time_interval": time_interval,
        "port_number": port_number
    }

    # clean up tmp folder and restart again
    if exists("anomaly_detection_tmp"):
        rmtree("anomaly_detection_tmp")
    makedirs("anomaly_detection_tmp", exist_ok=True)

    # for file_location in [f"{week_str.replace('_', '.')}.csv"]:
    for file_location in [f"{week_str.replace('_', '.')}.csv"]:
        print("Process file", file_location)
        i = 0
#         for df in pd.read_csv(file_location, index_col=0, chunksize=15000000):
        for df in pd.read_csv(file_location, names=header, usecols=header, chunksize=15000000):
            print("read")
            df = df.astype({
                'src_port': 'int32', 
                'dest_port': 'int32',  
                "forwarding_status": "int32", 
                "type_of_service": "int32", 
                "packets_exchanged": "int32",
                'number_of_bytes': 'int32'
            })
            detect(df, anomaly_list, cpu_count, **detection_dict)
            i += 1
            print("i:", i)


if __name__ == '__main__':
    file_dir_name = dirname(realpath(__file__))
    
    if len(sys.argv) > 1:
        week_str, time_interval, port_number, mse_threshold, time_series_threshold = sys.argv[1:]
        time_interval, port_number, time_series_threshold = int(time_interval), int(port_number), int(time_series_threshold)
        if '.' in mse_threshold:
            mse_threshold = float(mse_threshold)
        else:
            mse_threshold = int(mse_threshold)
        main(file_dir_name, week_str, time_interval, port_number, mse_threshold, time_series_threshold)
    else:
        main(file_dir_name, "august_week1", 600, 25, 2, 200)
#         main(file_dir_name, "august_week1", 60, 53, 4, 1000)
#         main(file_dir_name, "august_week1", 600, 6667, 0.1, 10)
