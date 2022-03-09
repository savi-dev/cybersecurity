import sys
import time
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from os import makedirs, walk
from os.path import join, dirname, realpath, exists

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

from model import AutoEncoder
from util import filter_ip, filter_bytes, filter_uniq, dest_ip_lst
from util import parallelize_dataframe, convert_timestamp, get_model_type, header


def _detect(shm_name, shape, dtype, save_dir, anomaly_time_series, **kwargs):
    # get parameters
    time_stamp, src_ip_idx, dest_ip_idx, bytes_idx, mse_err = anomaly_time_series
    time_interval = kwargs.get("time_interval", 60)
    traffic_threshold = kwargs.get("traffic_threshold", 0.2)
    flag_entropy = kwargs.get("flag_entropy", 6)
    print(datetime.utcfromtimestamp(time_stamp), src_ip_idx, dest_ip_idx, bytes_idx)

    # restore df from the shared memory
    shm = SharedMemory(shm_name)
    np_array = np.recarray(shape=shape, dtype=dtype, buf=shm.buf)
    df = pd.DataFrame.from_records(np_array, index='index')

    start_time = time.time()
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
    
    # re-filter the lowest two entropies with only the top values
    for (_, entropy) in sorted([(src_ip_len, "src_IP"), (dest_ip_len, "dest_IP"), (bytes_len, "number_of_bytes")])[:2]:
        value_counts = df_t[entropy].value_counts(normalize=True)  # get value count
        # only care about those above the threshold
        df_t = df_t[df_t[entropy].isin(value_counts[value_counts >= traffic_threshold].index)]

    print(datetime.utcfromtimestamp(time_stamp), src_ip_idx, dest_ip_idx, bytes_idx,
          "apply initial filter time:", time.time() - start_time, df_t.shape[0])

    assert df_t.shape[0] != 0, f"For part two this should not be 0, check what is going on. {time_stamp} {src_ip_idx} {dest_ip_idx} {df.shape[0]}"

    start_time = time.time()
    # filter out the attack traffic in bidirectional manner
    forward_tuple = ['src_IP', 'dest_IP', 'dest_port']
    reverse_tuple = ['dest_IP', 'src_IP', 'src_port']
    unique_tuple = df_t[forward_tuple].apply(tuple, axis=1).unique()
    df_t = filter_uniq(df, forward_tuple, reverse_tuple, unique_tuple)
    df_anomaly = df_t[df_t["timestamp"].between(time_stamp, time_stamp + time_interval - 1)]

    # filter flags and number_of_bytes and use them to separate attack traffic
    tmp = df_anomaly[df_anomaly["protocol"] != "UDP"]
    flags_counts = tmp["flags"].value_counts(normalize=True)[:flag_entropy]

    # forward and backward check is based on flags and number_of_bytes
    range_check = True
    flags_counts_list = flags_counts.index
    flags_counts = sum(flags_counts)
    if flags_counts >= 0.99:
        df_t = df_t[df_t["timestamp"].between(time_stamp - 1800, time_stamp + 1800 - 1)]
        df_anomaly = df_t[
            ((df_t["protocol"] != "UDP") & df_t["flags"].isin(flags_counts_list)) |
            (df_t["protocol"] == "UDP")
        ]
    else:
        df_t = df_t[df_t["timestamp"].between(time_stamp - 1800, time_stamp + 1800 - 1)]
        df_anomaly = df_t[
            (df_t["protocol"] == "UDP") | (df_t["timestamp"].between(time_stamp, time_stamp + time_interval - 1) | (df_t["protocol"] != "UDP"))
        ]
        range_check = False
        

    # if range_check:
    #     df_anomaly = df_t[df_t["timestamp"].between(time_stamp - time_interval * 60, time_stamp + time_interval * 60 - 1)]

    df_anomaly = df_anomaly.drop_duplicates()
    df_anomaly = df_anomaly.sort_index()
    save_file_name = f"t{time_stamp}_mse{mse_err}_src{src_ip_idx}_dest{dest_ip_idx}_bytes{bytes_idx}"
    df_anomaly.to_csv(join(save_dir, save_file_name + ".csv"), index=True)
    with open(join(save_dir, save_file_name + ".txt"), "w") as f:
        f.write(str(list(unique_tuple)))
    print("file written", save_file_name + ".csv")
    print("Second part filter time:", time.time() - start_time)
    shm.close()


def detect(df, cpu_count, save_dir, anomaly_list, **kwargs):
    # convert time string to utc timestamp
    df = parallelize_dataframe(df, convert_timestamp, cpu_count)
    min_time = int(df["timestamp"].min())
    max_time = int(df["timestamp"].max())
    df = df[(df["dest_port"] == kwargs["port_number"]) | (df["src_port"] == kwargs["port_number"])]
    df = df.astype({
        'src_port': 'int32', 
        'dest_port': 'int32',  
        "forwarding_status": "int32", 
        "type_of_service": "int32", 
        "packets_exchanged": "int32",
        'number_of_bytes': 'int32'
    })

    # create shared memory df
    np_array = df.to_records()
    shape, dtype = np_array.shape, np_array.dtype
    shm = SharedMemory(create=True, size=np_array.nbytes)
    shm_np_array = np.recarray(shape=shape, dtype=dtype, buf=shm.buf)
    np.copyto(shm_np_array, np_array)

    # parallelize here
    with Pool(cpu_count) as pool:
        pool.map(
            partial(_detect, shm.name, shape, dtype, save_dir, **kwargs),
            [x for x in anomaly_list if min_time <= x[0] <= max_time - kwargs["time_interval"] // 2]
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
    makedirs(f"anomaly_time_series_{week_str}_port{port_number}_detect_abs{time_series_threshold}", exist_ok=True)
    cpu_count = mp.cpu_count()

    # all anomaly time series stores here
    # format: [(timestamp, reconstruction MSE error, model name, [(src_ip, dest_ip, number_of_bytes), ...]), ...]
    assert exists(anomaly_list_file), "No anomaly list file found. Please check the config or run p1 first"
    anomaly_list = eval(open(anomaly_list_file, "r").read())
    print("In total: ", sum([len(x[3]) for x in anomaly_list]), "anomalous time series.")

    # evaluate if the dropped attack tensor has a lower MSE error
    if not exists(anomaly_list_file[:-4] + "_p2.txt"):
        start_time = time.time()
        valid_anomaly_time_series = []
        for name in ["weekday_morning", "weekday_evening", "weekend_morning", "weekend_evening"]:
            model = AutoEncoder()
            model.to(device)
            save_dict = torch.load(join(save_dir.format(name), "best_model.pt"), map_location=device)
            model.load_state_dict(save_dict["model"])
            model.eval()

            for root, _, files in walk("anomaly_detection_tmp"):
                for f in files:
                    time_stamp, mse_err, src_ip_idx, dest_ip_idx, bytes_idx = f[6:-3].split('_')
                    time_stamp, mse_err = int(time_stamp), float(mse_err)
                    if get_model_type(datetime.utcfromtimestamp(time_stamp)) == name:
                        with torch.no_grad():
                            tensor = torch.load(join(root, f), map_location=device)
                            recon = model(tensor.unsqueeze(0).unsqueeze(0))[0, 0]
                            new_mse_err = float(torch.sqrt(torch.mean((recon - tensor).pow(2))))
                            if new_mse_err < mse_err:
                                valid_anomaly_time_series.append(
                                    (time_stamp, int(src_ip_idx), int(dest_ip_idx), int(bytes_idx), mse_err)
                                )
        valid_anomaly_time_series.sort()
        with open(anomaly_list_file[:-4] + "_p2.txt", "w") as f:
            f.write(str(valid_anomaly_time_series))
        print("Part 2 model evaluation time:", time.time() - start_time)
        return
    else:
        valid_anomaly_time_series = eval(open(anomaly_list_file[:-4] + "_p2.txt", "r").read())
    print("In total: ", len(valid_anomaly_time_series), "anomalous time series for part 2.")

    detection_dict = {
        "traffic_threshold": total_traffic_threshold,
        "time_interval": time_interval,
        "flag_entropy": flag_entropy,
        "port_number": port_number
    }
    for file_location in [f"{week_str.replace('_', '.')}.csv"]:
        print("Process file", file_location)
        i = 0
#         for df in pd.read_csv(file_location, index_col=0, chunksize=15000000):
        for df in pd.read_csv(file_location, names=header, usecols=header, chunksize=15000000):
            print("read")
            detect(df, cpu_count, f"anomaly_time_series_{week_str}_port{port_number}_detect_abs{time_series_threshold}", valid_anomaly_time_series, **detection_dict)
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
