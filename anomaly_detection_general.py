import datetime
from functools import partial
from multiprocessing.shared_memory import SharedMemory
from os import makedirs, walk
from os.path import join, dirname, realpath, exists

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from model import AutoEncoder

from util import compute_tensor, compute_tensor_byte, filter_ip, filter_dest_port, filter_uniq
from util import parallelize_dataframe, convert_timestamp, get_model_type


def get_index(flatten_idx):
    third_idx = flatten_idx % 64
    second_idx = ((flatten_idx - third_idx) // 64) % 64
    first_idx = ((flatten_idx - third_idx - 64 * second_idx) // 64 ** 2) % 64
    return first_idx, second_idx, third_idx


def is_anomaly(model_dir, device, df, df_t, timestamp, entropy_tuple, mse_error, **kwargs):
    """
    Check if a time series is truly anomaly by removing the traffic and regenerate tensor to
    compare the mse error
    :param model_dir: model used for reconstruction
    :param device: the device that the model are on
    :param df: DataFrame that contains all traffic
    :param df_t: DataFrame that contains filtered traffic
    :param timestamp: starting timestamp used for tensor generation
    :param entropy_tuple: a tuple that contains the desired column name
    :param mse_error: previous MSE error
    :return: True/False
    """
    cpu_count = kwargs.get("cpu_count", 1)
    traffic_threshold = kwargs.get("traffic_threshold", 0.2)
    time_interval = kwargs.get("time_interval", 60)
    src_ip_idx = kwargs.get("src_ip_idx", None)
    dest_ip_idx = kwargs.get("dest_ip_idx", None)
    dest_port_idx = kwargs.get("dest_port_idx", None)

    for entropy in entropy_tuple:
        # re-filter with only the top values
        removed_traffic = []

        # get value count
        value_counts = df_t[entropy].value_counts(normalize=True)
        if entropy == "dest_port":
            # Combine consecutive ports
            value_counts = value_counts.sort_index()
            value_counts_index = value_counts.index.to_series()
            value_counts = value_counts.groupby(
                ((value_counts_index - value_counts_index.shift()) != 1).cumsum()
            ).transform('sum').sort_values(ascending=False)
        for index, value in zip(value_counts.index, value_counts):
            # check if above threshold
            if value < traffic_threshold:
                break
            removed_traffic.append(index)
        df_t = df_t[df_t[entropy].isin(removed_traffic)]

    if df_t.shape[0] == 0:  # print some warnings
        print(f"Error! {timestamp} {mse_error} with {src_ip_idx}, {dest_ip_idx}, {dest_port_idx}, has 0 row filtered.")
        with open("err_report.txt", "a") as f:
            f.write(f"{timestamp} {mse_error} ({src_ip_idx}, {dest_ip_idx}, {dest_port_idx})\n")
        return False, df_t
    with torch.no_grad():
        tensor = compute_tensor(
            timestamp, df.drop(df_t.index), "", return_tensor=True,
            time_interval=time_interval, cpu_count=cpu_count, device=device
        )
        tensor = tensor.to(device)

        model = AutoEncoder()
        model.to(device)
        save_dict = torch.load(join(model_dir, "best_model.pt"), map_location=device)
        model.load_state_dict(save_dict["model"])
        model.eval()
        recon = model(tensor.unsqueeze(0).unsqueeze(0))[0, 0]
        # calculate overall mse error
        new_mse_err = float(torch.sqrt(torch.mean((recon - tensor).pow(2))))
    return new_mse_err < mse_error, df_t


def _detect(shm_name, shape, dtype, model_dir, mse_err, time_stamp, save_dir, idxes, **kwargs):
    src_ip_idx, dest_ip_idx, dest_port_idx = idxes
    time_interval = kwargs.get("time_interval", 60)
    cpu_count = kwargs.get("cpu_count", 1)
    traffic_threshold = kwargs.get("traffic_threshold", 0.2)
    device = kwargs.get("device", "cpu")
    flag_entropy = kwargs.get("flag_entropy", 6)
    num_bytes_entropy = kwargs.get("num_bytes_entropy", 12)

    # Locate the shared memory by its name
    shm = SharedMemory(shm_name)
    # Create the np.recarray from the buffer of the shared memory
    np_array = np.recarray(shape=shape, dtype=dtype, buf=shm.buf)
    df = pd.DataFrame.from_records(np_array, index='index')

    print(src_ip_idx, dest_ip_idx, dest_port_idx)
    df_t = df[df["timestamp"].between(time_stamp, time_stamp + time_interval - 1)]
    tmp = filter_ip(df_t, ip_idx=src_ip_idx, src=True)
    tmp = filter_ip(tmp, ip_idx=dest_ip_idx, src=False)
    if dest_port_idx is not None:  # filter destination port if not AllPort
        tmp = filter_dest_port(tmp, dest_port_idx)

    # calculate the entropy for src ip, dest ip, and dest port
    src_ip_len = len(tmp["src_IP"].unique())
    dest_ip_len = len(tmp["dest_IP"].unique())
    dest_port_len = len(tmp["dest_port"].unique())

    # apply filter again based only on the lowest and second lowest entropy
    if max(src_ip_len, dest_ip_len, dest_port_len) == src_ip_len:
        df_t = filter_ip(df_t, ip_idx=dest_ip_idx, src=False)  # filter dest ip
        if dest_port_idx is not None:  # filter destination port if not AllPort
            df_t = filter_dest_port(df_t, dest_port_idx=dest_port_idx)
    elif max(src_ip_len, dest_ip_len, dest_port_len) == dest_ip_len:
        df_t = filter_ip(df_t, ip_idx=src_ip_idx, src=True)  # filter src ip
        if dest_port_idx is not None:  # filter destination port if not AllPort
            df_t = filter_dest_port(df_t, dest_port_idx=dest_port_idx)
    else:
        df_t = filter_ip(df_t, ip_idx=src_ip_idx, src=True)  # filter src ip
        df_t = filter_ip(df_t, ip_idx=dest_ip_idx, src=False)  # filter dest ip

    # Get the lowest and second lowest entropy column
    if dest_port_idx is None or (dest_port_idx is not None and dest_port_idx > 22):
        # if the port index is not a range, do not apply the threshold on the port
        if max(src_ip_len, dest_ip_len, dest_port_len) == dest_ip_len:
            entropy_tuple = ["src_IP"]
        elif max(src_ip_len, dest_ip_len, dest_port_len) == src_ip_len:
            entropy_tuple = ["dest_IP"]
        else:
            entropy_tuple = ["src_IP", "dest_IP"] if src_ip_len <= dest_ip_len else ["dest_IP", "src_IP"]
    else:
        tmp = [(src_ip_len, "src_IP"), (dest_ip_len, "dest_IP"), (dest_port_len, "dest_port")]
        entropy_tuple = [x[1] for x in sorted(tmp)[:2]]

    # Test if remove this anomaly traffic will lower the total MSE error
    config = {
        "cpu_count": cpu_count,
        "time_interval": time_interval,
        "traffic_threshold": traffic_threshold,
        "src_ip_idx": src_ip_idx,
        "dest_ip_idx": dest_ip_idx,
        "dest_port_idx": dest_port_idx
    }
    result, df_t = is_anomaly(model_dir, device, df, df_t, time_stamp, entropy_tuple, mse_err, **config)
    if not result:
        return

    # find the unique tuples
    unique_tuple = df_t[['src_IP', 'dest_IP', 'dest_port']].apply(tuple, axis=1).unique()

    # filter out the attack traffic in bidirectional manner
    df_t = filter_uniq(df, unique_tuple)
    df_anomaly = df_t[df_t["timestamp"].between(time_stamp, time_stamp + time_interval - 1)]

    # filter flags and number_of_bytes and use them to separate attack traffic
    tmp = df_anomaly[df_anomaly["protocol"] != "UDP"]
    flags_counts = tmp["flags"].value_counts(normalize=True)[:flag_entropy]
    bytes_counts = tmp["number_of_bytes"].value_counts(normalize=True)[:num_bytes_entropy]
    tmp = df_anomaly[df_anomaly["protocol"] == "UDP"]
    bytes_counts_udp = tmp["number_of_bytes"].value_counts(normalize=True)[:num_bytes_entropy]

    # forward and backward check is based on flags and number_of_bytes
    range_check = False
    flags_counts_list = flags_counts.index
    bytes_counts_list = bytes_counts.index
    bytes_counts_udp_list = bytes_counts_udp.index
    flags_counts = sum(flags_counts)
    bytes_counts = sum(bytes_counts)
    bytes_counts_udp = sum(bytes_counts_udp)
    if flags_counts >= 0.99 and bytes_counts >= 0.99 and bytes_counts_udp >= 0.99:
        def func(x):
            return x[
                ((x["protocol"] != "UDP") & x["flags"].isin(flags_counts_list) &
                 x["number_of_bytes"].isin(bytes_counts_list)) |
                ((x["protocol"] == "UDP") & x["number_of_bytes"].isin(bytes_counts_udp_list))
                ]

        df_t = func(df_t)
        df_anomaly = func(df_anomaly)
        range_check = True
    elif flags_counts >= 0.99 and bytes_counts >= 0.99:
        def func(x):
            return x[
                (x["protocol"] != "UDP") & x["flags"].isin(flags_counts_list) &
                x["number_of_bytes"].isin(bytes_counts_list)
                ]

        df_t = func(df_t)
        df_anomaly = func(df_anomaly)
        range_check = True
    elif flags_counts >= 0.99 and bytes_counts_udp >= 0.99:
        def func(x):
            return x[
                ((x["protocol"] != "UDP") & x["flags"].isin(flags_counts_list)) |
                ((x["protocol"] == "UDP") & x["number_of_bytes"].isin(bytes_counts_udp_list))
                ]

        df_t = func(df_t)
        df_anomaly = func(df_anomaly)
        range_check = True
    elif bytes_counts >= 0.99 and bytes_counts_udp >= 0.99:
        def func(x):
            return x[
                ((x["protocol"] != "UDP") & x["number_of_bytes"].isin(bytes_counts_list)) |
                ((x["protocol"] == "UDP") & x["number_of_bytes"].isin(bytes_counts_udp_list))
                ]

        df_t = func(df_t)
        df_anomaly = func(df_anomaly)
        range_check = True
    elif flags_counts >= 0.99:
        def func(x):
            return x[(x["protocol"] != "UDP") & x["flags"].isin(flags_counts_list)]

        df_t = func(df_t)
        df_anomaly = func(df_anomaly)
        range_check = True
    elif bytes_counts >= 0.99:
        def func(x):
            return x[(x["protocol"] != "UDP") & x["number_of_bytes"].isin(bytes_counts_list)]

        df_t = func(df_t)
        df_anomaly = func(df_anomaly)
        range_check = True
    elif bytes_counts_udp >= 0.99:
        def func(x):
            return x[(x["protocol"] == "UDP") & x["number_of_bytes"].isin(bytes_counts_udp_list)]

        df_t = func(df_t)
        df_anomaly = func(df_anomaly)
        range_check = True

    if range_check:
        # forward check
        i = 0
        df_check = df_t[df_t["timestamp"].between(time_stamp - time_interval * (i + 1),
                                                  time_stamp - time_interval * i - 1)]
        while df_check.shape[0] != 0 and i < (1800 / time_interval):
            df_anomaly = df_anomaly.append(df_check)
            i += 1
            df_check = df_t[df_t["timestamp"].between(time_stamp - time_interval * (i + 1),
                                                      time_stamp - time_interval * i - 1)]

        # backward check
        i = 0
        df_check = df_t[df_t["timestamp"].between(time_stamp + time_interval * (i + 1),
                                                  time_stamp + time_interval * (i + 2) - 1)]
        while df_check.shape[0] != 0 and i < (1800 / time_interval):
            df_anomaly = df_anomaly.append(df_check)
            i += 1
            df_check = df_t[df_t["timestamp"].between(time_stamp + time_interval * (i + 1),
                                                      time_stamp + time_interval * (i + 2) - 1)]

    df_anomaly = df_anomaly.drop_duplicates()
    df_anomaly = df_anomaly.sort_index()
    save_file_name = f"t{time_stamp}_mse{mse_err}_src{src_ip_idx}_dest{dest_ip_idx}_port{dest_port_idx}"
    df_anomaly.to_csv(join(save_dir, save_file_name + ".csv"), index=True)
    with open(join(save_dir, save_file_name + ".txt"), "w") as f:
        f.write(str(list(unique_tuple)))
    print("file written", save_file_name + ".csv")


def detect(df, save_dir, model_dir, anomaly_list, **kwargs):
    # get parameters
    cpu_count = kwargs.get("cpu_count", 2)
    device = kwargs.get("device", "cpu")

    # initialize model
    # model = AutoEncoder()
    # model.to(device)
    current_name = None

    # convert time string to utc timestamp
    df = parallelize_dataframe(df, convert_timestamp, cpu_count)
    min_time = int(df["timestamp"].min())
    max_time = int(df["timestamp"].max())

    np_array = df.to_records()
    shape, dtype = np_array.shape, np_array.dtype
    # Create a shared memory of size np_arry.nbytes
    shm = SharedMemory(create=True, size=np_array.nbytes)
    # Create a np.recarray using the buffer of shm
    shm_np_array = np.recarray(shape=shape, dtype=dtype, buf=shm.buf)
    # Copy the data into the shared memory
    np.copyto(shm_np_array, np_array)

    a = []
    for x in anomaly_list:
        if min_time <= x[0] <= max_time:
            for y in x[3]:
                a.append((x[0], x[1], x[2], y))

    res = []
    for time_stamp, mse_err, name, idxes in a:
        res.append(mp.Process(
            target=_detect,
            args=(shm.name, shape, dtype, model_dir.format(name), mse_err, time_stamp, save_dir, idxes),
            kwargs=kwargs
        ))
        if len(res) >= 8:
            for r in res:
                r.start()
            for r in res:
                r.join()
            res.clear()
    for r in res:
        r.start()
    for r in res:
        r.join()


    shm.close()
    shm.unlink()


def main(file_dir):
    # hyper parameter definition
    week_str = "july_week5"
    time_interval = 600
    tensor_dirs = [f"{week_str}_tensors_all_{time_interval // 60}min"]
    save_dir = "june_week_2to4_{}_"f"{time_interval // 60}min_model"
    random_seed = 42

    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    mse_threshold = 65  # 4
    time_series_threshold = 10000  # 2000
    total_traffic_threshold = 0.2
    port_dimension_threshold = 0.025
    flag_entropy = 6
    num_bytes_entropy = 12
    anomaly_save_dir = f"anomaly_time_series_{week_str}_{time_interval // 60}min_mse{mse_threshold}"
    anomaly_save_dir += f"_port{port_dimension_threshold * 100}_bidir_{total_traffic_threshold * 100}_1hr"
    anomaly_save_dir += f"_flag{flag_entropy}_numBytes{num_bytes_entropy}_hardcode_dest_UDP"
    anomaly_list_file = anomaly_save_dir.replace("anomaly_time_series", "anomaly_list") + ".txt"
    makedirs(anomaly_save_dir, exist_ok=True)
    cpu_count = mp.cpu_count()

    # all anomaly time series stores here
    # format: [(timestamp, reconstruction MSE error, model name, [(src_ip, dest_ip, dest_port), ...]), ...]
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
                                    for err_idx in torch.argsort(abs_error.flatten(), dim=-1, descending=True)[:10]:
                                        first_idx, second_idx, third_idx = get_index(err_idx)
                                        if abs_error[first_idx, second_idx, third_idx].item() > time_series_threshold:
                                            abs_error_list.append((first_idx.item(), second_idx.item(), third_idx.item()))
                                        else:
                                            break
                                    #  check for all dest port
                                    abs_port_error = abs(tensor.sum(2) - recon.sum(2))
                                    for err_idx in torch.argsort(abs_port_error.flatten(), dim=-1, descending=True):
                                        first_idx, second_idx, third_idx = get_index(err_idx)
                                        if abs_port_error[second_idx, third_idx].item() > time_series_threshold:
                                            append = True
                                            for (src_idx, dest_idx, port_idx) in abs_error_list:
                                                if src_idx == second_idx.item() and dest_idx == third_idx.item():
                                                    original_err = abs_error[src_idx, dest_idx, port_idx].item()
                                                    new_err = abs_port_error[second_idx, third_idx].item()
                                                    if new_err < original_err * (1 + port_dimension_threshold):
                                                        append = False
                                                    break
                                            if append:
                                                abs_error_list.append((second_idx.item(), third_idx.item(), None))
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
        "flag_entropy": flag_entropy,
        "num_bytes_entropy": num_bytes_entropy,
        "cpu_count": cpu_count,
        "device": device
    }
    # for file_location in [f"{week_str.replace('_', '.')}.csv"]:
    for file_location in ["july_week5_1min_mse4_port2.5_bidir_20.0_1hr_flag6_numBytes12_hardcode_dest_UDP_removed.csv"]:
        print("Process file", file_location)
        i = 0
        for df in pd.read_csv(file_location, index_col=0, chunksize=15000000):
        # for df in pd.read_csv(file_location, names=header, usecols=header, chunksize=15000000):
            print("read")
            detect(df, anomaly_save_dir, save_dir, anomaly_list, **detection_dict)
            i += 1
            print("i:", i)


if __name__ == '__main__':
    file_dir_name = dirname(realpath(__file__))
    main(file_dir_name)
