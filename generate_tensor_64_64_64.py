import os
import time
import numpy as np
import pandas as pd
import torch
import multiprocessing
from multiprocessing import Pool
from functools import partial
from multiprocessing.shared_memory import SharedMemory
from util import header, convert_timestamp, compute_tensor, compute_tensor_byte


def compute_tensor_mp(shm_name, shape, dtype, t, save_dir, **kwargs):
    # Locate the shared memory by its name
    shm = SharedMemory(shm_name)
    # Create the np.recarray from the buffer of the shared memory
    np_array = np.recarray(shape=shape, dtype=dtype, buf=shm.buf)
    df = pd.DataFrame.from_records(np_array, index='index')
    compute_tensor(t, df, save_dir, **kwargs)
    shm.close()


def filter_background(d):
    return d[d["label"] == "background"]


def filter_port(d, port=25):
    return d[(d["dest_port"] == port) | (d["src_port"] == port)]


def generate_from_single_csv(file_path, save_dir, time_interval=60, flt_bg=True, flt_port=False, cvt_ts=True, port_number=25):
    cpu_count = multiprocessing.cpu_count()
    print(cpu_count)
    i = 0
    # for df in pd.read_csv(file_path, index_col=0,
    #                       usecols=['index', 'timestamp', 'src_IP', 'dest_IP', 'dest_port', 'protocol'],
    #                       chunksize=15000000):
    for df in pd.read_csv(file_path, names=header,
                          usecols=['timestamp', 'src_IP', 'dest_IP', 'src_port', 'dest_port', 'protocol', 'number_of_bytes', 'label'],
                          chunksize=15000000):
        print("read", file_path)
        df = df.astype({
            'src_port': 'int32', 
            'dest_port': 'int32',  
            'number_of_bytes': 'int32'
        })
        with Pool(cpu_count) as pool:
            a = time.time()
            df_split = np.array_split(df, cpu_count)
            if flt_bg:  # for calibration (training) set, keep only background data.
                df_split = pool.map(filter_background, df_split)
            if flt_port:
                df_split = pool.map(partial(filter_port, port=port_number), df_split)
            if cvt_ts:  # convert timestamp
                df_split = pool.map(convert_timestamp, df_split)
            df = pd.concat(df_split)
        print("get background and convert time:", time.time() - a)
        start_time = df.timestamp.min()
        end_time = df.timestamp.max()
        a = time.time()
        
        
        np_array = df.to_records()
        shape, dtype = np_array.shape, np_array.dtype
        # Create a shared memory of size np_arry.nbytes
        shm = SharedMemory(create=True, size=np_array.nbytes)
        # Create a np.recarray using the buffer of shm
        shm_np_array = np.recarray(shape=shape, dtype=dtype, buf=shm.buf)
        # Copy the data into the shared memory
        np.copyto(shm_np_array, np_array)
        print(start_time, end_time, (end_time - start_time) / time_interval)
        with Pool(cpu_count) as pool:
            # the list of timestamp (type: int) for tensor.
            pool.map(
                partial(compute_tensor_mp, shm.name, shape, dtype,
                        save_dir=save_dir, time_interval=time_interval, cpu_count=cpu_count),
                range(start_time, end_time - (time_interval // 2), time_interval)
            )
        shm.close()
        shm.unlink()
        i += 1
        print(i, f"compute time for {round((end_time - start_time) / time_interval, 2)} timestamps:", time.time() - a)


def combine_tensors(tensors_dir, save_dir, combine_number):
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for root, _, files in os.walk(tensors_dir):
            files = sorted(files, key=lambda x: int(x[6:-3]), reverse=True)
            for idx, file in enumerate(files[:-9]):
                combined_tensor = torch.zeros(64, 64, 64)
                for f in files[idx:idx + combine_number]:
                    combined_tensor += torch.load(os.path.join(root, f))
                torch.save(combined_tensor, os.path.join(save_dir, file))


if __name__ == '__main__':
    file_dir_name = os.path.dirname(os.path.realpath(__file__))
    
#     for i in [2, 3, 4]:
#         generate_from_single_csv(os.path.join(file_dir_name, f"june.week{i}.csv"),
#                                  os.path.join(file_dir_name, f"june_week{i}_tensors_1min_port53"), 60,
#                                  flt_bg=True, flt_port=True, port_number=53, cvt_ts=True)
    
    generate_from_single_csv(os.path.join(file_dir_name, "june.week4.csv"),
                             os.path.join(file_dir_name, "june_week4_tensors_1min"), 60,
                             flt_bg=True, flt_port=False, port_number=None, cvt_ts=True)

