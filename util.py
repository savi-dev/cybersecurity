import os
from functools import partial
from multiprocessing import Pool

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch


header = ['timestamp', 'duration', 'src_IP', 'dest_IP',
          'src_port', 'dest_port', 'protocol', 'flags',
          'forwarding_status', 'type_of_service',
          'packets_exchanged', 'number_of_bytes', 'label']
ip_lst = [
    '143.72', '43.164', '204.97', '168.38',
    '194.233', '106.150', '210.46', '70.211',
    '133.54', '78.160', '74.158', '54.143',
    '74.159', '214.43', '165.129', '209.48',
    '42.219.159', '42.219.158', '42.219.157', '42.219.156',
    '42.219.155', '42.219.154', '42.219.153', '42.219.152',
    '42.219.151', '42.219.150', '42.219.149', '42.219.148',
    '42.219.147', '42.219.146', '42.219.145', '42.219.144',
    (0, 7), (8, 15), (16, 23), (24, 31), (32, 39), (40, 47),
    (48, 55), (56, 63), (64, 71), (72, 79), (80, 87), (88, 95),
    (96, 103), (104, 111), (112, 119), (120, 127), (128, 135),
    (136, 143), (144, 151), (152, 159), (160, 167), (168, 175),
    (176, 183), (184, 191), (192, 199), (200, 207), (208, 215),
    (216, 223), (224, 231), (232, 239), (240, 247), (248, 255)
]
num_bytes_lst = [
    (1, 20), (21, 40), (41, 60), (61, 80), (81, 100),
    (101, 120), (121, 140), (141, 160), (161, 180), 
    (181, 200), (201, 220), (221, 240), (241, 260),
    (261, 280), (281, 300), (301, 320), (321, 340),
    (341, 360), (361, 380), (381, 400), (401, 420),
    (421, 440), (441, 460), (461, 480), (481, 500),
    (501, 520), (521, 540), (541, 560), (561, 580),
    (581, 600), (601, 700), (701, 800), (801, 900),
    (901, 1000), (1001, 1100), (1101, 1200), (1201, 1300),
    (1301, 1400), (1401, 1500), (1501, 1600), (1601, 1700),
    (1701, 1800), (1801, 1900), (1901, 2000), (2001, 2100),
    (2101, 2200), (2201, 2300), (2301, 2400), (2401, 2500),
    (2501, 2600), (2601, 2800), (2801, 3000), (3001, 3200),
    (3201, 3400), (3401, 3600), (3601, 3800), (3801, 4000),
    (4001, 4200), (4201, 4400), (4401, 4600), (4601, 14600),
    (14601, 24600), (24601, 34600), 34601
]
dest_ip_lst = [
    '108.66.255.250', '192.143.87.120', '192.143.87.90', '192.143.87.124',
    '108.66.255.199', '108.66.255.194', '108.66.255.255', '192.143.87.95',
    '192.143.84.56', '192.143.84.60', '193.27.83.103', '193.27.1.120',
    '193.27.83.116', '121.106.2.63', '193.27.6.180', '193.26.243.174',
    '54.143.48.199', '193.27.6.136', '193.27.1.135', '54.143.48.135',
    '193.27.83.68', '193.26.243.182', '193.27.6.149', '193.27.6.165',
    '193.26.243.129', '193.27.83.86', '193.26.243.145', '193.43.63.49',
    '177.235.191.17', '55.83.104.75', '253.139.127.227', '192.22.7.102',
    '53.218.14.195', '192.22.7.103', '204.97.194.148', '192.22.25.40',
    '196.125.221.68', '196.121.33.4', '213.173.137.32', '213.173.139.59',
    '213.173.137.60', '42.219.158.161', '42.219.155.20', '42.219.158.188', 
    '42.219.154.185', '42.219.153.43', '42.219.158.160', '42.219.153.12',
    '42.219.155.103', '42.219.154.97', '42.219.158.179', '42.219.153.35', 
    '42.219.154.147', '42.219.145.18', '42.219.154.134', '42.219.154.108', 
    '42.219.154.100', '42.219.153.26', '42.219.154.128', '42.219.154.144',
    '42.219.153.45', '42.219.153.76', '42.219.154.190', (0, 255)
]
dest_port_lst = [
    # 0 - 22
    (0, 127), (128, 255),
    (256, 511), (512, 767),
    (768, 1024), (1024, 2047),
    (2048, 3071), (3072, 4095),
    (4096, 5119), (5120, 6143),
    (6144, 7167), (7168, 8191),
    (8192, 9215), (9216, 10239),
    (10240, 16383), (16384, 22527),
    (22528, 28671), (28672, 34815),
    (34816, 40959), (40960, 47103),
    (47104, 53247), (53248, 59391),
    (59392, 65535),
    # 23 - 48
    [(53, ['TCP', 'UDP'])],
    [(80, 'TCP')],
    [(80, 'UDP')],
    [([81, 8080, 8081, 8888, 9080, 3128, 6588, 7779, 1080], ['TCP', 'UDP'])],
    [(443, ['TCP', 'UDP'])],
    [(445, ['TCP', 'UDP'])],
    [(25, ['TCP', 'UDP']), (587, 'TCP')],
    [([23, 2323, 9527], ['TCP', 'UDP'])],
    [((20, 21), ['TCP', 'UDP']), (69, 'UDP')],
    [(8000, 'TCP')],
    [(110, ['TCP', 'UDP']), (995, 'TCP')],
    [([161, 162], ['TCP', 'UDP'])],
    [(22, ['TCP', 'UDP'])],
    [(123, ['TCP', 'UDP'])],
    [((5060, 5061), ['TCP', 'UDP'])],
    [([143, 993], ['TCP', 'UDP'])],
    [(389, ['TCP', 'UDP'])],
    [(3306, ['TCP', 'UDP']), (1433, 'TCP')],
    [((768, 783), 'ICMP')],
    [([0, 2048], 'ICMP')],
    [(1720, ['TCP', 'UDP']), ([1002, 5222], 'TCP')],
    [([3389, 5500] + [x for x in range(5800, 5811)] + [x for x in range(5900, 5911)], 'TCP')],
    [((135, 139), ['TCP', 'UDP'])],
    [([500, 4500], 'UDP'), ([1701, 1732], ['TCP', 'UDP']), (0, ['ESP', 'AH'])],
    [(1900, ['TCP', 'UDP']), ([6, 5000, 5431, 2048, 2869, 5351, 37215, 18067], 'TCP')],
    [(1723, 'TCP')],
    # 49 special action
    6667,  # (0, 'GRE'),7/13 updated
    # 50 - 65
    [((6881, 6999), ['TCP', 'UDP']), ((27000, 27050), ['TCP', 'UDP'])],
    [([111, 135], ('TCP', 'UDP')), ((6000, 6063), 'UDP')],
    [([554, 7070, 9090, 22010], ['TCP', 'UDP'])],
    [((1025, 1029), ['TCP', 'UDP'])],
    [([6343, 8291, 8728, 8729, 4153], 'TCP'), ([5678, 20561], 'UDP')],
    [(520, 'UDP')],
    [([5938, 55555, 6379], 'TCP'), ([3383, 1233], 'UDP')],
    [([5222, 5228], 'TCP')],
    [(32764, 'TCP'), ([53413, 39889], 'UDP')],
    [([5555, 7547, 30005], 'TCP')],
    [([9100, 515, 631, 81, 10554], 'TCP')],
    [([8083, 5678], ['TCP', 'UDP']),
     ([8181, 4786, 8443, 8007, 8008, 8009, 23455, 5380, 4567], 'TCP'),
     (18999, 'UDP')],
    [([61001, 37215, 52869, 2000, 7676], 'TCP'), (9999, ['TCP', 'UDP'])],
    [([10000, 4444, 27374, 1050], ['TCP', 'UDP']), (1024, 'TCP')]
]
ip_duplicate_map = {
    37: ['43.164'], 38: ['54.143'], 40: ['70.211'],
    41: ['74.158', '74.159', '78.160'], 45: ['106.150'],
    48: ['133.54'], 49: ['143.72'],
    52: ['165.129'], 53: ['168.38'], 56: ['194.233'],
    57: ['204.97'], 58: ['209.48', '210.46', '214.43']
}
# port number: ([remove_ports], [([duplicate_ports], [duplicate_protocal])])
port_duplicate_map = {
    0: ([20, 21, 22, 23, 25, 53, 80, 81, 110, 111, 123],
        [(0, ['ICMP', 'AH', 'ESP']), (6, 'TCP'), (69, 'UDP')]),
    1: ([143, 135, 136, 137, 138, 139, 161, 162], []),
    2: ([443, 445, 389], [(500, 'UDP')]),
    3: (554, [(520, 'UDP'), ([515, 587, 631], 'TCP')]),
    4: (993, [((768, 783), 'ICMP'), (995, 'TCP')]),
    5: ([1050, 1080, 1720, 1701, 1723, 1025, 1026, 1027, 1028, 1029, 1720, 1900],
        [(1233, 'UDP'), ([1024, 1433, 2000], 'TCP')]),
    6: (2323, [(2048, ['ICMP', 'TCP']), (2869, 'TCP')]),
    7: ([3128, 3306], [(3389, 'TCP')]),
    8: ([4444, 5060, 5061], [([4153, 4567, 4786, 5000], 'TCP')]),
    9: ([5351, 5431, 5678], [((5800, 5810), 'TCP'), ((5900, 5910), 'TCP'), ((6000, 6063), 'UDP'),
                             ([5000, 5222, 5228, 5351, 5380, 5431, 5500, 5555, 5938], 'TCP')]),
    10: ([7070, 6588, 6667] + [x for x in range(6881, 7000)], [([6343, 6379], 'TCP')]),  # special action
    11: ([7779, 8080, 8081, 8083], [([7547, 7676, 8007, 8008, 8009, 8181], 'TCP')]),
    12: ([8888, 9080, 9090], [([8291, 8443, 8728, 8729, 9100], 'TCP')]),
    13: ([9999, 10000], []),
    14: (None, [(10554, 'TCP')]),
    15: (22010, [(20561, 'UDP'), (18067, 'TCP')]),
    16: ([27374] + [x for x in range(27000, 27051)], [(23455, 'TCP')]),
    17: (27374, [([30005, 32764], 'TCP')]),
    18: (None, [(37215, 'TCP'), (39889, 'UDP')]),
    20: (None, [(52869, 'TCP')]),
    21: (None, [(53413, 'UDP'), (55555, 'TCP')]),
    22: (None, [(61001, 'TCP')])
}


def convert_timestamp(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S').values.astype(np.int64) // 10 ** 9
    return df


def get_available_memory():
    with open('/proc/meminfo') as file:
        for line in file:
            if 'MemAva' in line:
                free_mem_in_kb = int(line.split()[1])
                return free_mem_in_kb / 1024 / 1024  # convert to Gb


def get_model_type(t: datetime):
    result = "weekday_" if t.weekday() < 5 else "weekend_"
    result += "morning" if 8 <= t.hour < 20 else "evening"
    return result


def get_index(flatten_idx):
    third_idx = flatten_idx % 64
    second_idx = ((flatten_idx - third_idx) // 64) % 64
    first_idx = ((flatten_idx - third_idx - 64 * second_idx) // 64 ** 2) % 64
    return first_idx, second_idx, third_idx


def timestr_to_timestamp(time_str, str_format='%Y-%m-%d %H:%M:%S'):
    return int(datetime.strptime(time_str, str_format).replace(tzinfo=timezone.utc).timestamp())


# This VM has 16 cores.
def parallelize_dataframe(df, func, n_cores=16):
    # split the whole dataframe into n_cores parts and process them separately.
    df_split = np.array_split(df, n_cores)
    with Pool(n_cores) as pool:
        df = pd.concat(pool.map(func, df_split))
    return df


def compute_tensor_byte(t, df, save_dir, **kwargs):
    print(t)
    return_tensor = kwargs.get("return_tensor", False)
    time_interval = kwargs.get("time_interval", 60)
    device = kwargs.get("device", "cpu")

    tensor = torch.zeros([len(ip_lst), len(ip_lst), len(num_bytes_lst)]).to(device)
    # filter the time given specific time interval
    df_t = df[df["timestamp"].between(t, t + time_interval - 1)]

    for id1 in range(len(ip_lst)):
        df_d1 = filter_ip(df_t, id1, True)
        if df_d1.shape[0] == 0:
            continue
        for id2 in range(len(ip_lst)):
            df_d2 = filter_ip(df_d1, id2, False)
            if df_d2.shape[0] == 0:
                continue
            for id3, num_bytes in enumerate(num_bytes_lst):
                if id3 == 63:
                    tensor[id1, id2, id3] = df_d2[df_d2["number_of_bytes"] >= num_bytes].shape[0]
                else:
                    tensor[id1, id2, id3] = df_d2[df_d2["number_of_bytes"].between(*num_bytes)].shape[0]
    if return_tensor:
        return tensor
    else:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(tensor, os.path.join(save_dir, f'tensor{t}.pt'))
        

def compute_tensor(t, df, save_dir, **kwargs):
    print(t)
    return_tensor = kwargs.get("return_tensor", False)
    time_interval = kwargs.get("time_interval", 60)
    device = kwargs.get("device", "cpu")

    tensor = torch.zeros([len(ip_lst), len(ip_lst), len(num_bytes_lst)]).to(device)
    # filter the time given specific time interval
    df_t = df[df["timestamp"].between(t, t + time_interval - 1)]

    for id1 in range(len(ip_lst)):
        df_d1 = filter_ip(df_t, id1, True)
        if df_d1.shape[0] == 0:
            continue
        for id2 in range(len(ip_lst)):
            df_d2 = filter_ip(df_d1, id2, False)
            if df_d2.shape[0] == 0:
                continue
            for id3 in range(len(dest_port_lst)):
                tensor[id1, id2, id3] = filter_dest_port(df_d2, id3).shape[0]
    if return_tensor:
        return tensor
    else:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(tensor, os.path.join(save_dir, f'tensor{t}.pt'))
        

def filter_time(df, start, end):
    return df[df["timestamp"].between(start, end)]


def filter_ip(df, ip_idx, src=True):
    if df.shape[0] == 0:
        return df
    ip_type = "src_IP" if src else "dest_IP"
    ip = ip_lst[ip_idx]

    if ip_idx < 16:
        # x1.x2
        df_t = df[df[ip_type].apply(lambda x: (('.'.join(x.split('.')[:2])) == ip))]
    elif 16 <= ip_idx < 32:
        # x1.x2.x3
        df_t = df[df[ip_type].apply(lambda x: (('.'.join(x.split('.')[:3])) == ip))]
    else:
        # 32 ip ranges
        # src_ip is a ip range.
        df_t = df[df[ip_type].apply(lambda x: (ip[0] <= int(x[:x.find('.')]) <= ip[1]))]
        if df_t.shape[0] == 0:
            return df_t

        # exclude duplicate IP
        if ip_idx == 37:
            # (40, 47)
            df_t = df_t[df_t[ip_type].apply(lambda x: (('.'.join(x.split('.')[:3])) not in ip_lst[16:32]) and (('.'.join(x.split('.')[:2])) not in ip_duplicate_map[ip_idx]))]       
        elif ip_idx in [38, 40, 41, 45, 48, 49, 52, 53, 56, 57, 58]:
            df_t = df_t[df_t[ip_type].apply(lambda x: (('.'.join(x.split('.')[:2])) not in ip_duplicate_map[ip_idx]))]
    return df_t


def filter_dest_port(df, dest_port_idx):
    dest_port = dest_port_lst[dest_port_idx]
    if dest_port_idx < 23:
        # port ranges
        # reset df_d2_t for query.
        df_t = df[df["dest_port"].between(*dest_port)]
        if df_t.shape[0] == 0:
            return df_t
        if dest_port_idx != 19:
            dup_ports, group_ports = port_duplicate_map[dest_port_idx]
            # remove duplicate ports
            if isinstance(dup_ports, int):
                df_t = df_t[~(df_t["dest_port"] == dup_ports)]
            elif isinstance(dup_ports, list):
                df_t = df_t[~df_t["dest_port"].isin(dup_ports)]
            # remove special ports
            for ports, protocols in group_ports:
                if df_t.shape[0] == 0:
                    return df_t
                if isinstance(ports, int):
                    port_mask = df_t["dest_port"] == ports
                elif isinstance(ports, tuple):
                    port_mask = df_t["dest_port"].between(*ports)
                else:
                    port_mask = df_t["dest_port"].isin(ports)
                if isinstance(protocols, str):
                    protocol_mask = df_t["protocol"] == protocols
                else:
                    protocol_mask = df_t["protocol"].isin(protocols)
                df_t = df_t[~(port_mask & protocol_mask)]
    elif dest_port_idx == 49:
        # updated 07/13/2020
        # port 6667 related to botnet
        df_t = df[(df["dest_port"] == dest_port)]
    else:
        mask = df.index < 0
        for ports, protocols in dest_port:
            if isinstance(ports, int):
                port_mask = df["dest_port"] == ports
            elif isinstance(ports, tuple):
                port_mask = df["dest_port"].between(*ports)
            else:
                port_mask = df["dest_port"].isin(ports)
            if isinstance(protocols, str):
                protocol_mask = df["protocol"] == protocols
            else:
                protocol_mask = df["protocol"].isin(protocols)
            mask |= (port_mask & protocol_mask)
        df_t = df[mask]
    return df_t


def filter_bytes(df, bytes_idx):
    if bytes_idx == 63:
        return df[df["number_of_bytes"] >= num_bytes_lst[bytes_idx]]
    return df[df["number_of_bytes"].between(*num_bytes_lst[bytes_idx])]


def filter_uniq(df, forward_tuple, reverse_tuple, unique_tuple):
    df_t = df[
        df[forward_tuple].apply(tuple, axis=1).isin(unique_tuple)
        |
        df[reverse_tuple].apply(tuple, axis=1).isin(unique_tuple)
    ]
    return df_t
