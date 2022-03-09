import pandas as pd
from util import header


if __name__ == "__main__":
    with open("august.week1.tmp.csv", "w") as f:
        f.write("")
        
    cols = ['duration', 'src_port', 'dest_port', 'forwarding_status', 
            'type_of_service', 'packets_exchanged', 'number_of_bytes']
    i = 0
    for df in pd.read_csv("august.week1.csv", names=header, chunksize=15000000):
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(axis=0, how='any')
        df = df.astype({
            'src_port': 'int32', 
            'dest_port': 'int32',  
            "forwarding_status": "int32", 
            "type_of_service": "int32", 
            "packets_exchanged": "int32",
            'number_of_bytes': 'int32'
        })
        df.to_csv("august.week1.tmp.csv", mode="a", index=False, header=False) 
        i += 1
        print(i)
