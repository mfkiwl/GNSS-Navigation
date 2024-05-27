import requests
import gzip   

try:
    obs = gzip.decompress(requests.get("https://cddis.nasa.gov/archive/gnss/data/daily/2021/342/21p/BRDM00DLR_S_20213420000_01D_MN.rnx.gz").content) # get obs and decompress    
        # write nav data
    print(obs)
except Exception as e:
    print(e)