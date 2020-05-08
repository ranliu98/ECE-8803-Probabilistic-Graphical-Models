import numpy as np
import os
import scipy.io as sio

HW5_data_file = os.path.join("data-HWK2-2020","Data-Problem-5","diseaseNet.mat")
data = sio.loadmat(HW5_data_file)["pot"]

print(data)

D=20
S=40