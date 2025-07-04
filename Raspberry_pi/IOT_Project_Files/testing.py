from modelcode import AutoEncoder,intrpl
import numpy as np
ae = AutoEncoder()
ecg_data = np.load("test_data.npy")
print(ecg_data.shape)

for i in range(8):
    result = ae.call(intrpl(ecg_data[i].reshape(1,100)))
    print(result)
