import numpy as np

#Given You have a workload of size M*K, K*N
M=256

K=256
N=256
MK_array = np.array(np.int32, M*K)
KN_array = np.array
# Leading eleents have to be padded to sa scol and row
for i in M:
    input_array = np.array([[i]:i])




systolic_array_width = 4
rows, cols = input_array.shape
total_width = cols + systolic_array_width - 1

padded_array = np.zeros((rows, total_width), dtype=input_array.dtype)

for i in range(rows):
    padded_array[i, i:i+cols] = input_array[i]

print(padded_array)