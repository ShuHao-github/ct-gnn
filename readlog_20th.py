import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str)
args = parser.parse_args()

file = args.file_path
#file = "/home/hwl/hm/device-placement-master/model_ptb_pgnn_2gpu_i/GNN_I2GNN-2th_s2_ptb_2gpu"
print(file)
Total_time = []
NN_time = []
best_pl_runtime = []
for line in open(file,"r",encoding='UTF-8'):
    if "Total time" in line:
        #print(line)
        line = line.split(":")
        Total_time.append(float(line[1]))
    if "NN time" in line:
        #print(line)
        line = line.split(":")
        NN_time.append((float(line[1])))
    if "Episode best pl runtime" in line:
        #print(line)
        line = line.split(" ")
        #print((float(line[4])))
        best_pl_runtime.append((float(line[4])))

# print('Len(Total_time): ', len(Total_time))
# print('Len(NN_time): ', len(NN_time))
# print('Len(best_pl_runtime): ', len(best_pl_runtime))

# print(Total_time[:360])
# Total_time_v1 = np.reshape(Total_time,[20,360])
# Total_time_v2=0
# Total_time_v3=[]
# for i in range(5):
#     Total_time_v3.append(np.min(Total_time_v1[i]))
#     Total_time_v2 = Total_time_v2 + np.min(Total_time_v1[i])
#
# print('average total time: ', Total_time_v2/5)
# print('min total time: ', np.min(Total_time_v3))
# print('average total time: ', np.average(Total_time_v3))

# print(best_pl_runtime[:360])
best_pl_runtime_v1 = np.reshape(best_pl_runtime, [int(len(best_pl_runtime)/20), 20])
# print(best_pl_runtime_v1)
# print(best_pl_runtime_v1.shape)
Total_time_v1 = np.reshape(Total_time, [int(len(Total_time)/20), 20])
# print(Total_time_v1)
# print(Total_time_v1.shape)
NN_time_v1 = np.reshape(NN_time, [int(len(NN_time)/20), 20])
# print(NN_time_v1)
# print(NN_time_v1.shape)

best_pl_runtime_v2 = 0
best_pl_runtime_v3 = []

Total_time_v2=0
Total_time_v3=[]

NN_time_v2=0
NN_time_v3 = []

for i in range(int(len(best_pl_runtime)/20)):
    # best_pl_runtime1 = best_pl_runtime_v1[i].copy()
    # print('best_pl_runtime_v1[i]: ', best_pl_runtime_v1[i])
    # best_pl_runtime_v1[i].sort()
    # best_pl_runtime2 = best_pl_runtime_v1[i][0]
    # idx = list(best_pl_runtime1).index(best_pl_runtime2)

    best_pl_runtime_v3.append(best_pl_runtime_v1[i][19])
    Total_time_v3.append(Total_time_v1[i][19])
    NN_time_v3.append(NN_time_v1[i][19])

# print("NN time", NN_time_v3)
# print("Total time", Total_time_v3)
# print("run time", best_pl_runtime_v3)

    # print('idx: ', idx)
    # print(' best_pl_runtime_v1[i]: ',  best_pl_runtime_v1[i])
    # print('best_pl_runtime_v3: ', best_pl_runtime_v3)
    # print('Total_time_v3: ', Total_time_v3)
    # print('NN_time_v3: ', NN_time_v3)

    # best_pl_runtime_v2 = best_pl_runtime_v2 + best_pl_runtime_v1[i][idx]
    # Total_time_v2 = Total_time_v2 + Total_time_v1[i][idx]
    # NN_time_v2 = NN_time_v2 + NN_time_v1[i][idx]

# print('average total time: ', Total_time_v2 / 5)
# print('min total time: ', np.min(Total_time_v3))
print('average total time: ', np.average(Total_time_v3))

# print('average NN time: ', NN_time_v2 / 5)
# print('min NN time: ', np.min(NN_time_v3))
print('average NN time: ', np.average(NN_time_v3))

# print('average best_pl_runtime time: ', best_pl_runtime_v2 / 5)
# print('min best_pl_runtime time: ', np.min(best_pl_runtime_v3))
print('average best_pl_runtime time: ', np.average(best_pl_runtime_v3))



