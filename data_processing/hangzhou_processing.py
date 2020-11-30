import pandas as pd
import numpy as np
import datetime
import os

# Change to target data directory
delta_t = 15
inflow_t = "outflow" + str(delta_t)
os.chdir("/home/jjou/data/data_hangzhou/flow_data/flow-data-" + str(delta_t))

# Read all npz files
files = ["hangzhou_subway_" + "2019-01-" + str(i).zfill(2) + "_outflow1" + ".npy" for i in range(1, 26)]

# Generate data(columns, data)
stand_num = 81
time_points = int(105 * 10 / delta_t)
data = {}

for i in range(stand_num):
    # Read each files
    for file in files:
        day_data = np.load(file)
        # print(day_data.shape)
        # Read 105 time points
        for j in range(time_points):
            # Use `i` as stand no(start at 0)
            if i not in data:
                data[i] = [day_data[j][i]]
            else:
                data[i].append(day_data[j][i])

# Create index (data + time)
time_stamps= (pd.DataFrame(columns=['NULL'],
                  index=pd.date_range('2019-01-01T06:00:00Z', '2019-01-25T23:30:00Z',
                                      freq=str(delta_t) + "T"))
       .between_time('06:00','23:15')
       .index.strftime('%Y-%m-%d %H:%M:%S')
       .tolist()
)
print(len(time_stamps))

for i in range(len(time_stamps)):
    time_stamps[i] = datetime.datetime.strptime(time_stamps[i], '%Y-%m-%d %H:%M:%S')
    time_stamps[i] = np.datetime64(time_stamps[i])

print(type(time_stamps[0]))

# Generate dataframe and save as h5 files
df = pd.DataFrame(data, index =time_stamps) 
os.chdir("/home/jjou/sunjiahui/Graph-WaveNet/data_processing/processed_data/hangzhou")
df.to_hdf("outflow" + str(delta_t) + '.h5', "outflow"+str(delta_t))
df.to_csv("outflow" + str(delta_t) + '.csv')


