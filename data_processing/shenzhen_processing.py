import pandas as pd
import numpy as np
import datetime
import os

# Change to target data directory
delta_t = 30
inflow_t = "outflow" + str(delta_t)
os.chdir("../../../data/data_shenzhen/flow_data/flow-data-" + str(delta_t))

# Read all npz files
files = ["shenzhen_subway_201706" + str(i).zfill(2) + "_outflow1.npy" for i in range(1, 31)]

# Generate data(columns, data)
stand_num = 165
time_points = int(15 * 66 / delta_t)
data = {}

for i in range(stand_num):
    # Read each files
    for file in files:
        day_data = np.load(file)
        # Read 105 time points
        for j in range(time_points):
            # Use `i` as stand no(start at 0)
            if i not in data:
                data[i] = [day_data[j][i]]
            else:
                data[i].append(day_data[j][i])

# Create index (data + time)
time_stamps= (pd.DataFrame(columns=['NULL'],
                  index=pd.date_range('2017-06-01T06:30:00Z', '2017-06-30T23:00:00Z',
                                      freq=str(delta_t) + "T"))
       .between_time('06:30','22:30')
       .index.strftime('%Y-%m-%d %H:%M:%S')
       .tolist()
)

# print(time_stamps)
print(len(time_stamps))

for i in range(len(time_stamps)):
    time_stamps[i] = datetime.datetime.strptime(time_stamps[i], '%Y-%m-%d %H:%M:%S')
    time_stamps[i] = np.datetime64(time_stamps[i])

# Generate dataframe and save as h5 files
df = pd.DataFrame(data, index =time_stamps) 
os.chdir("/home/jjou/sunjiahui/Graph-WaveNet/data_processing/processed_data/shenzhen")
df.to_hdf("outflow" + str(delta_t) + '.h5', "outflow"+str(delta_t))

df.to_csv("outflow" + str(delta_t) + '.csv')
