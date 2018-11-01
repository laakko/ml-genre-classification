# Import the libraries
import pandas as pd
import numpy as np

# Load the data
train_data = pd.read_csv('train_data.csv').values
train_labels = pd.read_csv('train_labels.csv').values
test_data = pd.read_csv('test_data.csv').values

# Rhythm patterns: 1-168
# Chroma: 169-216
# MFCCs: 217-264


data_means = np.zeros((test_data.shape[0], 3))
#print(data_means)

print(test_data.shape[0])

for i in range(0, test_data.shape[0]):
    data_means[i][0] = np.mean(test_data[i][0:167])
    data_means[i][1] = np.mean(test_data[i][168:215])
    data_means[i][2] = np.mean(test_data[i][216:264])


print(data_means)
print(data_means.shape)



