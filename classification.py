# Import the libraries
import pandas as pd
import numpy as np

# Load the data
train_data = pd.read_csv('train_data.csv').values
train_labels = pd.read_csv('train_labels.csv').values
test_data = pd.read_csv('test_data.csv').values


def get_data_means(X):

    # Rhythm patterns: 1-168
    # Chroma: 169-216
    # MFCCs: 217-264

    data = np.zeros((X.shape[0], 3))
    for i in range(0, X.shape[0]):
        data[i][0] = np.mean(X[i][0:167])
        data[i][1] = np.mean(X[i][168:215])
        data[i][2] = np.mean(X[i][216:264])
    
    column_maxes = np.amax(data, axis=0)

    data[:,0] = data[:,0] / column_maxes[0]
    data[:,1] = data[:,1] / column_maxes[1]
    data[:,2] = data[:,2] / column_maxes[2]

    return data


def sigmoid_func(z): 

    sigmoid = 1/(1+np.exp(-z))

    return sigmoid

def gradient(X,y,w):

    N= np.size(X,0)

    grad =  (1/N)*np.dot(X.T,sigmoid_func(np.dot(X,w.T))-y)

    return grad

def logisticRegression_func(X,y,step_size, K):
    
    N = X.shape[0]
    d = X.shape[1]
    # Initialize w as 1xd array.
    w = np.zeros((1,d))
    loss = float('inf')
    loss_list = []
    for i in range(K):

        grad = gradient(X,y,w)
        w = w - step_size*grad.T # 1xd
        z = np.dot(X,w.T) # Nx1

        h = sigmoid_func(z) # Nx1
        loss_1 = - np.dot( y.T, np.log(h) ) # 1x1 y^T*log(h(z))
        loss_0 = - np.dot( (np.ones((N,1))-y).T, np.log(np.ones((N,1))-h) ) # 1x1 (1-y)^T*log(1-h(z))
        loss = loss_1 + loss_0
        loss_list.append(loss[0]/N)

    
    return loss_list, w

def get_labels(y, genre):

    labels = np.zeros((y.shape[0], 1))
    for i in range(y.shape[0]):
        if(y[i] == genre):
            labels[i] = 1
        else:
            labels[i] = 0
            
    return labels



y_predict = []
step_size = 1
num_iter = 300


X_train = get_data_means(train_data)
X_test = get_data_means(test_data)


for i in range(1,11):
    
    _, w_opt = logisticRegression_func(X_train, get_labels(train_labels, i), step_size, num_iter)
    y_predict_i = sigmoid_func(np.dot(X_test, w_opt.T))  

    N = y_predict_i.shape[0]
    if not len(y_predict):
        y_predict = np.zeros((N,11))
        y_predict[:,i] = y_predict_i.reshape(X_test.shape[0])
    else:
        y_predict[:,i] = y_predict_i.reshape(X_test.shape[0])

y_hat = y_predict.argmax(axis=1)
print(y_hat)
