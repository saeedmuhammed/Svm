import numpy as np
import pandas as pd;


def my_svm(dataset, output):

    rate = 0.0001 # step for gradient descent
    iterats = 1000 # no of iterations
    lamda = 1/iterats
    weights = np.zeros(dataset.shape[1]) # Create an array for storing the weights
    b = 0

    for iterats in range(1,iterats):

        for n, data in enumerate(dataset):

            if ((output[n] * np.dot(data, weights) + b) < 1):
                weights = weights + rate * (np.dot(data , output[n]) - (2 * lamda * weights) )  #if not correctly
                b = b + rate * (output[n] - (2 * lamda * b))


            else:
                weights = weights - rate * (2 * lamda * weights)   #if correctly
                b = b - rate * (2 * lamda * b)




    return weights , b




def predict(test_data,weights,b):
    results = []
    for data in test_data:
        result = np.dot(data,weights) + b
        if(result < 0):
            results.append(-1)
        else:
            results.append(1)

    return results
def accuracy (y,result):
    c=0
    for i in range (0,len(result)):

        if (y[i] == result[i]):
            c=c+1

    return c/len(result)


Path = 'heart.csv';
data = pd.read_csv(Path, header=0, usecols=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs','restecg', 'thalach' , 'exang', 'oldpeak','slope','ca','thal','target'])

df = data.sample(frac=1) #shuffle data

trainData=df[ : int(len(data)* (80/100))] #get 80 of dataset for traing
testData=df[  len(trainData) :  len(data) ]  #get the rest for testing




# Separating X (Training Set) from Y (Target Variable).
cols = trainData.shape[1];
x = trainData.iloc[:, 0: cols - 1];
y = trainData.iloc[:, cols - 1 :cols];

xt = testData.iloc[:, 0: cols - 1];
yt = testData.iloc[:, cols - 1 :cols];


# Matrix X and Y To make -> Convert From Data Frames into numpy Matrices
x = np.array(x.values);
y = np.array(y.values);
y=np.where(y==0,-1,1) #change output values from 0 to -1 to work in svm


xt = np.array(xt.values);
yt = np.array(yt.values);
yt=np.where(yt==0,-1,1)

ytrain=[]
ytest=[]
for i in range(0, len(y)): #make output in 1D arr not 2D
    ytrain.append(y[i][0])

for i in range(0, len(yt)):
    ytest.append(yt[i][0])




weights , b = my_svm(x,ytrain)


result = predict(xt, weights,b)

print(accuracy(ytest,result))

print(result)
print(ytest)

print(len(result))
print(len(ytest))


