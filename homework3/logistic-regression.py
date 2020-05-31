#  Exercise 4: Logistic Regression Implementation

import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from scipy.io import loadmat
dataset = loadmat('hw03_dataset.mat')
dataset2 = loadmat('hw04_data.mat')
print(dataset2)

#  split up the dataset into training and testing
training_x = dataset.get('X_trn')  # 126 x 2 array
training_y = dataset.get('Y_trn')  # 126 x 1 array
testing_x = dataset.get('X_tst')  # 14 x 2 array
testing_y = dataset.get('Y_tst')  # 14 x 1 array

train = []
i = 0
while i <= 125:
    train.append((training_x[i], training_y[i]))
    i += 1


#  Takes in a dataset of the form {(x1, y1), ..., (xn, yn)} and outputs the weight vector w, and the bias b in the
#  logistic regression model y = sigma * ((w^T)x + b)
def logistic_regr(training):

    #  splits the given training data into arrays with it's x and y values
    x_train = []
    y_train = []
    j = 0
    while j < len(training):
        x_train.append(training[j][0])
        y_train.append(training[j][1])
        j += 1

    #  creates and trains the logistic regression model
    logisticRegr = LogisticRegression()
    logisticRegr.fit(training_x, training_y.ravel())

    w = logisticRegr.coef_
    b = logisticRegr.intercept_

    return w, b


#  part b:

#  get w, b from the training data
log_reg_results = logistic_regr(train)
w1 = log_reg_results[0]
b1 = log_reg_results[1]
#  w1 = [[ 3.10947174 -1.35961301]]
#  b1 = [2.15266207]

#  Reporting classification error on the training set:
logr = LogisticRegression()
logr.fit(training_x, training_y.ravel())
train_predictions = logr.predict(training_x)

train_diff = 0
k = 0
while k < 125:
    if train_predictions[k] != training_y[k][0]:
        train_diff += 1
    k += 1

train_error = (train_diff / len(training_x)) * 100
print(train_error)

#  the training classification error is 3.175%


#  Reporting classification error on the testing set:
test_predictions = logr.predict(testing_x)

test_diff = 0
m = 0
while m < 14:
    if test_predictions[m] != testing_y[m][0]:
        test_diff += 1
    m += 1

test_error = (test_diff / len(testing_x)) * 100
print(test_error)

#  the testing classification error is 7.143%


#  plotting the training data:
training_x_1 = []
training_x_2 = []
for x in training_x:
    training_x_1.append(x[0])
    training_x_2.append(x[1])

print(w1[0][0])


def log_func(x):
    return (0.5 - w1[0][0] * x - b1) / w1[0][1]


X = np.linspace(-2, 2)
Y = log_func(X)

plt.scatter(training_x_1, training_x_2, c=training_y.ravel(), s=10)
plt.plot(X, Y)
plt.show()

#  plotting the testing data:
testing_x_1 = []
testing_x_2 = []
for x in testing_x:
    testing_x_1.append(x[0])
    testing_x_2.append(x[1])

plt.scatter(testing_x_1, testing_x_2, c=testing_y.ravel(), s=10)
plt.plot(X, Y)
plt.show()