from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

dataset = loadmat('hw03_dataset.mat')

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

#  part a

#  takes in a training dataset and testing set and outputs the parameters of the hyperplane and classification of the
#  training data and test data into the class {-1, +1}
def svm(training, x_test):
    x_train = []
    y_train = []
    j = 0
    while j < len(training):
        x_train.append(training[j][0])
        y_train.append(training[j][1])
        j += 1

    svm_clf = LinearSVC()

    svm_clf.fit(x_train, y_train)

    y_pred = svm_clf.predict(x_test)

    #  parameters of the hyperplane
    w1 = svm_clf.coef_
    b1 = svm_clf.intercept_

    #  predictions
    train_predictions_pre = svm_clf.predict(training_x)
    test_predictions_pre = svm_clf.predict(x_test)

    train_predictions = []
    for t in train_predictions_pre:
        train_predictions.append(2*t - 1)

    test_predictions = []
    for t in test_predictions_pre:
        test_predictions.append(2*t - 1)

    return w1, b1, train_predictions, test_predictions

#  part b

svm_results = svm(train, testing_x)
w1 = svm_results[0]
b1 = svm_results[1]
train_pred = svm_results[2]
test_pred = svm_results[3]

#  Reporting classification error on the training set:
train_diff = 0
k = 0
while k < 125:
    if train_pred[k] != 2*training_y[k][0] - 1:
        train_diff += 1
    k += 1

train_error = (train_diff / len(training_x)) * 100
print(train_error)

#  the training classification error is 2.381%


#  Reporting classification error on the testing set:
test_diff = 0
m = 0
while m < 14:
    if test_pred[m] != 2*testing_y[m][0] - 1:
        test_diff += 1
    m += 1

test_error = (test_diff / len(testing_x)) * 100
print(test_error)

#  the testing classification error is 0.0%

#  plotting the training data:
training_x_1 = []
training_x_2 = []
for x in training_x:
    training_x_1.append(x[0])
    training_x_2.append(x[1])

print(w1[0][0])


def svm_func(x):
    return -1 * (w1[0][0] * x + b1) / w1[0][1]


X = np.linspace(-2, 2)
Y = svm_func(X)

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