import numpy as np
import statistics as stats
import matplotlib.pyplot as plt

from scipy.io import loadmat
dataset = loadmat('dataset_hw2.mat')
training_x = dataset.get('X_trn')
training_y = dataset.get('Y_trn')
testing_x = dataset.get('X_tst')
testing_y = dataset.get('Y_tst')
validation_x = dataset.get('X_val')
validation_y = dataset.get('Y_val')

training = []
i = 0
while i < np.size(training_x):
    training.append((training_x[i][0], training_y[i][0]))
    i += 1

testing = []
i = 0
while i < np.size(testing_x):
    testing.append((testing_x[i][0], testing_y[i][0]))
    i += 1

validation = []
i = 0
while i < np.size(validation_x):
    validation.append((validation_x[i][0], validation_y[i][0]))
    i += 1


def get_col_mean_std(matrix):
    num_rows = matrix.shape[0]
    num_cols = matrix.shape[1]

    column_means = []
    column_std = []

    transpose = matrix.transpose()
    i = 0
    while i < num_cols - 1:
        column_means.append(np.sum(transpose[i]) / num_rows)
        column_std.append(stats.stdev(transpose[i]))
        i += 1

    return column_means, column_std


def normalize1(matrix):
    num_cols = matrix.shape[1]

    column_means = get_col_mean_std(matrix)[0]
    column_std = get_col_mean_std(matrix)[1]

    # get the mean of a column
    for row in matrix:
        for j in range(num_cols - 1):
            (row[j] - column_means[j]) / column_std[j]

    return matrix


def normalize2(matrix, column_means, column_std):
    num_cols = matrix.shape[1]

    # get the mean of a column
    for row in matrix:
        for j in range(num_cols - 1):
            (row[j] - column_means[j]) / column_std[j]

    return matrix


def lin_reg(train, validation, test, s, n):
    if s == 0:
        return lin_reg_closed(train, validation, test, n)
    elif s == 1:
        return lin_reg_gradient(train, validation, test, n)


def lin_reg_closed(train, validation, test, n):
    train_size = np.size(train) // 2

    train_x_data = get_x_y(train)[0]
    train_y_data = get_x_y(train)[1]
    y_array = np.array(train_y_data).reshape(-1, 1)

    # Applying basis expansion to the x values in the training data set
    phi_array = basis_expansion(train_x_data, n)
    phi_matrix = np.array(phi_array).reshape(train_size, n + 1)

    # Implementation of the closed-form solution to find theta
    phi_transpose = phi_matrix.transpose()  # gets the transpose of x
    mult_tr_array = np.matmul(phi_transpose, phi_matrix)  # multiplies phi-transpose and phi
    inv_tr_array = np.linalg.pinv(mult_tr_array)  # takes the inverse of the product of phi-transpose and phi
    theta = np.matmul(inv_tr_array, np.matmul(phi_transpose, y_array))  # multiplies the inverse by phi-transpose and y
    # return the theta and the different regression errors
    return theta, errors(train, validation, test, theta, phi_matrix, n, 0)


def lin_reg_gradient(train, validation, test, n):
    train_size = np.size(train) // 2

    max_iter = 1000
    train_x_data = get_x_y(train)[0]
    train_y_data = get_x_y(train)[1]
    train_y_array = np.array(train_y_data).reshape(-1, 1)
    phi_array = basis_expansion(train_x_data, n)
    phi_matrix = np.array(phi_array).reshape(train_size, n + 1)
    normalized_phi = normalize1(phi_matrix)
    i = 0
    initial_theta = []
    while i < n + 1:
        initial_theta.append([0])
        i += 1
    learning_rate = 0.0000001
    epsilon = 0.01
    step_size = 1
    total_iter = 0
    curr_theta = initial_theta

    while step_size > epsilon and total_iter < max_iter:
        prev_theta = curr_theta
        curr_theta = prev_theta - learning_rate * np.matmul(normalized_phi.transpose(),
                                                            np.matmul(normalized_phi, prev_theta) - train_y_array)
        step_size = l2_norm(curr_theta, prev_theta)
        total_iter += 1
    return curr_theta, errors(train, validation, test, curr_theta, phi_matrix, n, 1)


# Calculates the l2_norm of the given 2 vectors
def l2_norm(v1, v2):
    total_sum = 0
    length = np.size(v1)
    j = 0
    while j < length:
        total_sum += (v1[j][0] - v2[j][0]) ** 2
        j += 1
    return np.sqrt(total_sum)


# Applies the basis expansion on a given dataset x for a certain n value.
# Takes each value x and a number n returns a matrix where each row is x^1 x^2 ... x^n 1.
def basis_expansion(x, n):
    phi_array = []
    for point in x:
        row_i = []
        for i in range(1, n + 1):
            row_i.append(point ** i)
        row_i.append(1)
        phi_array.append(row_i)
    return phi_array


def basis_expansion_for_x(x, n):
    row_i = []
    for i in range(1, n + 1):
        row_i.append(x ** i)
    row_i.append(1)
    return np.array(row_i)


# extracts the x and y values from a dataset and puts them in two arrays
def get_x_y(data):
    x_data = []
    y_data = []
    for point in data:
        x_data.append(point[0])
        y_data.append(point[1])
    return x_data, y_data


# Calculates the mean squared error of a vector (sum of all components divided by their length
def mse(vector):
    length = np.size(vector)
    ret_sum = 0
    for entry in vector:
        for i in entry:
            ret_sum += i * i
    return ret_sum / length


def errors(train, validation, test, theta, phi_matrix, n, s):
    validation_size = np.size(validation) // 2
    test_size = np.size(test) // 2

    train_y_data = get_x_y(train)[1]
    y_array = np.array(train_y_data).reshape(-1, 1)

    # Finding the regression error for the training data
    if s == 0:
        regression_error_train = mse(y_array - np.matmul(phi_matrix, theta))
    if s == 1:
        regression_error_train = mse(y_array - np.matmul(normalize1(phi_matrix), theta))

    # Applying basis expansion to the validation data set and then finding the regression error
    validation_x_data = get_x_y(validation)[0]
    validation_y_data = get_x_y(validation)[1]
    validation_y_array = np.array(validation_y_data).reshape(-1, 1)
    phi_validation = basis_expansion(validation_x_data, n)

    if s == 0:
        phi_matrix_validation = np.array(phi_validation).reshape(validation_size, n + 1)
    if s == 1:
        phi_matrix_validation = normalize2(np.array(phi_validation).reshape(validation_size, n + 1),
                                           get_col_mean_std(phi_matrix)[0], get_col_mean_std(phi_matrix)[1])
    regression_error_validation = mse(validation_y_array - np.matmul(phi_matrix_validation, theta))

    # Applying basis expansion to the test data set and then finding the regression error
    test_x_data = get_x_y(test)[0]
    test_y_data = get_x_y(test)[1]
    test_y_array = np.array(test_y_data).reshape(-1, 1)
    phi_test = basis_expansion(test_x_data, n)
    if s == 0:
        phi_matrix_test = np.array(phi_test).reshape(test_size, n + 1)
    if s == 1:
        phi_matrix_test = normalize2(np.array(phi_test).reshape(test_size, n + 1),
                                     get_col_mean_std(phi_matrix)[0], get_col_mean_std(phi_matrix)[1])
    regression_error_test = mse(test_y_array - np.matmul(phi_matrix_test, theta))

    return regression_error_train, regression_error_validation, regression_error_test


def closed_form_expansion(n):
    response = lin_reg(training, validation, testing, 0, n)
    return response[1][0], response[1][2]


for x in range(1, 10):
    plt.scatter(x, closed_form_expansion(x)[0])
    plt.scatter(x, closed_form_expansion(x)[1])
plt.title('Closed Form training and testing error as a function of n = {1,9}')
plt.show()

# 2. When we graph the training and testing errors as a function of n, we see that originally the errors are higher for
# 1,2 and then they reach nearly zero.

# Plot the training data vs. learned function:
def plot_training_vs_learned(n):
    k = 0
    while k < np.size(training_x):
        plt.scatter(training_x[k][0], training_y[k][0])
        plt.scatter(training_x[k][0], np.matmul(lin_reg(training, validation, testing, 0, n)[0].transpose(),
                                                basis_expansion_for_x(training_x[k][0], m)))
        k += 1
    plt.title(str('Learned funciton vs training data for n=' + str(n)))
    plt.show()

print(basis_expansion_for_x(training_x[2][0], 3))

for m in range(1, 10):
    plot_training_vs_learned(m)

print(np.shape)



