import numpy as np
import matplotlib.pyplot as plt

# The given training data for this model.
training = [(0.1, 0.15), (0.5, 0.4), (0.9, 0.85), (1.5, 1.62), (-0.2, -0.17), (-0.5, -0.42)]


# Finds and returns the value of theta using the closed form solution when given training data.
# The closed form solution is: (((X^T)X)^-1)(X^T)Y
def find_theta(train):
    x_data = []
    y_data = []
    for point in train:
        x_data.append(point[0])
        y_data.append(point[1])

    x_array = np.array(x_data).reshape(-1, 1)
    y_array = np.array(y_data).reshape(-1, 1)

    x_transpose = x_array.transpose()  # gets the transpose of x
    mult_tr_array = np.matmul(x_transpose, x_array)  # multiplies x-transpose and x
    inv_tr_array = np.linalg.pinv(mult_tr_array)  # takes the inverse of the product of x-transpose and x
    theta = np.matmul(inv_tr_array, np.matmul(x_transpose, y_array))  # multiplies the inverse by x-transpose and y

    return theta


# Plots the graph of the given points and the estimated line using the given theta.
def plot_theta(train, theta):
    for pt in train:
        plt.scatter(pt[0], pt[1])
    x = np.arange(-10, 10, 0.1)
    plt.ylim(-10, 10)
    plt.plot(x, x * theta[0][0])
    plt.title('Training Data vs Regression Model')
    plt.show()


# finds the value of theta for the training data, prints it out, and plots a graph of the points and the estimated line
# using theta.
theta_calculated = find_theta(training)
print(theta_calculated)
plot_theta(training, theta_calculated)
