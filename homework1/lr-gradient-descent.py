import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# The function to be graphed
def f(x, y):
    return (x - 2) ** 2 + (y - 3) ** 2


# Graphing the function f(x1,x2) = (x1 - 2)^2 + (x2 - 3)^2
fig = plt.figure()
projection = fig.gca(projection='3d')
X = np.linspace(0, 3)
Y = np.linspace(0, 3)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)
projection.plot_surface(X, Y, Z)
fig.show()


# calculates the gradient of a tuple containing two points by returning a tuple with the derivative with respect to
# x1 and the derivative with respect to x2 when you plug in the given initial point.
def gradient(initial_pt):
    return 2 * (initial_pt[0] - 2), 2 * (initial_pt[1] - 3)


# Calculates the l2_norm of the given 2 points
def l2_norm(p1, p2):
    diff1 = p1[0] - p2[0]
    diff2 = p1[1] - p2[1]
    return np.sqrt((diff1 ** 2) + (diff2 ** 2))


# Executes the gradient descent given the initial point, the max number of iterations, and the learning rate. Returns
# the sequence of points obtained by the gradient descent.
def gradient_descent(initial_pt, max_iter, learning_rate):
    epsilon = 0.000001
    points = [initial_pt]
    step_size = 1
    total_iter = 0
    curr_theta = initial_pt
    while step_size > epsilon and total_iter < max_iter:
        prev_theta = curr_theta
        curr_theta = (prev_theta[0] - learning_rate * (gradient(prev_theta))[0],
                      prev_theta[1] - learning_rate * (gradient(prev_theta))[1])
        step_size = l2_norm(curr_theta, prev_theta)
        total_iter += 1
        points.append(curr_theta)
    return points


# Plots the sequence of points, including the learning rate in the title of the plot.
def plot_gd(points, learning_rate):
    for pt in points:
        plt.scatter(pt[0], pt[1])
    plt.title('Gradient Descent for Learning Rate of ' + str(learning_rate))
    plt.show()


# Below outputs the summary of the points generated from my gradient descent implementation and a statement relaying
# the convergence/divergence for different inputs. The message that relays the number of steps until convergence
# includes and is counting the first point inputted as a step, and this is reflected in the graph as well.

# The results for the point (1,2) with a learning rate of 0.01 shows convergence after 510 steps.
gd1 = gradient_descent((1, 2), 600, 0.01)
print(gd1)
print('converges after ', len(gd1), ' steps')
plot_gd(gd1, 0.01)

# The results for the point (1,2) with a learning rate of 0.5 shows convergence after 3 steps.
gd2 = gradient_descent((1, 2), 600, 0.5)
print(gd2)
print('converges after ', len(gd2), ' steps')
plot_gd(gd2, 0.5)

# Since this learning rate is much too large for this algorithm, running it with as many points as above will cause a
# 'Result too large' OverflowError, which proves that it diverges. Therefore, I ran it with 120 iterations, which is the
# maximum number before the error. But we can clearly see that it diverges, since the points continue to get
# significantly larger.
gd3 = gradient_descent((1, 2), 120, 10.0)
print(gd3)
print('diverges')
plot_gd(gd3, 10.0)
