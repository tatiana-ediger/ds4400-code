import numpy as np
import matplotlib.pyplot as plt
import math as m
from mpl_toolkits.mplot3d import Axes3D


# The function to be graphed
def f(x1, x2):
    return ((-2 * x1 * x1) - (3 * x1 * x2) + (2 * x2 * x2)) * np.sin(x1)


# calculates the gradient of a tuple containing two points by returning a tuple with the derivative with respect to
# x1 and the derivative with respect to x2 when you plug in the given initial point.
def gradient(initial_pt):
    x1 = initial_pt[0]
    x2 = initial_pt[1]
    return np.math.cos(x1) * ((-2 * x1 * x1) - (3 * x1 * x2) + (2 * x2 * x2)) + \
           (np.math.sin(x1) * ((-4 * x1) + (3 * x2))), np.math.sin(x1) * (-3 * x1 + 4 * x2)


# Calculates the l2_norm of the given 2 points
def l2_norm(p1, p2):
    diff1 = p1[0] - p2[0]
    diff2 = p1[1] - p2[1]
    return np.sqrt((diff1 ** 2) + (diff2 ** 2))


# Executes the gradient descent given the initial point, the max number of iterations, and the learning rate. Returns
# the sequence of points obtained by the gradient descent.
def gradient_descent(initial_pt, max_iter, learning_rate):
    epsilon = 0.01
    points = [initial_pt]
    step_size = 1
    total_iter = 0
    curr_theta = initial_pt
    while step_size > epsilon and total_iter < max_iter and abs(curr_theta[0]) < 6 and abs(curr_theta[1] < 6):
        prev_theta = curr_theta
        curr_theta = (prev_theta[0] - learning_rate * (gradient(prev_theta))[0],
                      prev_theta[1] - learning_rate * (gradient(prev_theta))[1])
        step_size = l2_norm(curr_theta, prev_theta)
        total_iter += 1
        points.append(curr_theta)
    return points[-1], f(points[-1][0], points[-1][1])


# Plots the sequence of points.
def plot_gd(points, learning_rate):
    for pt in points[-5:]:
        plt.scatter(pt[0], pt[1])
    plt.title('Gradient Descent for Learning Rate of ' + str(learning_rate))
    plt.show()

all_points = []

gd1 = gradient_descent((-3, -4), 10000, 0.1)
all_points.append(gd1[0])
print(gd1)

gd2 = gradient_descent((4, -3), 10000, 0.1)
all_points.append(gd2[0])
print(gd2)

gd3 = gradient_descent((1, 5), 10000, 0.1)
all_points.append(gd3[0])
print(gd3)

gd4 = gradient_descent((-4, -3), 10000, 0.1)
all_points.append(gd4[0])
print(gd4)

gd5 = gradient_descent((-5, 5), 10000, 0.1)
all_points.append(gd5[0])
print(gd5)

print(all_points)


def plot_gd(points):
    for pt in points:
        plt.scatter(pt[0], pt[1])
    plt.title('Final Points Obtained using GD')
    plt.show()


# 1. Plot the 5 final points
plot_gd(all_points)
# All the points are not the same

# 2. Final Function Values:
# initial input |               final point                     |   function value at final point
# ______________________________________________________________________________________________
# (-3,-4)       |   (-6.09555291885903, -4.257498358047349)     |   -21.62189649893478
# (4, -3)       |   (9.082836215267214, -5.992971087203522)     |   23.517495626842194
# (1, 5)        |   (-6.424138925370329, 1.3658212389276003)    |   7.373549693221117
# (-4, -3)      |   (-7.79797985103361, -3.0)                   |   173.52633552412945
# (-5, 5)       |   (-10.483701352295181, 1.6437650386790152)   |   -141.85858430464185
# The best minimum value is -141.859, which occurs at the final point of an initial input of (-5, 5)


# 3. Visualize the function in 3D
# Graphing the function f(x1,x2) = (x1 - 2)^2 + (x2 - 3)^2
fig = plt.figure()
projection = fig.gca(projection='3d')
X = np.linspace(-6, 6)
Y = np.linspace(-6, 6)
X, Y = np.meshgrid(X, Y)
projection.plot_surface(X, Y, f(X, Y))
fig.show()
# From looking at this graph, we can see that the 5 points are not the same because there are various local
# extrema in the graph, so when we use gradient descent, we reach either the endpoints of the graph or one of
# other local extrema.
