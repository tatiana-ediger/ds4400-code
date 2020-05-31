ML HW1 README

There are two python files included in my submission: lr-gradient-descent.py and lr-closed-form.py

The first file (gradient descent) first graphs the 3d plot of the multivariable function f given in the assignment. Then an implementation of gradient descent is defined, and later called on a set of points at different learning rates. The plots are generated for each of these three cases. The imports in this file are are numpy, matplotlib.pyplot, and Axes3D from mpl_toolkits.mplot3D. In order to generate the plots and the results, just run the file and the output will be 4 plots and the sequence of points and a statement about convergence or divergence for each of the cases.

The second file (closed form) defines a function to find theta given a set of training data. This function is called with a given set of training data, and then this data is plotted in 2d alongside the estimated line y = (theta*)x. The imports in this file are: numpy and matplotlib.pyplot. In order to generate the plots and results, simply run the file and the output will be one plot and the calculated value of theta for the given training data.

Screenshots of all five plots are also included.
