import numpy as np
import matplotlib.pyplot as plt


# Hyper Parameters
num_epochs = 60
learning_rate = 0.001

# Toy Dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# initialize parameters
theta = np.array([0])
b = np.array([0])

# run gradient descent and print out loss
for epoch in range(num_epochs):
    # change this part
    grad_theta = 0
    grad_beta = 0

# plot predictions
plt.plot(x_train, y_train, 'ro', label='Original data')

# get your final predictions and plot it
plt.legend()
plt.show()
