import numpy as np


def running_average(x, window):
    return np.convolve(x, np.ones(window) / window, mode="valid")


print([i for i in range(10)])
print(running_average([i for i in range(10)], 3))
