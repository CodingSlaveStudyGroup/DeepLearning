import numpy as np
import matplotlib.pyplot as plt
from week_5_optimizer import *


def fx(x):
    return 10 * np.where(x == 0.0, 1.0, np.sin(x) / x)


def dfdx(x):
    return 10 * np.where(x == 0.0, 0.0, (np.cos(x) / x) - (np.sin(x) / np.square(x)))


if __name__ == '__main__':
    optimizers = []
    optimizers.append(SGDOptimizer(eta=0.01))
    optimizers.append(SGDOptimizer(eta=1.1))
    optimizers.append(SGDOptimizer(eta=1.2))

    N = 16
    x_logs = []
    y_logs = []
    colors = ['r', 'g', 'b']
    for typ in range(0, len(optimizers)):
        x_log = []
        y_log = []
        param_dict = {'x': np.array([2.0])}
        grad_dict = {'x': np.zeros(1)}
        for cnt in range(0, N):
            x_log.append(np.copy(param_dict['x']))
            y_log.append(fx(param_dict['x']))
            grad_dict['x'] = dfdx(param_dict['x'])
            optimizers[typ].optimize(param_dict, grad_dict)
        x_logs.append(x_log)
        y_logs.append(y_log)

    x = np.linspace(0, 8, 64)
    y = fx(x)

    # draw fx
    ax = plt.subplot(121) # 1: nrow, 2: ncol, 3: index of axes
    plt.plot(x, y)

    # draw trajectory
    for typ in range(0, len(optimizers)):
        plt.plot(x_logs[typ], y_logs[typ], color=colors[typ], marker='o', label='η='+str(optimizers[typ].eta))
    plt.legend()

    # draw
    ax = plt.subplot(122)
    for typ in range(0, len(optimizers)):
        plt.plot(np.arange(0, N, 1), y_logs[typ], c=colors[typ], label='η='+str(optimizers[typ].eta))
    plt.legend()

    plt.show()