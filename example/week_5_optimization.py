import numpy as np
import matplotlib.pyplot as plt
from week_5_optimizer import *

class SimpleNetwork:
    def __init__(self):
        self.param_dict = {}
        self.grad_dict = {}
        self.init_param('W', np.random.normal(loc=0.0, scale=1e-2, size=1))
        self.init_param('b', np.zeros(1))

    def init_param(self, name, init_value):
        self.param_dict[name] = np.copy(init_value)
        self.grad_dict[name] = np.zeros_like(init_value)

    def estimate(self, x_data):
        # s := x * W + b
        # y := sigmoid(s)
        # Note that this is 0D problem. If you use 1D, you have to change * np.dot()
        return sigmoid(x_data * self.param_dict['W'] + self.param_dict['b'])

    # pnorm: Parameter Normalization
    def train(self, x_data, y_data, pnorm=0.0):
        # Forward
        # s := x_data * W + b
        # y_hat := sigmoid(s)
        y_hat = self.estimate(x_data)

        # loss := sum of every elements of 0.5 * (y_hat - y_data)^2
        # You can use average instead, if you have proper reasons.
        # But if you use average, you MUST divide dL/dy by batch size.
        diff = y_hat - y_data
        loss_err = 0.5 * np.sum(np.square(diff))
        loss_prm = 0.5 * pnorm * np.sum(np.square(self.param_dict['W']) + np.square(self.param_dict['b']))
        loss = loss_err + loss_prm
        # loss = 0.5 * np.average(np.square(diff), axis=0)

        # Backward
        # dL/dy := y_hat - y_data
        dLdy = diff
        # dLdy = diff / np.size(x_data, 0)

        # dL/ds := sigmoid'(s) * dL/dy
        dLds = y_hat * (1 - y_hat) * dLdy

        # dW := W^T @ dL/ds
        # db := dL/ds
        # Note that numpy's dLds has batch axis, so you have to sum all of these.
        self.grad_dict['W'] = np.sum(x_data * dLds, axis=0) + pnorm * self.param_dict['W']
        self.grad_dict['b'] = np.sum(dLds, axis=0) + pnorm * self.param_dict['b']

        return loss

    # This is similar to train, but it's different with shape modification.
    # This assumes that parameter's shape is [-1, 1], not simply [1]
    # DO NOT USE this for general purpose. It's for this week's only.
    def loss_function_test(self, x_data, y_data, pnorm=0.0):
        # Forward
        # s := x_data * W + b
        # y_hat := sigmoid(s)
        y_hat = self.estimate(x_data)

        # loss := sum of every elements of 0.5 * (y_hat - y_data)^2
        # You can use average instead, if you have proper reasons.
        # But if you use average, you MUST divide dL/dy by batch size.
        diff = y_hat - y_data
        loss_err = 0.5 * np.sum(np.square(diff), axis=1)
        loss_prm = 0.5 * pnorm * np.sum(np.square(self.param_dict['W']) + np.square(self.param_dict['b']))
        loss = loss_err + loss_prm
        # loss = 0.5 * np.average(np.square(diff), axis=0)

        return loss

# Sometimes I miss you, "int main(int argc, char** argv)"
if __name__ == '__main__':
    # Hyperparameter
    pnorm = 0.1
    N = 32
    T = 256

    # Prepare data.
    x_data = np.linspace(-2.0, 2.0, N)
    y_data = sigmoid(2*x_data - 1) + np.random.normal(loc=0.0, scale=5e-2, size=np.size(x_data))

    # Prepare structure.
    network = SimpleNetwork()
    opts = []
    opts.append(SGDOptimizer(eta=1e-3))
    opts.append(MomentumOptimizer(eta=1e-3, alpha=0.9))
    opts.append(AdagradOptimizer(eta=1e-2))
    opts.append(RMSPropOptimizer(eta=1e-2, rho=0.9))
    opts.append(AdadeltaOptimizer(rho=0.9))
    opts.append(AdamOptimizer(alpha=1e-2, beta1=0.9, beta2=0.999))
    opts.append(COCOBOptimizer(alpha=100))
    labels = ['SGD η=1e-3', 'Momentum η=1e-3, α=0.9',
              'Adagrad η=1e-2', 'RMSProp η=1e-2', 'Adadelta ρ=0.9',
              'Adam α=1e-2, β1=0.9, β=0.999', 'COCOB']

    # Train!
    logs = []
    W_logs = []
    b_logs = []
    for opt in opts:
        # Log loss trajectory for each optimizer, with same start point
        log = []
        W_log = []
        b_log = []
        network.init_param('W', np.array([-3.0]))
        network.init_param('b', np.array([2.0]))
        for t in range(0, T):
            loss = network.train(x_data, y_data, pnorm)
            log.append(loss)
            W_log.append(np.copy(network.param_dict['W']))
            b_log.append(np.copy(network.param_dict['b']))
            opt.optimize(network.param_dict, network.grad_dict)
        logs.append(log)
        W_logs.append(W_log)
        b_logs.append(b_log)

    # Show original data and its estimation
    fig0, ax0 = plt.subplots()
    ax0.plot(x_data, y_data, label='original')
    ax0.plot(x_data, network.estimate(x_data), label='estimated')
    plt.legend()

    # Show loss function trajectory
    t = np.arange(0, T)
    fig, ax = plt.subplots()
    for n in range(len(logs)):
        ls = np.array(logs[n])
        ax.plot(t, ls, label=labels[n])
    plt.legend()  # 범례를 표시함
    #plt.show()

    # Show contour trajectory
    zoom = 5.0
    res = 100
    cres = 20
    fig2, ax2 = plt.subplots()
    ax2.set_xlim(-zoom, zoom)
    ax2.set_ylim(-zoom, zoom)
    pw = np.linspace(-zoom, zoom, res) # 1D array
    pb = np.linspace(-zoom, zoom, res) # 1D array
    u, v = np.meshgrid(pw, pb) # 2D array
    network.init_param('W', np.reshape(u, (-1, 1)))
    network.init_param('b', np.reshape(v, (-1, 1)))
    z = network.loss_function_test(x_data, y_data, pnorm).reshape(res, res)
    ct = ax2.contour(pw, pb, z, cres)
    for n in range(len(logs)):
        plt.plot(W_logs[n], b_logs[n], label=labels[n])
        plt.scatter(W_logs[n], b_logs[n], s=3.0)
    plt.legend()
    plt.show()