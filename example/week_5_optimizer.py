import numpy as np


# Almost every codes were written by Phryxia

# Stochastic Gradient Descent Optimizer
#
#   θ := θ - η * dL/dθ
#
#   Hyperparameter
#       η: Learning rate. Typical value is 1e-3 ~ 1e-4
class SGDOptimizer:
    def __init__(self, eta=1e-3):
        self.eta = eta

    def optimize(self, param_dict, grad_dict):
        for key in param_dict:
            param_dict[key] -= self.eta * grad_dict[key]


# Momentum Optimizer(1999)
#
#   v  := α * v - η * dL/dθ
#   θ := θ + v
#
#   Hyperparameter
#       η: Learning rate. Typical value is 1e-3 ~ 1e-4
#       α: Momentum rate. Typical value is 0.3 ~ 0.9
class MomentumOptimizer:
    def __init__(self, eta=1e-3, alpha=0.5):
        self.eta = eta
        self.alpha = alpha
        self.v = None

    def optimize(self, param_dict, grad_dict):
        self.__check_variables(param_dict)
        for key in param_dict:
            self.v[key] = self.alpha * self.v[key] - self.eta * grad_dict[key]
            param_dict[key] += self.v[key]

    def __check_variables(self, param_dict):
        if self.v is None:
            self.v = {}
            for key, param in param_dict.items():
                self.v[key] = np.zeros_like(param)


# Adagrad Optimizer(2011)
#
#   g  := dL/dθ
#   G  := G + g^2
#   θ := θ - η/sqrt(G + ε) * g
#
#   Hyperparameter
#       η: Initial learning rate. Typical value is 1e-2
#       ε: Safety guard from dividing zero. Typical value is 1e-8
class AdagradOptimizer:
    def __init__(self, eta=1e-2, epsilon=1e-8):
        self.eta = eta
        self.epsilon = epsilon
        self.G = None

    def optimize(self, param_dict, grad_dict):
        self.__check_variables(param_dict)
        for key in param_dict:
            g = grad_dict[key]
            self.G[key] += np.square(g)
            param_dict[key] -= self.eta / np.sqrt(self.G[key] + self.epsilon) * g

    def __check_variables(self, param_dict):
        if self.G is None:
            self.G = {}
            for key, param in param_dict.items():
                self.G[key] = np.zeros_like(param)


# RMSProp Optimizer(2012)
#
#   g  := dL/dθ
#   G  := ρ * G + (1 - ρ) * g^2
#   θ := θ - η/sqrt(G + ε) * g
#
#   Hyperparameter
#       η: Initial learning rate. Typical value is 1e-3
#       ρ: Decay rate of exponential averaging. Typical value is 0.9
#       ε: Safety guard from dividing zero. Typical value is unknowned. (Try 1e-6)
class RMSPropOptimizer:
    def __init__(self, eta=1e-3, rho=0.9, epsilon=1e-6):
        self.eta = eta
        self.rho = rho
        self.epsilon = epsilon
        self.G = None

    def optimize(self, param_dict, grad_dict):
        self.__check_variables(param_dict)
        for key in param_dict:
            g = grad_dict[key]
            self.G[key] = self.rho * self.G[key] + (1.0 - self.rho) * np.square(g)
            param_dict[key] -= self.eta / np.sqrt(self.G[key] + self.epsilon) * g

    def __check_variables(self, param_dict):
        if self.G is None:
            self.G = {}
            for key, param in param_dict.items():
                self.G[key] = np.zeros_like(param)


# Adadelta Optimizer(2012)
#
#   g  := dL/dθ
#   G  := ρ * G + (1 - ρ) * g^2
#   △ := sqrt((D + ε)/(G + ε)) * g
#   D  := ρ * D + (1 - ρ) * △^2
#   θ := θ - △
#
#   Hyperparameter
#       ρ: Decay rate of exponential averaging. Typical value is 0.9 ~ 0.99
#       ε: Safety guard from dividing zero. Typical value is 1e-6
class AdadeltaOptimizer:
    def __init__(self, rho=0.9, epsilon=1e-6):
        self.rho = rho
        self.epsilon = epsilon
        self.G = None
        self.T = None
        self.D = None

    def optimize(self, param_dict, grad_dict):
        self.__check_variables(param_dict)
        for key in param_dict:
            g = grad_dict[key]
            self.G[key] = self.rho * self.G[key] + (1.0 - self.rho) * np.square(g)
            self.T[key] = np.sqrt((self.D[key] + self.epsilon) / (self.G[key] + self.epsilon)) * g
            self.D[key] = self.rho * self.D[key] + (1.0 - self.rho) * np.square(self.T[key])
            param_dict[key] -= self.T[key]

    def __check_variables(self, param_dict):
        if self.G is None:
            self.G = {}
            self.T = {}
            self.D = {}
            for key, param in param_dict.items():
                self.G[key] = np.zeros_like(param)
                self.T[key] = np.zeros_like(param)
                self.D[key] = np.zeros_like(param)


# ADaptive Moment estimation Optimizer(2015)
#
#   g  := dL/dθ
#   m  := β1 * m + (1 - β1) * g
#   v  := β2 * v + (1 - β2) * g^2
#   M  := m / (1 - β1^t) (where t = 1, 2, ...)
#   V  := v / (1 - β2^t)
#   θ := θ - α * M / (sqrt(V) + ε)
#
#   Hyperparameter
#       α: Initial learning rate. Typical value is 1e-3
#       β1: Decay rate of momentum. Typical value is 0.9
#       β2: Decay rate of RMS of gradient. Typical value is 0.999
#       ε: Safety guard from dividing zero. Typical value is 1e-8
class AdamOptimizer:
    def __init__(self, alpha=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.beta1c = beta1
        self.beta2c = beta2
        self.m = None
        self.v = None

    def optimize(self, param_dict, grad_dict):
        self.__check_variables(param_dict)
        for key in param_dict:
            g = grad_dict[key]
            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * np.square(g)
            M = self.m[key] / (1.0 - self.beta1c)
            V = self.v[key] / (1.0 - self.beta2c)
            param_dict[key] -= self.alpha * M / (np.sqrt(V) + self.epsilon)
        self.beta1c *= self.beta1
        self.beta2c *= self.beta2

    def __check_variables(self, param_dict):
        if self.m is None:
            self.m = {}
            self.v = {}
            for key, param in param_dict.items():
                self.m[key] = np.zeros_like(param)
                self.v[key] = np.zeros_like(param)


# Continuous Coin Betting Optimizer(2017)
#
#   g  := - dL/dθ
#   L  := max(L, |g|)
#   G  := G + |g|
#   R  := max(R + (θ - θ0) * g, 0)
#   S  := S + g
#   θ := θ + S * (L + R) / (L * max(G + L, α * L))
#
#   Hyperparameter
#       α: Safety guard from having too big step size. Typical value is 100
#       ε: Safety guard from dividing by zero. Originally this doesn't exist.
class COCOBOptimizer:
    def __init__(self, alpha=100, epsilon=1e-6):
        self.alpha = alpha
        self.epsilon = epsilon
        self.L = None
        self.G = None
        self.R = None
        self.S = None
        self.t0 = None

    def optimize(self, param_dict, grad_dict):
        self.__check_variables(param_dict)
        for key in param_dict:
            g = -grad_dict[key]
            self.L[key] = np.maximum(self.L[key], np.abs(g))
            self.G[key] += np.abs(g)
            self.R[key] = np.maximum(self.R[key] + (param_dict[key] - self.t0[key]) * g, 0)
            self.S[key] += g
            L = self.L[key]
            R = self.R[key]
            param_dict[key] += self.S[key] * (L + R) / (L * np.maximum(self.G[key] + L, self.alpha * L) + self.epsilon)

    def __check_variables(self, param_dict):
        if self.L is None:
            self.L = {}
            self.G = {}
            self.R = {}
            self.S = {}
            self.t0 = {}
            for key, param in param_dict.items():
                self.L[key] = np.zeros_like(param)
                self.G[key] = np.zeros_like(param)
                self.R[key] = np.zeros_like(param)
                self.S[key] = np.zeros_like(param)
                self.t0[key] = np.copy(param)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class SimpleNetwork:
    def __init__(self):
        self.param_dict = {}
        self.grad_dict = {}
        self.__create_param('W', np.random.normal(loc=0.0, scale=1e-2, size=1))
        self.__create_param('b', np.zeros(1))

    def __create_param(self, name, init_value):
        self.param_dict[name] = np.copy(init_value)
        self.grad_dict[name] = np.zeros_like(init_value)

    def estimate(self, x_data):
        # s := x * W + b
        # y := sigmoid(s)
        # Note that this is 0D problem. If you use 1D, you have to change * np.dot()
        return sigmoid(x_data * self.param_dict['W'] + self.param_dict['b'])

    def train(self, x_data, y_data):
        # Forward
        # s := x_data * W + b
        # y_hat := sigmoid(s)
        y_hat = self.estimate(x_data)

        # loss := sum of every elements of 0.5 * (y_hat - y_data)^2
        # You can use average instead, if you have proper reasons.
        # But if you use average, you MUST divide dL/dy by batch size.
        diff = y_hat - y_data
        loss = 0.5 * np.sum(np.square(diff), axis=0)
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
        self.grad_dict['W'] = np.sum(x_data * dLds, axis=0)
        self.grad_dict['b'] = np.sum(dLds, axis=0)

        print('loss = ' + str(loss))


# Sometimes I miss you, "int main(int argc, char** argv)"
if __name__ == '__main__':
    # Prepare data.
    x_data = np.array([-1.0, 0.0, 1.0])
    y_data = np.array([0.0, 0.5, 1.0])

    # Prepare structure.
    network = SimpleNetwork()
    #optimizer = SGDOptimizer(eta=1e-3)
    #optimizer = MomentumOptimizer(eta=1e-3, alpha=0.5)
    #optimizer = AdagradOptimizer(eta=1e-2)
    #optimizer = RMSPropOptimizer(eta=1e-2, rho=0.9)
    #optimizer = AdadeltaOptimizer(rho=0.9)
    #optimizer = AdamOptimizer(alpha=1e-3, beta1=0.9, beta2=0.999)
    optimizer = COCOBOptimizer(alpha=100)

    # Train!
    for i in range(0, 1024):
        network.train(x_data, y_data)
        optimizer.optimize(network.param_dict, network.grad_dict)
