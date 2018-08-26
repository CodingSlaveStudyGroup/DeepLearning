import numpy as np


# 평범한 단층 퍼셉트론입니다. 다층 퍼셉트론은 학습시킬 수 없습니다.
# 아주 단순한 모델로, input_size개의 입력을 받아서 1개의 출력으로 뱉습니다.
class Perceptron:
    # 입력 차원이 input_size인 퍼셉트론을 생성합니다.
    def __init__(self, input_size):
        # 가중치는 N(0, 1^2)로, 편향은 0으로 초기화합니다.
        # 편향은 스칼라 값이지만 굳이 numpy로 초기화해 준 이유는
        # 계산에서 서로 호환되게 하기 위해서입니다.
        self.w = np.random.standard_normal(input_size)
        self.b = np.zeros(1)

    # 입력을 x로 두고 퍼셉트론의 출력을 계산합니다.
    def compute(self, x):
        # 책에서는 np.sum(w * x) + b를 사용하고 있는데
        # 이렇게 하면 x의 차원만큼의 임시 변수가 생기기 때문에
        # dot를 써줘서 바로 스칼라 numpy array로 만들어주는 것이 좋습니다.
        pre_active = self.w.dot(x) + self.b

        # 파이썬의 정신나간 3항 연산자입니다.
        # C의 "pre_active > 0.5 ? 1.0 : 0.0"과 같습니다.
        return 1.0 if pre_active > 0.5 else 0.0

    # 입력을 x로 두고 출력이 y가 되도록 델타 룰을 사용해 학습하고 오차를 반환합니다.
    # alpha는 학습률로 너무 낮으면 학습이 느리고, 너무 높으면 고장납니다.
    def learn(self, x, y, alpha):
        y_estimated = self.compute(x)
        diff = y_estimated - y
        self.w -= alpha * diff * x
        self.b -= alpha * diff
        return np.abs(diff)


# 2개짜리 AND 게이트를 학습시켜봅시다.
perceptron = Perceptron(2)
train_set = [(np.array([0, 0]), np.array([0])),
             (np.array([0, 1]), np.array([0])),
             (np.array([1, 0]), np.array([0])),
             (np.array([1, 1]), np.array([1]))]


# epoch란 주어진 학습 데이터 집합을 몇 바퀴 돌았는지를 세는 단위입니다.
# 비슷한 표현으로 iteration이 있으나, 정확히 같지는 않습니다.
# iteration은 뒤에서 배울 batch를 몇 번 수행하였는지 세는 단위입니다.
epoch = 0
err = 1e+8
while err > 1e-16:
    epoch += 1
    err = 0.0
    for train_pair in train_set:
        err += perceptron.learn(train_pair[0], train_pair[1], 0.1)
    print(err)


# 학습이 제대로 되었다면 가중치와 편향을 구경합시다.
print('learning end with epoch ' + str(epoch))
print('w = ')
print(perceptron.w)
print('b = ')
print(perceptron.b)