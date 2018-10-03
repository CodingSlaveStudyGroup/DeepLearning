# 4주차: 오차역전파법

## 목차

## 오차역전파법

우리가 지금까지 봐 온 신경망에서는 가중치를 업데이트할 때 **수치 미분법**을 사용해서 기울기를 구했다. 수치 미분법은 로직도 간단하고 이해하기 쉽지만, 계산하는 데 오래 걸린다는 단점이 있다. 신경망의 학습 효율은 계산을 수행하는 데 걸리는 시간과 직결되어 있기 때문에 시간이 많이 걸리는 수치 미분법은 효율에 치명적이다. 그래서 더 효율적으로 (어떤 의미에서는 더 정확하게) 기울기를 구하기 위해 **오차역전파법(*Backward-propagation of Error*, 줄여서 *Backpropagation*)**이라는 기법이 고안되었다. 오차역전파법을 표현하는 방법은 크게 수식과 **계산 그래프** 두 가지가 있다. 두 표현 방법 모두 한번 구경해 보자.

### 오차역전파법 (feat. 수식)

보통 위키피디아 등지에서는 오차역전파법을 수식으로 표현한다. 먼저 오차역전파법을 표현하는 수식을 보자.

한 신경망 N에 대하여 목적지 가중치를 w(ij), 은닉층을 통틀어 `net`, i번째 활성화 함수를 o(i), 그리고 손실 함수를 E라 할 때, E의 w(ij)에 대한 기울기는 다음과 같이 나타낼 수 있다.

![backpropagation using expressions][exp1]

이게 무엇을 의미하는지 직관적으로 이해하기 어렵다.

### 오차역전파법 (feat. 계산 그래프)

위와 같은 신경망의 기울기를 구하되, 이번에는 계산 그래프로 표현해보기로 하자. 아래 그림의 기호가 의미하는 바는 위와 같다.

![backpropatation using computational graph](/img/week_4_1.png)

표현의 편의를 위해 여기서는 w(21)이라는 특정한 가중치에 대한 E의 기울기를 구해 보았다. 제일 왼쪽의 보라색 선이 우리가 구하고자 하는 미분값이고, 파란색 화살표는 그 미분값을 구하는 **역전파** 과정이다. 결과적으로 나오는 수식은 위와 같지만, 어떤 방식으로 결과가 도출되는지 더 간결하게 알 수 있다.

## 계산 그래프

먼저 위 그림에 사용된 기법인 **계산 그래프(*Computational Graph*)**에 대해 알아보자. 계산 그래프는 복수의 **노드(*Node*)**와 **에지(*Edge*)**로 구성돼 있다. 계산 그래프와 친해지기 위해 먼저 간단한 수식을 계산 그래프로 표현해 보자.

### 계산 그래프로 문제 풀기

> 랭호가 한 권에 6600원짜리 라이트노벨 2권을 골라서 달러화로 결제했다. 그 때의 원-달러 환율은 1100원이고, 배송비 5달러가 더 부과됐다. 랭호는 몇 달러를 지불했을까?

암산을 할 수 있을 정도로 쉬운 문제이지만, 이번에는 계산 그래프를 이용해서 풀어 보자.

![sample computational graph question](/img/week_4_2.png)

위 그림과 같이 계산 그래프의 노드에는 **연산**이 들어가고, 에지는 **결과값**과 **결과값이 흐르는 곳**을 지정해 준다. 여기서 화살표를 따라가며 계산해서 값을 구하는 것을 **순전파(*Forward-propagation*)**라고 한다. 반대로, 화살표의 반대 방향으로 따라가며 계산을 하는 것을 **역전파(*Backward-propagation*)**라고 한다.

### 국소적 계산

계산 그래프의 가장 큰 특징은 한 노드가 자신과 직접 관계된 부분만 가지고서도 결과를 출력할 수 있다는 것이다. 위 문제에 내용을 추가해 보자.

> 랭호가 라이트노벨을 사는 김에 친구들의 물건도 같이 사주기로 했다. 사야는 나가토로 동인지를, 계피는 타카기 브로마이드를, 리지는 적혈구 모자를 샀는데, 세 물건의 가격은 모두 합해서 33달러였다. 배송비는 똑같이 5달러이다. 랭호는 몇 달러를 지불했을까?

위 문제를 계산 그래프로 그리면 다음과 같다.

![local computation with computational graph](/img/week_4_3.png)

여기서 유의할 점은 친구들의 물건값과 라이트노벨값을 더하는 노드는 친구들이 정확히 무엇을 샀는지 알 필요가 없다는 점이다. 덧셈 노드는 그저 흘러들어온 값에만 신경을 쓰면 된다.

### 역전파

계산 그래프에서 역전파를 할 때는 주어진 노드를 역산하는 것이 아니라 국소적 미분을 계산해서 흘려넣어 준다. 즉, 이전 노드에서 흘러들어온 기울기에 자신의 기울기를 곱하여 다시 흘려보낸다.

![backwards-propagation and partial derivative](/img/week_4_4.png)

위 역전파 과정은 **연쇄법칙(*Chain Rule*)**을 이용하여 해석학적 미분을 쉽게 구할 수 있게 해 준다. 예를 들어, 위 문제에서 원-달러 환율을 향해 전체 값으로부터 역전파를 해 나가면 환율이 바뀔 때 전체 가격은 얼마나 바뀌는지 쉽게 구할 수 있다.

## 오차역전파법 구현하기

이제 계산 그래프를 배웠으니, 실제로 파이썬으로 구현해서 적용해보자. 노드 클래스는 기본적으로 다음과 같은 구조를 가진다. (덕 타이핑 덕분에 굳이 노드 클래스를 만들어 상속할 필요는 없지만...)

| 멤버 함수 | 역할 |
|-----|-----|
| forward | 순전파 계산을 수행한다. 필요에 따라 매개변수의 수는 달라질 수 있다. |
| backward | 역전파 계산을 수행한다. 반환값은 매개변수에 따라 다르다. |

그러면 바로 신경망에 사용되는 연산들을 계산 그래프 노드로 구현해 보자. 조금 더 노드 구현에 익숙해지고 싶다면 [부록 페이지](/week_4_addendum.md)에 위에서 우려먹은 문제를 파이썬으로 구현해 두었다. 아래에서는 파이썬 구현만 보이고 유도하는 과정을 보이지는 않았다. 위 [부록 페이지](/week_4_addendum.md)에 유도하는 과정도 적어 두었다.

### ReLU

ReLU는 최근 들어 많이 사용하는 활성화 함수이다. 함수의 정의가 간단해서 계산도 빠르기에 자주 쓰인다.

![ReLU function definition][exp2]

입력이 0보다 크기만 하면 그대로 뱉어내는 함수이기 때문에 입력값이 0보다 컸다면 미분값도 흘러들어온 그대로 뱉어 준다. 그럼 바로 파이썬으로 구현해 보자.

```python
class Relu:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x <= 0) # x의 원소 중 0보다 크면 False, 아니면 True로 마스크를 만들어 주고
        out = x.copy() # x를 out 변수에 member-wise copy해 준 뒤
        out[self.mask] = 0 # 방금 만든 마스크에 해당하는 원소를 전부 0으로 만들고
        return out # 그 값을 반환한다.
    def backward(self, dout):
        dout[self.mask] = 0 # 아까 만들어둔 마스크를 이용해서 필요한 부분을 버리고
        dx = dout # 새 변수에 옮겨준 뒤
        return dx # 반환한다.
```

### Sigmoid

로지스틱 함수라고도 불리는 시그모이드 함수는 활성화 함수계의 터줏대감이다. 정의는 다음과 같다.

![sigmoid function definition][exp3]

복잡해 보이지만 미분값은 간단하다.

![sigmoid function partial derivative][exp4]

이렇게 결과값만 가지고도 미분을 계산해 줄 수 있다. 파이썬으로도 구현해 보자.

```python
class Sigmoid:
    def __init__(self):
        self.out = None
    def forward(self, x):
        out = 1 / (1 + np.exp(-x)) # 결과값을 먼저 계산해주자
        self.out = out # 나중에 써먹어야 하니까 저장해 둔 뒤
        return out # 결과값을 반환한다.
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out # 아까 저장한 결과값으로 미분을 계산하고
        return dx # 반환한다.
```

### Affine

**어파인 변환(*Affine Transformation*)**은 기하학에서 쓰이는 용어이다. 신경망의 순전파때 쓰이는 변환이다. 어파인 변환은 선형 변환보다 포괄적인 개념인데, 물체의 평행이동까지 설명할 수 있는 변환이다. 즉, 어파인 공간 X, Y에 대해 다음과 같은 어파인 변환이 있을 때,

![affine transformation f][exp5]

공간 X 안의 벡터 x, 공간 Y 안의 벡터 b, 선형 변환 M에 대하여 다음이 성립한다.

![affine transformation mapping][exp6]

여기서는 행렬 두 개를 곱한 후 다른 행렬을 더하는 것을 어파인 변환이라 부른다. 즉, 다음 식과 같다.

![affine transformation in this book][exp7]

아무래도 행렬을 가지고 계산을 하는 연산이라 미분 구하는 게 ~~귀찮다~~ 복잡하다. 과정을 보고 싶다면 [부록 페이지](/week_4_addendum.md)에 노가다를 해 뒀다. 하지만 미분 결과는 예쁘게 나온다.

![affine transformation partial derivative][exp8]

비슷하게, **W**로 미분하면 결과값이 **X**의 전치행렬으로 바뀐다. 다만 행렬곱이기 때문에 곱하는 위치에 주의하자. 순서를 바꿔서 곱하면 행렬의 모양이 안 맞아서 넘파이가 싫어한다. 그러면 파이썬으로 구현해 보자.

```python
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    def forward(self, x)
        self.x = x # 미분할 때 쓰니까 입력값도 저장해 두고
        out = np.dot(x, self.W) + self.b # 넘파이가 싫어하지 않게 순서를 잘 맞춰서 곱해준 뒤 편향을 더해주고
        return out # 반환한다.
    def backward(self, dout):
        dx = np.dot(dout, self.W.T) # 넘파이가 싫어하지 않게 순서를 맞춰 전치행렬을 곱해준 뒤
        self.dW = np.dot(self.x.T, dout) # 학습할 때 써야하니까 W에 대한 미분값도 저장해주고
        self.db = np.sum(dout, axis=0) # 같은 이유로 편향에 대한 미분도 저장해주고
        return dx # X에 대한 미분값을 흘려보내준다.
```


[exp1]: https://latex.codecogs.com/gif.latex?%5Cfrac%20%7B%5Cpartial%20E%7D%20%7B%5Cpartial%20w_%7Bij%7D%7D%20%3D%20%5Cfrac%20%7B%5Cpartial%20E%7D%20%7B%5Cpartial%20o_i%7D%20%5Ccdot%20%5Cfrac%20%7B%5Cpartial%20o_i%7D%20%7B%5Cpartial%20%5Ctextup%20%7Bnet%7D%7D%20%5Ccdot%20%5Cfrac%20%7B%5Cpartial%20%5Ctextup%20%7Bnet%7D%7D%20%7B%5Cpartial%20w_%7Bij%7D%7D

[exp2]: https://latex.codecogs.com/gif.latex?%5Ctextup%20%7BReLU%7D%20%28x%29%20%3D%20%5Cleft%20%5C%7B%20%5Cbegin%20%7Bmatrix%7D%20x%20%5Cleft.%5Cright.%20%28x%20%3E%200%29%20%5C%5C%200%20%5Cleft.%5Cright.%20%28x%20%5Cleq%200%29%20%5Cend%20%7Bmatrix%7D

[exp3]: https://latex.codecogs.com/gif.latex?%5Ctextup%20%7BSigmoid%7D%20%28x%29%20%3D%20%5Cfrac%20%7B1%7D%20%7B1%20&plus;%20e%5E%7B-x%7D%7D

[exp4]: https://latex.codecogs.com/gif.latex?%5Cfrac%20%7B%5Cpartial%20%5Ctextup%20%7BSigmoid%7D%7D%20%7B%5Cpartial%20x%7D%20%3D%20%5Ctextup%20%7BSigmoid%7D%281%20-%20%5Ctextup%20%7BSigmoid%7D%29

[exp5]: https://latex.codecogs.com/gif.latex?f%3A%20X%20%5Crightarrow%20Y

[exp6]: https://latex.codecogs.com/gif.latex?x%20%5Cmapsto%20Mx%20&plus;%20b

[exp7]: https://latex.codecogs.com/gif.latex?%5Ctextbf%20%7BAffine%7D%20%3D%20%5Cboldsymbol%20%7BX%7D%20%5Ccdot%20%5Cboldsymbol%20%7BW%7D%20&plus;%20%5Cboldsymbol%20%7BB%7D

[exp8]: https://latex.codecogs.com/gif.latex?%5Cfrac%20%7B%5Cpartial%20%5Ctextbf%20%7BAffine%7D%7D%20%7B%5Cpartial%20%5Cboldsymbol%20X%7D%20%3D%20%5Cboldsymbol%20W%5ET
