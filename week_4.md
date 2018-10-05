# 4주차: 오차역전파법

## 목차

1. 오차역전파법
    1. with 수식
    2. with 계산 그래프
2. 계산 그래프
    1. 계산 그래프로 문제 풀기
    2. 국소적 계산
    3. 역전파
3. 오차역전파법 구현 준비
    1. ReLU
    2. Sigmoid
    3. Affine
    4. Softmax-with-Loss
4. 오차역전파법 구현하기

## 오차역전파법

우리가 지금까지 봐 온 신경망에서는 가중치를 업데이트할 때 **수치 미분법** 을 사용해서 기울기를 구했다. 수치 미분법은 로직도 간단하고 이해하기 쉽지만, 계산하는 데 오래 걸린다는 단점이 있다. 신경망의 학습 효율은 계산을 수행하는 데 걸리는 시간과 직결되어 있기 때문에 시간이 많이 걸리는 수치 미분법은 효율에 치명적이다. 그래서 더 효율적으로 (어떤 의미에서는 더 정확하게) 기울기를 구하기 위해 **오차역전파법(*Backward-propagation of Error*, 줄여서 *Backpropagation*)** 이라는 기법이 고안되었다. 오차역전파법을 표현하는 방법은 크게 수식과 **계산 그래프** 두 가지가 있다. 두 표현 방법 모두 한번 구경해 보자.

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

먼저 위 그림에 사용된 기법인 **계산 그래프(*Computational Graph*)** 에 대해 알아보자. 계산 그래프는 복수의 **노드(*Node*)** 와 **에지(*Edge*)** 로 구성돼 있다. 계산 그래프와 친해지기 위해 먼저 간단한 수식을 계산 그래프로 표현해 보자.

### 계산 그래프로 문제 풀기

> 랭호가 한 권에 6600원짜리 라이트노벨 2권을 골라서 달러화로 결제했다. 그 때의 원-달러 환율은 1100원이고, 배송비 5달러가 더 부과됐다. 랭호는 몇 달러를 지불했을까?

암산을 할 수 있을 정도로 쉬운 문제이지만, 이번에는 계산 그래프를 이용해서 풀어 보자.

![sample computational graph question](/img/week_4_2.png)

위 그림과 같이 계산 그래프의 노드에는 **연산** 이 들어가고, 에지는 **결과값** 과 **결과값이 흐르는 곳** 을 지정해 준다. 여기서 화살표를 따라가며 계산해서 값을 구하는 것을 **순전파(*Forward-propagation*)** 라고 한다. 반대로, 화살표의 반대 방향으로 따라가며 계산을 하는 것을 **역전파(*Backward-propagation*)** 라고 한다.

### 국소적 계산

계산 그래프의 가장 큰 특징은 한 노드가 자신과 직접 관계된 부분만 가지고서도 결과를 출력할 수 있다는 것이다. 위 문제에 내용을 추가해 보자.

> 랭호가 라이트노벨을 사는 김에 친구들의 물건도 같이 사주기로 했다. 사야는 나가토로 동인지를, 계피는 타카기 브로마이드를, 리지는 적혈구 모자를 샀는데, 세 물건의 가격은 모두 합해서 33달러였다. 배송비는 똑같이 5달러이다. 랭호는 몇 달러를 지불했을까?

위 문제를 계산 그래프로 그리면 다음과 같다.

![local computation with computational graph](/img/week_4_3.png)

여기서 유의할 점은 친구들의 물건값과 라이트노벨값을 더하는 노드는 친구들이 정확히 무엇을 샀는지 알 필요가 없다는 점이다. 덧셈 노드는 그저 흘러들어온 값에만 신경을 쓰면 된다.

### 역전파

계산 그래프에서 역전파를 할 때는 주어진 노드를 역산하는 것이 아니라 국소적 미분을 계산해서 흘려넣어 준다. 즉, 이전 노드에서 흘러들어온 기울기에 자신의 기울기를 곱하여 다시 흘려보낸다.

![backwards-propagation and partial derivative](/img/week_4_4.png)

위 역전파 과정은 **연쇄법칙(*Chain Rule*)** 을 이용하여 해석학적 미분을 쉽게 구할 수 있게 해 준다. 예를 들어, 위 문제에서 원-달러 환율을 향해 전체 값으로부터 역전파를 해 나가면 환율이 바뀔 때 전체 가격은 얼마나 바뀌는지 쉽게 구할 수 있다.

## 오차역전파법 구현 준비

이제 계산 그래프를 배웠으니, 나중에 오차역전파법에 쓸 계층 클래스를 미리 준비해 두자. 노드 클래스는 기본적으로 다음과 같은 구조를 가진다. (덕 타이핑 덕분에 굳이 노드 클래스를 만들어 상속할 필요는 없다)

| 멤버 함수 | 역할 |
| ----- | ----- |
| forward | 순전파 계산을 수행한다. 필요에 따라 매개변수의 수는 달라질 수 있다. |
| backward | 역전파 계산을 수행한다. 반환값은 매개변수에 따라 다르다. |

그러면 바로 신경망에 사용되는 연산들을 계산 그래프 노드로 구현해 보자. 복잡한 수식을 구현하기 전에 조금 더 노드 구현에 익숙해지고 싶다면 [부록 페이지](/week_4_addendum.md)에 위에서 우려먹은 문제를 파이썬으로 구현해 두었다. 아래에서는 파이썬 구현만 보이고 유도하는 과정을 보이지는 않았다. [부록 페이지](/week_4_addendum.md)에 유도하는 과정도 적어 두었다.

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
        out = x.copy()       # x를 out 변수에 member-wise copy해 준 뒤
        out[self.mask] = 0   # 방금 만든 마스크에 해당하는 원소를 전부 0으로 만들고
        return out           # 그 값을 반환한다.

    def backward(self, dout):
        dout[self.mask] = 0 # 아까 만들어둔 마스크를 이용해서 필요한 부분을 버리고
        dx = dout           # 새 변수에 옮겨준 뒤
        return dx           # 반환한다.
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
        self.out = out             # 나중에 써먹어야 하니까 저장해 둔 뒤
        return out                 # 결과값을 반환한다.
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out # 아까 저장한 결과값으로 미분을 계산하고
        return dx                               # 반환한다.
```

### Affine

**어파인 변환(*Affine Transformation*)** 은 기하학에서 쓰이는 용어이다. 신경망의 순전파때 쓰이는 변환이다. 어파인 변환은 선형 변환보다 포괄적인 개념인데, 물체의 평행이동까지 설명할 수 있는 변환이다. 즉, 어파인 공간 X, Y에 대해 다음과 같은 어파인 변환이 있을 때,

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
        self.x = x                       # 미분할 때 쓰니까 입력값도 저장해 두고
        out = np.dot(x, self.W) + self.b # 넘파이가 싫어하지 않게 잘 곱한 뒤 편향을 더해주고
        return out                       # 반환한다.

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)      # 넘파이가 싫어하지 않게 순서를 맞춰 전치행렬을 곱해준 뒤
        self.dW = np.dot(self.x.T, dout) # 학습할 때 써야하니까 W에 대한 미분값도 저장해주고
        self.db = np.sum(dout, axis=0)   # 같은 이유로 편향에 대한 미분도 저장해주고
        return dx                        # X에 대한 미분값을 흘려보내준다.
```

### Softmax-with-Loss

신경망은 먼저 **데이터를 학습** 하여 만들어진 가중치를 이용하여 **추론** 하는 것이 주된 작업이다. 학습할 때에는 정답과 얼마나 다른지 계산한 뒤 이를 바탕으로 가중치를 변경해야 하기 때문에 마지막에 손실 함수를 사용해 줘야 하지만, 추론할 때에는 애초에 정답이 주어져 있지 않으므로 손실 함수를 넣는 것이 불가능하다. 비슷하게, 학습할 때에는 RAW 결과값을 손실 함수를 쓸 수 있는 확률값으로 바꿔주는 함수가 필요하지만, 추론할 때에는 어차피 제일 높은 값만 가져오니까 확률값으로 바꾸는 의미가 없다. 그러니까, 아예 값을 확률으로 바꿔 주는 **Softmax** 함수와 손실 함수인 **Cross Entropy Error** 함수를 하나로 합쳐 버리자. 함수값은 다음과 같다.

![softmax and cee equation][exp9]

이 두 식을 합성해 버리면 우리가 원하는 Softmax-with-Loss 함수가 튀어나온다. 책에서도 상당히 귀찮았는지 함수값은 안 써놓고 미분 계산하는 부분도 뒤쪽 부록으로 빼 두었다. 순전파 수식은 참 못생겼지만 미분을 하면 예뻐진다. 이 함수의 결과값 L의 Softmax 함수의 입력값 a(k)에 대한 미분을 구하면 다음과 같다.

![softmax-with-loss partial derivative][exp10]

여기에서 y(k)는 Softmax 함수의 순전파 출력값이고, t(k)는 정답 레이블의 값이다.

이 함수에 대한 또 다른 사담은 [부록 페이지](/week_4_addendum.md)에 더 있다. 이제 함숫값도 구했겠다, 파이썬으로 구현해 보자.

```python
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        self.t = t                                      # 정답 레이블을 복사해 두고
        self.y = softmax(x)                             # 소프트맥스를 계산해서 저장해 둔 뒤
        self.loss = cross_entropy_error(self.y, self.t) # 손실을 구하고
        return self.loss                                # 반환한다.
    
    def backward(self, dout=1):             # 이 계층이 마지막 계층인 경우가 많으므로 dout에 디폴트 파라미터를 준다.
        batch_size = self.t.shape[0]        # 배치의 크기를 저장하고
        dx = (self.y - self.t) / batch_size # 미분을 구한 뒤 정답 레이블의 크기로 나눠 배치 하나의 평균 기울기를 구한 뒤
        return dx                           # 반환한다.
```

## 오차역전파법 구현하기

이제 오차역전파법을 위한 준비물을 모두 만들었으니 구현해 보자. 2개의 은닉층을 가지는 신경망을 나타내는 클래스를 만들고 `TwoLayerNet`이라고 이름을 짓자. 먼저 어떤 인스턴스 변수가 있는지 보자.

| 인스턴스 변수 | 설명 |
| ----- | ----- |
| `params` | 신경망의 매개변수를 저장하는 딕셔너리 |
| `layers` | 신경망의 계층을 저장하는 순서가 있는 딕셔너리. 계층을 순서대로 보관해서 순전파나 역전파를 수행하기 편하다. |
| `lastLayer` | 신경망의 마지막 계층. 손실 함수가 있는 계층을 넣어 따로 보관해서 추론할 때 쓸데없는 연산을 줄인다. |

그 다음은 메서드 목록이다.

| 메서드 시그니처 | 설명 |
| ----- | ----- |
| `predict(self, x)` | 추론을 수행한다. 마지막 계층을 지나지 않고 결과를 뱉는다. |
| `loss(self, x, t)` | 손실 함수의 값을 구한다. t는 정답 레이블이다. |
| `accuracy(self, x, t)` | 정확도를 구한다. |
| `numerical_gradient(self, x, t)` | 수치적 미분을 구한다. 해석적 미분을 검증할때만 쓰고, 실제 학습에는 사용하지 않는다. (느려서) |
| `gradient(self, x, t)` | 오차역전파법을 이용해서 해석적 미분을 구한다. |

이제 실제로 구현해보자.

```python
class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
```

이제 이 신경망을 사용해서 학습을 진행하려면 `gradient` 함수를 이용해서 기울기를 구한 뒤, `TwoLayerNet.params`를 빼면 된다. 신경망 클래스에 `learn` 함수를 넣어도 되지 않았을까 하는 생각도 든다.

[exp1]: https://latex.codecogs.com/gif.latex?%5Cfrac%20%7B%5Cpartial%20E%7D%20%7B%5Cpartial%20w_%7Bij%7D%7D%20%3D%20%5Cfrac%20%7B%5Cpartial%20E%7D%20%7B%5Cpartial%20o_i%7D%20%5Ccdot%20%5Cfrac%20%7B%5Cpartial%20o_i%7D%20%7B%5Cpartial%20%5Ctextup%20%7Bnet%7D%7D%20%5Ccdot%20%5Cfrac%20%7B%5Cpartial%20%5Ctextup%20%7Bnet%7D%7D%20%7B%5Cpartial%20w_%7Bij%7D%7D

[exp2]: https://latex.codecogs.com/gif.latex?%5Ctextup%20%7BReLU%7D%20%28x%29%20%3D%20%5Cleft%20%5C%7B%20%5Cbegin%20%7Bmatrix%7D%20x%20%5Cleft.%5Cright.%20%28x%20%3E%200%29%20%5C%5C%200%20%5Cleft.%5Cright.%20%28x%20%5Cleq%200%29%20%5Cend%20%7Bmatrix%7D

[exp3]: https://latex.codecogs.com/gif.latex?%5Ctextup%20%7BSigmoid%7D%20%28x%29%20%3D%20%5Cfrac%20%7B1%7D%20%7B1%20&plus;%20e%5E%7B-x%7D%7D

[exp4]: https://latex.codecogs.com/gif.latex?%5Cfrac%20%7B%5Cpartial%20%5Ctextup%20%7BSigmoid%7D%7D%20%7B%5Cpartial%20x%7D%20%3D%20%5Ctextup%20%7BSigmoid%7D%281%20-%20%5Ctextup%20%7BSigmoid%7D%29

[exp5]: https://latex.codecogs.com/gif.latex?f%3A%20X%20%5Crightarrow%20Y

[exp6]: https://latex.codecogs.com/gif.latex?x%20%5Cmapsto%20Mx%20&plus;%20b

[exp7]: https://latex.codecogs.com/gif.latex?%5Ctextbf%20%7BAffine%7D%20%3D%20%5Cboldsymbol%20%7BX%7D%20%5Ccdot%20%5Cboldsymbol%20%7BW%7D%20&plus;%20%5Cboldsymbol%20%7BB%7D

[exp8]: https://latex.codecogs.com/gif.latex?%5Cfrac%20%7B%5Cpartial%20%5Ctextbf%20%7BAffine%7D%7D%20%7B%5Cpartial%20%5Cboldsymbol%20X%7D%20%3D%20%5Cboldsymbol%20W%5ET

[exp9]: https://latex.codecogs.com/gif.latex?%5Ctextup%20%7BSoftmax%7D%20%28a%29%20%3D%20%5Cfrac%20%7Be%5E%7Ba_k%7D%7D%20%7B%5Csum_%7Bi%20%3D%201%7D%5E%7Bn%7D%20e%5E%7Ba_i%7D%7D%20%5C%5C%20%5C%5C%20%5Ctextup%20%7BCEE%7D%20%28x%2C%20t%29%20%3D%20-%20%5Csum_%7Bk%7D%20t_k%20%5Clog%20x_k

[exp10]: https://latex.codecogs.com/gif.latex?%5Cfrac%20%7B%5Cpartial%20L%7D%20%7B%5Cpartial%20a_k%7D%20%3D%20y_k%20-%20t_k
