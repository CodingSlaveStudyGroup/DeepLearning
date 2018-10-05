# 오차역전파법 부록

여기에는 그냥 넘겨도 상관 없는 내용들을 담아 두었다.

## 계산 그래프와 친해지기

본문에서 우려먹은 문제를 계산 그래프로 그려보자. 문제는 다음과 같다.

> 랭호가 한 권에 6600원짜리 라이트노벨 2권을 골라서 달러화로 결제했다. 그 때의 원-달러 환율은 1100원이고, 배송비 5달러가 더 부과됐다. 랭호는 몇 달러를 지불했을까?

그리고 이 문제를 표현한 계산 그래프는 다음과 같다.

![sample computational graph](/img/week_4_2.png)

여기에는 덧셈, 곱셈, 나눗셈 총 세 종류의 노드가 사용되었다. 그러면 이 노드를 파이썬으로 구현해 보자.

### 덧셈 노드

덧셈은 2개의 입력을 받아, 순전파 시에는 두 입력을 더하고, 역전파 시에는 미분값을 그대로 흘려보낸다.

```python
class Add:
    def __init__(self):
        pass
    
    def forward(self, x, y):
        return x + y # 더해서 그냥 흘려보낸다
    
    def backward(self, dout):
        return (dout, dout) # 이전 계층의 미분값을 그대로 흘려보낸다
```

### 곱셈 노드

곱셈은 2개의 입력을 받아, 순전파 시에는 두 입력을 더하고, 역전파 시에는 미분값에 입력을 *바꿔서* 곱한다.

```python
class Multiply:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x   # 미분할 때 필요하니까 저장해 둔다
        self.y = y   # 이것도
        return x * y # 그 다음 곱해서 흘려보낸다

    def backward(self, dout):
        dx = dout * self.y # 이전 미분값에 입력을 바꿔서 곱한다
        dy = dout * self.x # 이것도
        return (dx, dy)    # 그 다음에 흘려보낸다
```

### 나눗셈 노드

나눗셈은 2개의 입력 x, y를 받아, 순전파 시에는 x를 y로 나눠 반환하고, 역전파 시에는 각각 y로 나누고, -x를 곱한 뒤 y의 제곱으로 나눠 반환한다.

```python
class Divide:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x   # 미분할 때 필요하니까 저장해 둔다
        self.y = y   # 이것도
        return x / y # 그 다음 나눠 준다

    def backward(self, dout):
        dx = dout / self.y                    # x에 대한 미분을 구하려면 이전 미분값을 y로 나눈다
        dy = dout * (-self.x) / (self.y ** 2) # y에 대한 미분을 구하려면 이전 미분값에 (-x/(y^2))를 곱한다
        return (dx, dy)                       # 그리고 흘려 준다
```

### 그래프 구현

이제 노드도 준비되었으니 실제로 구현해보자.

```python
lnovel = 6600              # 라이트노벨 가격
lnovel_num = 2             # 산 라이트노벨 갯수
currency_conversion = 1100 # 원-달러 환율
shipping_fee = 5           # 배송비

lnovel_multiple_layer = Multiply() # 라이트노벨 2개의 가격을 계산
currency_convert_layer = Divide()  # 라이트노벨 가격을 달러화로 변환
shipping_fee_layer = Add()         # 배송비를 추가

# 순전파
lnovel_price = lnovel_multiple_layer.forward(lnovel, lnovel_num)
lnovel_price_dollar = currency_convert_layer.forward(lnovel_price, currency_conversion)
final_price = shipping_fee_layer.forward(lnovel_price_dollar, shipping_fee)

# 역전파
dfinal_price = 1
(dlnovel_price_dollar, dshipping_fee) = shipping_fee_layer.backward(dfinal_price)
(dlnovel_price, dcurrency_conversion) = currency_convert_layer.backward(dlnovel_price_dollar)
(dlnovel, dlnovel_num) = lnovel_multiple_layer.backward(dlnovel_price)
```

## ReLU 미분하기

ReLU 함수의 식은 다음과 같다.

![relu function definition][exp1]

이 함수를 미분하면 각각의 정의를 따로 미분해 주므로 다음과 같아진다.

![relu function partial derivative][exp2]

## Sigmoid 미분하기

시그모이드 함수의 식은 다음과 같다.

![sigmoid function definition][exp3]

여기서는 속미분을 사용해서 기울기를 구해 보자. 분모를 a라 두면,

![sigmoid function derivative part 1][exp4]

이다. 이번에는 exp(-x)를 b로 두고 계산해보자. 그러면,

![sigmoid function derivative part 2][exp5]

이다. 마지막으로 위 식을 끝까지 미분하면,

![sigmoid function derivative part 3][exp6]

이렇게 된다. 이 식을 조금만 더 정리해보자.

![sigmoid function derivative part 4][exp7]

이렇게 책에 나온것처럼 출력 값만 가지고 시그모이드 함수의 미분값을 구할 수 있다. 와!

## Affine 미분하기

> 행렬의 미분법은 [여기](http://cs231n.stanford.edu/vecDerivs.pdf)를 참고했다.

이 책에서 **어파인** 이라고 불리는 연산은 기본적으로 다음과 같다.

![affine operation][exp8]

먼저 **배치(*Batch*)** 를 고려하지 않은 일반적인 연산을 생각해보자. 이 때, **X**는 크기 `n`의 행벡터, **W**는 크기 `n×m`의 행렬, **B**는 크기 `m`의 행벡터, **Y**는 크기 `m`의 행벡터이다. 하지만 벡터를 넘파이에 집어넣으면 `(n, )` 형식으로 열벡터로 표현된다. 이 경우 넘파이에서 자동으로 행벡터로 바꿔서 계산해준다.

![variable definitions][exp9]

이제 미분을 해 보자. 벡터 미분의 정의에 따라, 벡터를 다른 벡터로 미분하면 각 원소에 대한 편미분을 모두 구해야 하기 때문에 행렬이 나온다.

![differentiating vector with vector][exp10]

그럼 여기에서 한번 계산을 해 볼 수 있을 것 같다. 전부 계산하는 건 귀찮으니까 `1 < i < n`을 만족하는 `i`와 `1 < j < m`을 만족하는 `j`를 임의로 골라서 x(i), y(j)에 대한 값을 구해보도록 하자. 편향 벡터를 더하는 부분은 스칼라 덧셈과 마찬가지로 미분해도 변함이 없기 때문에 생략하도록 하자. 그렇다면 먼저 **Y**의 각 원소는 행벡터 **X**와 행렬 **W**안의 특정한 열벡터를 스칼라곱 한 것이다. 즉,

![definition of y(j)][exp11]

을 만족한다. 그럼 이 식을 x(i)에 대해 편미분 해 보자. 하지만 위 식에서 x(i)가 출현하는 곳은 한 군데밖에 없기 때문에, 편미분하면 다음과 같다.

![differentiating y(j) with respect to x(i)][exp12]

이 식을 일반화시켜서 위 행렬에 꽂아넣으면 다음과 같은 행렬이 나온다.

![actual differentiation][exp13]

이렇게 해서 **W**의 전치행렬이 미분값이라는 점을 알아냈다. 역전파 때 넘파이가 화내지 않게 크기를 잘 맞춰서 넣어주도록 하자.

...하지만 책은 여기서 한 술 더 떠서 배치용 어파인 계층을 만든다. 배치용 어파인 계층의 특징은 입력이 행벡터가 아닌 행렬 모양이라는 것이다. 따라서, 행렬을 행렬으로 미분하는 모양새가 되는데, 원칙적으로 행렬을 행렬으로 미분하면 4차원 텐서가 나온다. (계산해보면 2차원 안에 포함시킬 수 있게 나오긴 한다.) 그래서 여기에서의 입력값은 행렬이 아니라 행벡터의 배열으로 보는 것이 더 이해하기 쉽다. 즉,

```python
for row_vector in X:
    result = row_vector.dot(W)
    Y.append(result)
```

정도로 해석하는 게 더 간단하다고 생각한다.

## Softmax-with-Loss 

이 함수는 책 부록에 미분 과정이 계산 그래프를 이용해 그려져 있어서 훑어보면 된다.

[exp1]: https://latex.codecogs.com/gif.latex?%5Ctextup%20%7BReLU%7D%20%28x%29%20%3D%20%5Cleft%20%5C%7B%20%5Cbegin%20%7Bmatrix%7D%20x%20%5Cleft.%5Cright.%20%28x%20%3E%200%29%20%5C%5C%200%20%5Cleft.%5Cright.%20%28x%20%5Cleq%200%29%20%5Cend%20%7Bmatrix%7D

[exp2]: https://latex.codecogs.com/gif.latex?%5Cfrac%20%7B%5Cpartial%20%5Ctextup%20%7BReLU%7D%7D%20%7B%5Cpartial%20x%7D%20%3D%20%5Cleft%20%5C%7B%20%5Cbegin%20%7Bmatrix%7D%20x%20%5Cleft.%20%5Cright.%20%28x%20%3E%200%29%20%5C%5C%200%20%5Cleft.%20%5Cright.%20%28x%20%5Cleq%200%29%20%5Cend%20%7Bmatrix%7D%20%5Cright.

[exp3]: https://latex.codecogs.com/gif.latex?%5Ctextup%20%7BSigmoid%7D%20%28x%29%20%3D%20%5Cfrac%20%7B1%7D%20%7B1%20&plus;%20e%5E%7B-x%7D%7D

[exp4]: https://latex.codecogs.com/gif.latex?%5Cfrac%20%7B%5Cpartial%20%5Ctextup%20%7BSigmoid%7D%7D%20%7B%5Cpartial%20x%7D%20%3D%20%5Cfrac%20%7B%5Cpartial%20%5Ctextup%20%7BSigmoid%7D%7D%20%7B%5Cpartial%20a%7D%20%5Ccdot%20%5Cfrac%20%7B%5Cpartial%20a%7D%20%7B%5Cpartial%20x%7D%20%3D%20-%20%5Cfrac%20%7B1%7D%20%7B%281%20&plus;%20e%5E%7B-x%7D%29%5E2%7D%20%5Ccdot%20%5Cfrac%20%7B%5Cpartial%7D%20%7B%5Cpartial%20x%7D%20%281%20&plus;%20e%5E%7B-x%7D%29

[exp5]: https://latex.codecogs.com/gif.latex?%5Cfrac%20%7B%5Cpartial%20%5Ctextup%20%7BSigmoid%7D%7D%20%7B%5Cpartial%20x%7D%20%3D%20%5Cfrac%20%7B%5Cpartial%20%5Ctextup%20%7BSigmoid%7D%7D%20%7B%5Cpartial%20a%7D%20%5Ccdot%20%5Cfrac%20%7B%5Cpartial%20a%7D%20%7B%5Cpartial%20b%7D%20%5Ccdot%20%5Cfrac%20%7B%5Cpartial%20b%7D%20%7B%5Cpartial%20x%7D%20%3D%20-%20%5Cfrac%20%7B1%7D%20%7B%281%20&plus;%20e%5E%7B-x%7D%29%5E2%7D%20%5Ccdot%201%20%5Ccdot%20%5Cfrac%20%7B%5Cpartial%7D%20%7B%5Cpartial%20x%7D%20%28e%5E%7B-x%7D%29

[exp6]: https://latex.codecogs.com/gif.latex?%5Cfrac%20%7B%5Cpartial%20%5Ctextup%20%7BSigmoid%7D%7D%20%7B%5Cpartial%20x%7D%20%3D%20%5Cfrac%20%7B%5Cpartial%20%5Ctextup%20%7BSigmoid%7D%7D%20%7B%5Cpartial%20a%7D%20%5Ccdot%20%5Cfrac%20%7B%5Cpartial%20a%7D%20%7B%5Cpartial%20b%7D%20%5Ccdot%20%5Cfrac%20%7B%5Cpartial%20b%7D%20%7B%5Cpartial%20x%7D%20%3D%20-%20%5Cfrac%20%7B1%7D%20%7B%281%20&plus;%20e%5E%7B-x%7D%29%5E2%7D%20%5Ccdot%201%20%5Ccdot%20%28-e%5E%7B-x%7D%29%20%5C%5C%20%5Ctherefore%20%5Cfrac%20%7B%5Cpartial%20%5Ctextup%20%7BSigmoid%7D%7D%20%7B%5Cpartial%20x%7D%20%3D%20%5Cfrac%20%7Be%5E%7B-x%7D%7D%20%7B%281%20&plus;%20e%5E%7B-x%7D%29%5E2%7D

[exp7]: https://latex.codecogs.com/gif.latex?%5Cfrac%20%7B%5Cpartial%20%5Ctextup%20%7BSigmoid%7D%7D%20%7B%5Cpartial%20x%7D%20%3D%20%5Cfrac%20%7Be%5E%7B-x%7D%7D%20%7B%281%20&plus;%20e%5E%7B-x%7D%29%5E2%7D%20%3D%20%5Cfrac%20%7B1%7D%20%7B1%20&plus;%20e%5E%7B-x%7D%7D%20%5Ccdot%20%5Cfrac%20%7Be%5E%7B-x%7D%7D%20%7B1%20&plus;%20e%5E%7B-x%7D%7D%20%5C%5C%20%5C%5C%20%3D%20%5Ctextup%20%7BSigmoid%7D%20%5Ccdot%20%281%20-%20%5Ctextup%20%7BSigmoid%7D%29%20%5C%5C%20%5Cbecause%20%5Ctextup%20%7BSigmoid%7D%20%3D%20%5Cfrac%20%7B1%7D%20%7B1%20&plus;%20e%5E%7B-x%7D%7D

[exp8]: https://latex.codecogs.com/gif.latex?%5Cboldsymbol%20Y%20%3D%20%5Cboldsymbol%20X%20%5Ccdot%20%5Cboldsymbol%20W%20&plus;%20%5Cboldsymbol%20B

[exp9]: https://latex.codecogs.com/gif.latex?%5Cboldsymbol%20X%20%3D%20%5Cbegin%20%7Bpmatrix%7D%20x_1%20%26%20%5Ccdots%20%26%20x_n%20%5Cend%20%7Bpmatrix%7D%20%5C%5C%20%5Cboldsymbol%20W%20%3D%20%5Cbegin%20%7Bpmatrix%7D%20w_%7B11%7D%20%26%20%5Ccdots%20%26%20w_%7B1m%7D%20%5C%5C%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%20%5C%5C%20w_%7Bn1%7D%20%26%20%5Ccdots%20%26%20w_%7Bnm%7D%20%5Cend%20%7Bpmatrix%7D%20%5C%5C%20%5Cboldsymbol%20B%20%3D%20%5Cbegin%20%7Bpmatrix%7D%20b_1%20%26%20%5Ccdots%20%26%20b_m%20%5Cend%20%7Bpmatrix%7D%20%5C%5C%20%5Cboldsymbol%20Y%20%3D%20%5Cbegin%20%7Bpmatrix%7D%20y_1%20%26%20%5Ccdots%20%26%20y_m%20%5Cend%20%7Bpmatrix%7D

[exp10]: https://latex.codecogs.com/gif.latex?%5Cfrac%20%7B%5Cpartial%20%5Cboldsymbol%20Y%7D%20%7B%5Cpartial%20%5Cboldsymbol%20X%7D%3D%20%5Cbegin%20%7Bpmatrix%7D%20%5Cfrac%20%7B%5Cpartial%20%5Cboldsymbol%20Y%7D%20%7B%5Cpartial%20x_1%7D%20%26%20%5Ccdots%20%26%20%5Cfrac%20%7B%5Cpartial%20%5Cboldsymbol%20Y%7D%20%7B%5Cpartial%20x_n%7D%20%5Cend%20%7Bpmatrix%7D%20%3D%20%5Cbegin%20%7Bpmatrix%7D%20%5Cfrac%20%7B%5Cpartial%20y_1%7D%20%7B%5Cpartial%20x_1%7D%20%26%20%5Ccdots%20%26%20%5Cfrac%20%7B%5Cpartial%20y_1%7D%20%7B%5Cpartial%20x_n%7D%20%5C%5C%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%20%5C%5C%20%5Cfrac%20%7B%5Cpartial%20y_m%7D%20%7B%5Cpartial%20x_1%7D%20%26%20%5Ccdots%20%26%20%5Cfrac%20%7B%5Cpartial%20y_m%7D%20%7B%5Cpartial%20x_n%7D%20%5Cend%20%7Bpmatrix%7D

[exp11]: https://latex.codecogs.com/gif.latex?y_j%20%3D%20%5Cboldsymbol%20X%20%5Ccdot%20%5Cboldsymbol%20W_%7B*j%7D%20%3D%20%5Csum_%7Bi%7D%20x_i%20%5Ccdot%20w_%7Bij%7D%20%5C%5C%20%3D%20x_1%20w_%7B1j%7D%20&plus;%20x_2%20w_%7B2j%7D%20&plus;%20%5Ccdots%20&plus;%20x_i%20w_%7Bij%7D%20&plus;%20%5Ccdots%20&plus;%20x_n%20w_%7Bnj%7D

[exp12]: https://latex.codecogs.com/gif.latex?%5Cfrac%20%7B%5Cpartial%20y_j%7D%20%7B%5Cpartial%20x_i%7D%20%3D%20w_%7Bij%7D

[exp13]: https://latex.codecogs.com/gif.latex?%5Cfrac%20%7B%5Cpartial%20%5Cboldsymbol%20Y%7D%20%7B%5Cpartial%20%5Cboldsymbol%20X%7D%20%3D%20%5Cbegin%20%7Bpmatrix%7D%20%5Cfrac%20%7B%5Cpartial%20y_1%7D%20%7B%5Cpartial%20x_1%7D%20%26%20%5Ccdots%20%26%20%5Cfrac%20%7B%5Cpartial%20y_1%7D%20%7B%5Cpartial%20x_n%7D%20%5C%5C%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%20%5C%5C%20%5Cfrac%20%7B%5Cpartial%20y_m%7D%20%7B%5Cpartial%20x_1%7D%20%26%20%5Ccdots%20%26%20%5Cfrac%20%7B%5Cpartial%20y_m%7D%20%7B%5Cpartial%20x_n%7D%20%5Cend%20%7Bpmatrix%7D%20%3D%20%5Cbegin%20%7Bpmatrix%7D%20w_%7B11%7D%20%26%20%5Ccdots%20%26%20w_%7Bn1%7D%20%5C%5C%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%20%5C%5C%20w_%7B1m%7D%20%26%20%5Ccdots%20%26%20w_%7Bnm%7D%20%5Cend%20%7Bpmatrix%7D%20%3D%20%5Cboldsymbol%20W%5ET
