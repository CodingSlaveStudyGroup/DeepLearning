# 오차역전파법 부록

여기에는 그다지 몰라도 상관 없는 내용들을 담아 두었다.

## 계산 그래프와 친해지기

본문에서 우려먹은 문제를 계산 그래프로 그려보자. 문제는 다음과 같다.

> 랭호가 한 궘에 6600원짜리 라이트노벨 2권을 골라서 달러화로 결제했다. 그 때의 원-달러 환율은 1100원이고, 배송비 5달러가 더 부과됐다. 랭호는 몇 달러를 지불했을까?

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
        return x + y
    
    def backward(self, dout):
        return (dout, dout)
```

### 곱셈 노드

곱셈은 2개의 입력을 받아, 순전파 시에는 두 입력을 더하고, 역전파 시에는 미분값에 입력을 *바꿔서* 곱한다.

```python
class Multiply:
    def __init__(self):
        self.x = None
	self.y = None

    def forward(self, x, y):
        self.x = x
	self.y = y
	return x * y

    def backward(self, dout):
        dx = dout * self.y
	dy = dout * self.x
	return (dx, dy)
```

### 나눗셈 노드

나눗셈은 2개의 입력 x, y를 받아, 순전파 시에는 x를 y로 나눠 반환하고, 역전파 시에는 각각 y로 나누고, -x를 곱한 뒤 y의 제곱으로 나눠 반환한다.

```python
class Divide:
    def __init__(self):
        self.x = None
	self.y = None

    def forward(self, x, y):
        self.x = x
	self.y = y
	return x / y

    def backward(self, dout):
        dx = dout / self.y
	dy = dout * (-self.x) / (self.y ** 2)
	return (dx, dy)
```

### 그래프 구현

이제 노드도 준비되었으니 실제로 구현해보자.

```python
lnovel = 6600
lnovel_num = 2
won_dollar_conversion = 1100
shipping_fee = 5

lnovel_multiple_layer = Multiply()
currency_convert_layer = Divide()
shipping_fee_layer = Add()

# 순전파
lnovel_price = lnovel_multiple_layer.forward(lnovel, lnovel_num)
lnovel_price_dollar = currency_convert_layer.forward(lnovel_price, won_dollar_conversion)
final_price = shipping_fee_layer.forward(lnovel_price_dollar, shipping_fee)


