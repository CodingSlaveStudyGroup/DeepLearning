# 3주차: 신경망 학습

## 목차 

1. 데이터
2. 손실함수
3. 경사 하강법
4. 간단한 학습 알고리즘 구현



## 데이터

기계학습을 할 때는 **데이터**가 중심이 된다. 

알고리즘을 밑바닥부터 설계하는 대신 데이터에서 **특징** _feature_ 를 추출하고 그 특징을 학습시키는 방법이 있다. 예를들어 컴퓨터 비전 분야에선 SIFT, SURF, HOG 등의 특징을 이용하며, 이미지 데이터를 벡터로 변환하고, 변환된 벡터를 가지고 지도 학습의 대표 분류 기법인 SVM, KNN 등으로 학습할 수 있다. 

신경망의 경우 이미지를 있는 그대로 학습시키게 된다. 즉, 신명망은 모든 문제를 주어진 데이터 그대로를 입력 데이터로 활용해 __end to end__ 로 학습할 수 있다. 

기계학습은 __훈련 데이터__ _training data_ 와 __시험 데이터__ _test data_ 로 나눠 학습을 수행하는 것이 일반적이다. 

훈련 데이터만으로 학습을 하며 최적의 매개변수를 찾는다. 시험 데이터는 __범용 능력__ 을 테스트하기 위해 사용한다. 범용 능력이란 훈련 데이터에 포함되지 않은 데이터로도 올바르게 문제를 풀어나가는 능력을 의미한다. 

특정 데이터셋에만 지나치게 최적화된 상태를 __오버피팅__ _overfitting_ 이라고 한다.



## 손실함수

__손실 함수__ _loss function_ 는 신경망 성능의 __나쁨__ 을 나타내는 지표이다. 해당 신경망이 데이터를 얼마나 처리를 __못__하느냐를 나타낸다.

정확도 대신 손실 함수를 사용하는 이유는 미분과 관련이 있다. 정확도를 사용하게 되는 경우 대부분 미분 값이 0이 나오므로 매개변수의 값을 바꾸기 어렵기 때문에 정확도 대신 나쁨을 나타내는 손실 함수를 사용하게 된다. 

다음은 많이 쓰이는 손실함수들이다. 

#### 평균 제곱 오차 _mean squared error, MSE_ 

수식은 다음과 같다.

![1538242330887](..\img\week3_2.png)

여기서 _yk_는 신경망의 출력, _tk_ 는 정답 레이블, _k_ 는 데이터 차원의 수이다. 

위 함수를 코드로 구현하면 아래와 같다.


```python
def mean_squaured_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)
```



#### 교차 엔트로피 오차 _cross entropy error, CEE_

수식은 다음과 같다.

![1538242330887](..\img\week3_1.png)

교재에선 log로 표기되어 있으나 밑이 _e_ 인 로그함수이므로 ln으로 표기하였다. 



위 함수를 코드로 구현하면 아래와 같다.


```python
def corss_entropy_error(y, t):
    delta = 1e-7 // np.log에 0이 들어가면 -inf가 되므로 작은 값을 더해줌
    return -np.sum(t * np.log(y + delta))
```


#### 미니 배치 학습

학습을 하는 경우 많은 데이터를 사용하게 된다. 이 때, 데이터를 대상으로 일일이 손실 함수를 모두 계산하게 되면 시간이 오래 걸리게 되므로 일부만 골라 근사치로 이용하게 된다.

이 때, 랜덤하게 골라낸 일부의 데이터를 __미니배치__ _mini-batch_ 라고 부르며, 이것을 이용하여 학습을 하는 방법을 __미니배치 학습__ 이라고 부른다.

넘파이의 np.random.choice() 함수를 사용하면 간단하게 무작위로 데이터를 골라낼 수 있다.

![1538242330887](..\img\week3_3.png)

여기서 에타는 __학습률__ 을 나타낸다. 한 번의 학습으로 얼마나 학습해야 할지, 매개변수 값을 얼마나 갱신해야 할지 결정하는 것이 학습률이다. 

학습률 값은 미리 정해놓아야 하는데, 이 값이 적절한 값을 가지지 못하면 최적의 값을 찾을 수 없다. 

## 경사 하강법

가장 최적의 매개변수를 찾는 것은 손실 함수의 최솟값이 될 때의 매개변수를 찾는 것이다.

함수의 가장 최소값이 되는 부분은 기울기가 0이 된다. 

주의할 점은 기울기가 0이라고 해서 반드시 최솟값이라는 보장은 없다는 것이다. __고원__ _plateau_ 라고 하는 학습이 진행되지 않는 정체기에 빠질 수 있다. 

경사 하강법은 다음과 같이 구현할 수 있다. 

```python
// f == 최적화 하려는 함수
// init_x == 초깃값
// lr == 학습률 (learning rate)
// step_num == 반복 학습 횟수
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr *grad
    return x
```

학습률과 같은 매개변수를 __하이퍼파라미터__ _hyper parameter_ 라고 한다. 



## 간단한 학습 알고리즘 구현

__1단계 - 미니배치__

__2단계 - 기울기 산출__

__3단계 - 매개변수 갱신__

__4단계 - 1~3단계 반복__



이 방법은 경사 하강법으로 매개변수를 갱신하는 방법이며, 미니배치를 이용하므로 __확률적 경사 하강법__ _stochastic gradient descent, SGD_ 라고 부른다.  

MNIST 데이터셋을 이용해 학습을 수행하는 것은 교재를 참고하도록 한다. 

__에폭__ _epoch_은 학습에서 훈련 데이터를 모두 소진했을 때의 횟수에 해당한다. 