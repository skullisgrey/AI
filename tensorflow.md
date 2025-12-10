# TENSORFLOW에 대해 다룹니다.

## TENSOR란?

### 간단히 보면 수학적으로는 회전에 관한 개념. 여기서는 회전에 관한 개념은 없이, 차원과 구조만 차용함.

텐서에는 랭크가 있고, rank 0 = scalar, rank 1 = vector, rank 2 = matrix로 볼 수 있다.(여기서 모든 matrix가 rank2 tensor는 아니지만, rank2 tensor는 NxN matrix로 표기 가능.)

기본적으로 데이터를 matrix에 저장하는데, 여러 데이터가 있다면 다차원 matrix로 볼 수 있다.

여기서, tensor의 개념을 이용할 수 있다. 예를 들어 rank 3의 경우, NxN의 matrix가 N개 쌓인 형태로 볼 수 있다.

## TENSORFLOW 설치법

cmd에서, 원하는 폴더에 들어간 뒤, **pip3 install-directml-plugin**을 입력하여 라이브러리를 설치하여 준다.

단, python이 window store에서 설치한 파일인 경우 다음과 같이 뜬다.

<img width="1016" height="309" alt="image" src="https://github.com/user-attachments/assets/7f24bfa7-40cb-4b25-b1b5-0352c161cc8a" />

이 때는 다른 python을 사용해줘야 한다.

TENSORFLOW를 설치했다면, 같은 폴더에서 **pip3 install opencv-python**을 입력하여 라이브러리를 설치하여 준다.

## TENSORFLOW로 인공 신경망 구현하기

### 2입력 2출력에 관한 예시다.

```

import tensorflow as tf
import numpy as np

X=np.array([[.05,.10]])
YT=np.array([[.01,.99]])
W=np.array([[.15,.25],[.20,.30]])
B=np.array([.35,.35])
W2=np.array([[.40,.50],[.45,.55]])
B2=np.array([.60,.60])

model=tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
	tf.keras.layers.Dense(2),
    tf.keras.layers.Dense(2)
])

model.layers[0].set_weights([W,B])
model.layers[1].set_weights([W2,B2])

model.compile(optimizer='sgd',
             loss='mse')
		
Y=model.predict(X)
print(Y)

model.fit(X,YT,epochs=999)

print('W=',model.layers[0].get_weights()[0])
print('B=',model.layers[0].get_weights()[1])
print('W2=',model.layers[1].get_weights()[0])
print('B2=',model.layers[1].get_weights()[1])

Y=model.predict(X)
print(Y)

```

### 코드 해석

np.array는() 숫자의 나열을 생각하면 된다.

벡터는 ([ ]), 행렬은 ([[  ]])로 나타난다. [ ] 는 하나의 줄을 생각하면 된다. 따라서 2x2 행렬은 np.array([[a_11,a_12],[a_21,a_22]]) 로 나타낸다.

```
model=tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
	tf.keras.layers.Dense(2),
    tf.keras.layers.Dense(2)
])

```

이거는 층을 순서대로 쌓는 모델을 만듦. TENSORFLOW는 라이브러리가 keras이므로 keras 사용. (pytorch의 경우는 keras가 아닌 다른거 사용.)

입력은 2개이고, 2,는 2개의 값을 가진 벡터를 의미.

Dense는 출력이 2개. --> 입력이 2개, 출력이 2개가 됨.

```

model.layers[0].set_weights([W,B])
model.layers[1].set_weights([W2,B2])

model.compile(optimizer='sgd',
             loss='mse')
		
Y=model.predict(X)

```

초기 가중치, 편향값을 layers[0], layers[2]에서 지정.

compile에서는, 가중치와 편향값 업데이트에 관한 정보이고, sgd는 확률적 경사 하강법, mse는 평균 제곱 오차를 의미.

Y는 초기 X값을 넣어서 나오는 예측값.

```

model.fit(X,YT,epochs=999)

```

YT에 접근할 때 까지, 위의 작업을 반복. epochs = 999면 1000번 반복.

## 라이브러리마다 차이는 있지만, 기본적으로는 입력 - 예측값 - 출력값 개념.

### 처음 입력에 의해 나오는 예측값을 출력값과 비교하여, 가중치와 편향값이 조절되면서 다음 입력에서 예측값이 변경, 가중치 편향값 조절... 최종적으로 예측값이 출력값에 수렴하게 됨.

