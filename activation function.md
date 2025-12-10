# ACTIVATION FUNCTION에 대한 내용을 다룹니다.

## 활성화 함수(ACTIVATION FUNCTION)이란?

### 쉽게 말하자면, 인공 신경에 복잡함을 더해주는 장치.

활성화 함수가 없다면, 정해진 선형 연산만 반복. --> 매 학습마다 똑같은 결과가 나옴

활성화 함수가 있다면, 비선형성의 도입으로 인하여 활성화 함수에 따라서 여러 패턴이 나타나게 됨. --> 매 학습마다 조금씩 다른 값이 나옴.

활성화 함수의 특성으로 인하여, 복잡한 패턴도 학습 가능. 이는 위에서 말한 여러 패턴이 나타나는 것, 경우의 다양성 증가와 연결됨.

### signoid

**1 / (1 + e^-x)** 형태의 함수.

### ReLU

**y = z (z > 0) , y = 0 (otherwise)**

### softmax

**양자역학의 파동함수와 normalization, orthonormal과 유사한 개념**

**y_i = Σ (e^z_i)/e^z_j, j to K** 형태. 

## tensorflow에서 activation function 사용법

다음과 같은 코드를 보자.

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

여기서, 

```

model=tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
	tf.keras.layers.Dense(2),
    tf.keras.layers.Dense(2)
])

```

에서 Dense에 ACTIVATION을 추가할 수 있다.

그러면,

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
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(2, activation='linear')
])

model.layers[0].set_weights([W,B])
model.layers[1].set_weights([W2,B2])

model.compile(optimizer='sgd',
            loss='mse')

Y=model.predict(X)
print(Y)

model.fit(X,YT,epochs=599)

print('W=',model.layers[0].get_weights()[0])
print('B=',model.layers[0].get_weights()[1])
print('W2=',model.layers[1].get_weights()[0])
print('B2=',model.layers[1].get_weights()[1])

Y=model.predict(X)
print(Y)

```

```

model=tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(2, activation='linear')
])

```

ACTIVATION FUNCTION은 tensorflow에 있는 함수이기 때문에, 저렇게 추가해주면 알아서 처리됨.

ACTIVATION FUNCTION을 적용하지 않으려면 linear 혹은 아예 지우면 됨.

## softmax의 경우

softmax는 평균 제곱 오차를 적용하기 힘들다.

일반적으로 평균 제곱 오차는 **E = (y - yT)^2 / 2** 지만, 소프트 맥스에서는 

**E = - Σ yYlogy** 형태로 표현됨. // 볼츠만 엔트로피 k_B logΩ 와 유사한 형태. 이를 크로스 엔트로피라고 함. E를 z에 대해 편미분하면, y - yT가 나옴.

또한, 목표값 yT가 단 하나만 1이고, 나머지는 0이어야만 함.

따라서, 위 코드에서 yT를 바꿔줘야하고,

```

model.compile(optimizer='sgd',
              loss='categorical_crossentropy') 

```

loss를 mse가 아닌 categorical_crossentropy로 바꿔줘야 함.







