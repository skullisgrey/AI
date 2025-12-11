# mnist에 대해 다룹니다.

## mnist

손글씨 숫자 데이터.

텐서 플로우에 내장되어 있고, 다음과 같은 형태로 불러올 수 있다.

```

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(X, YT) , (x, yt) = mnist.load_data()

```

tensorflow를 불러오고. 그 안에 있는 mnist dataset을 사용

데이터를 (X, YT), (x, yt)에 로드. 여기서 데이터는 X는 학습용 손글씨 숫자 데이터, x는 시험용 손글씨 숫자 데이터.
 
YT는 학습용 손글씨 숫자 라벨, yt는 시험용 손글씨 숫자 라벨. // 각 60000개, 10000개.

X와 x는 전부 28x28픽셀로 구성된 그림이고, 1픽셀의 크기는 8비트 --> 0~255의 값을 가짐.

## 예제

```

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(X, YT), (x,yt) = mnist.load_data()

X, x = X/255, x/255
X, x = X.reshape((60000,784)), x.reshape((10000,784))

model = tf.keras.Sequential([
    tf.keras.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X,YT,epochs=5)

model.evaluate(x,yt)

```

X, x를 0~1의 값으로 바꿔서 잘 학습되도록 만들기.

reshape는 행렬을 벡터로 바꾸기 위해서 사용. Dense에서는 벡터만 들어갈 수 있음. /// a x b 행렬을 (a + b) x 1 행렬인 벡터로 변경.

shape에 784는 mnist의 그림이 28x28 픽셀이기 때문. // 28 x 28 = 784

activation을 softmax를 쓰기 때문에, loss는 mse가 아닌 crossentropy 사용.

metrics는 학습 중 볼 정보 --> 여기서는 정확도를 선택.

여기서 epoch는 5지만, 60000개의 학습 데이터가 있기 때문에, 1875가 나옴. // keras의 기본 batch size = 32. 60000/32 = 1875

evaluate는 loss와 accuracy 출력.

### 이미지 및 예측값/실제값 비교

```

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(X, YT), (x, yt) = mnist.load_data()

X, x = X/255, x/255
X, x = X.reshape((60000,784)), x.reshape((10000,784))

model = tf.keras.Sequential([
    tf.keras.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'linear')
])

model.compile(optimizer = 'adam', loss = 'mse')

model.fit(X,YT,epochs=20)

y=model.predict(x)
print(y[0])

import matplotlib.pyplot as plt

x=x.reshape(10000,28,28)

plt.figure(figsize=(10,10))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x[i],cmap=plt.cm.binary)
    plt.xlabel(f'y{y[i]} yt{yt[i]}')
plt.show()

```

실행하면 다음과 같은 플롯이 뜬다.

<img width="1538" height="884" alt="image" src="https://github.com/user-attachments/assets/d07add50-9973-4abf-ba0b-fb839cab2c7d" />

여기서, 플롯을 for i in range(100):
    plt.subplot(10,10,i+1) 이렇게 해놨기 때문에 10x10 총 100장이 뜬다.

y값이 예측값이고, yt가 실제값이다.
