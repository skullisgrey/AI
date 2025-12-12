# CNN에 대해 다룹니다.

## CNN이란?

Convolutional Neural Network의 약자. 합성곱 신경망이라고도 함.

이미지, 영상, 음성, 시계열 데이터 처리에 특화됨.

이미지에서 작은 필터(kernel)을 움직이면서 연산 --> 커널을 통해 구석구석 살펴보는 걸로 이해 가능.

필터는 N x N 행렬.

필터가 이동하는 단위를 stride라고 한다.

### 예시

<img width="100" height="100" alt="image" src="https://github.com/user-attachments/assets/0b22e949-5777-4071-9767-c656eb3c7b36" />

다음과 같은 100x100 이미지가 있다고 하자. 그리고 필터를 3x3으로 쓴다면 다음과 같이 볼 수 있다. 필터는 검정색이다.

<img width="100" height="100" alt="image" src="https://github.com/user-attachments/assets/d4275f85-25e8-4e30-9472-6cbf380b46fc" />

만일 stride가 1이라면 다음과 같이 이동한다.

<img width="100" height="100" alt="image" src="https://github.com/user-attachments/assets/0cc8b6c2-b5d1-4cf0-bfc3-44ad4956e2e4" />

stride가 10이라면 다음과 같다.

<img width="100" height="100" alt="image" src="https://github.com/user-attachments/assets/07e7ce92-b63d-4c1e-949e-fa210467b464" />

stride에 따라 필터가 이동하면서, 이미지를 훑고 지나간다.

## CNN에서 convolution 값

필터에 들어온 이미지 영역의 값을 각 성분에 대해 곱한다. --> C_ij = A_ij x B_ij  // 수학에서의 행렬곱 C_ij = ΣA_ik x B_kj 와 다름.

이 후, 행렬의 성분들을 전부 합친다. 이 값이 convolution 값이다.

```

import numpy as np

np.random.seed(1)

image = np.random.randint(5, size=(3,3))
print('image = \n', image)

filter = np.random.randint(5, size=(3,3))
print('filter = \n', filter)

image_x_filter = image * filter
print('image_x_filter = \n', image_x_filter)

convolution = np.sum(image_x_filter)
print('convolusion = \n', convolution)

```

이 예시를 통하여 빠르게 이해 할 수 있다.

## kernel을 이용한 필터링

```

import numpy as np
import cv2
import matplotlib.pyplot as plt

image_color = cv2.imread('google.png')
print('image_color.shape =', image_color.shape)
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
print('image.shape =', image.shape)

filter = np.array([
    [-1,-1,-1],
    [-1,9,-1],
    [-1,-1,-1]
])

image_pad = np.pad(image,((1,1),(1,1)))
print('image_pad.shape = ', image_pad.shape)

convolution = np.zeros_like(image)

for row in range(image.shape[0]):
    for col in range(image.shape[1]):
        window = image_pad[row:row+3, col:col+3]
        convolution[row, col] = np.clip(np.sum(window*filter), 0, 255)

images = [image, convolution]
labels = ['gray', 'convolution']

plt.figure(figsize=(10,5))
for i in range(len(images)):
    plt.subplot(1,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(images[i], cmap=plt.cm.gray)
    plt.xlabel(labels[i])
plt.show()

```

회색으로 필터링 된 이미지와, 지정한 3x3 필터값에 따른 필터링 된 이미지를 비교하는 코드이다.

convolution[row, col] = np.clip(np.sum(window*filter), 0, 255)에 의해 최종적으로 8비트 흑백 이미지로 출력된다.

여기서 이미지는 구글 로고를 사용하였고 결과는 다음과 같이 나온다.

<img width="894" height="410" alt="image" src="https://github.com/user-attachments/assets/3d91e333-d43a-418f-9136-902fb244935b" />

필터를 이용하여, 필터값에 따른 가중치를 이용하여 새로운 값을 만들어내는 것이 CNN과 유사하다고 볼 수 있음.

np.sum --> 커널 영역 합산

window*filter --> CNN에서 conv2D연산

np.clip --> 간단한 활성화 혹은 정규화

간단히 말하자면, CNN은 고급필터라고 생각하면 나름 머리가 아프지 않을 것.

## CNN 맛보기

```
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((60000,28,28,1))
x_test = x_test.reshape((10000,28,28,1))

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)
```

입력 : 28x28x1의 이미지

첫 출력 : 28x28x32 (3x3의 필터 32개 사용)

두번째 출력 : 28x28x64 (3x3의 필터 64개, 단 여기서 각 장 마다 앞서 실행한 필터링에서 얻은 정보가 포함되어 있음.) // 전체 필터는 3x3x32x64가 됨. 혼동 주의

MaxPooling2D --> a x b 영역을 1x1영역으로 압축시키는 작업. 여기서 나오는 값은 a x b 영역 내의 최댓값. // 여기선 14x14x64로 줄어듦

tf.keras.layers.Flatten() --> 행렬을 벡터로 바꾸는 작업. Dense에서 쓰기 위함.

Dense - 뉴런. 처음에는 128개의 뉴런과 연결, 그 후 10개의 뉴런과 연결 --> 14x14x64개의 성분이 들어가 128개의 아웃풋이 나오고, 이게 들어가서 최종적으로 10개의 아웃풋.

model.summary() --> 여러 파라미터 값.

이를 실행하면 다음과 같은 로그가 나온다.

```

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 32)        320       
                                                                 
 conv2d_1 (Conv2D)           (None, 28, 28, 64)        18496     
                                                                 
 max_pooling2d (MaxPooling2D  (None, 14, 14, 64)       0         
 )                                                               
                                                                 
 flatten (Flatten)           (None, 12544)             0         
                                                                 
 dense (Dense)               (None, 128)               1605760   
                                                                 
 dense_1 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 1,625,866
Trainable params: 1,625,866
Non-trainable params: 0
_________________________________________________________________

1875/1875 [==============================] - 10s 5ms/step - loss: 0.1113 - accuracy: 0.9664
Epoch 2/5
1875/1875 [==============================] - 9s 5ms/step - loss: 0.0365 - accuracy: 0.9881
Epoch 3/5
1875/1875 [==============================] - 9s 5ms/step - loss: 0.0216 - accuracy: 0.9927
Epoch 4/5
1875/1875 [==============================] - 9s 5ms/step - loss: 0.0147 - accuracy: 0.9949
Epoch 5/5
1875/1875 [==============================] - 9s 5ms/step - loss: 0.0101 - accuracy: 0.9965

313/313 [==============================] - 1s 2ms/step - loss: 0.0445 - accuracy: 0.9871 

```

### param# 해석

param 수는 기본적으로 학습으로 조절할 수 있는 숫자들의 총합이라고 볼 수 있음.

첫번째 : 3x3x32 + 32 // 32는 편향 수. 필터 수와 같음.

두번째 : 3x3x32x64 + 64 

세번째, 네번째 : 파라미터가 없는 연산. 따라서 파라미터는 0 // 값 자체가 바뀌는 게 아닌, 일종의 재배치이기 때문.

다섯번째 : 12544x128 + 128 

여섯번째 : 128x10 + 10

### epoch 및 accuracy 관련

훈련이 될수록, accuracy가 올라감 --> 훈련 데이터에 대한 정확도. 여기서 훈련 데이터는 x_train = x_train.reshape((60000,28,28,1))에서.

313/313 [==============================] - 1s 2ms/step - loss: 0.0445 - accuracy: 0.9871 

훈련한 뒤, 새로운 데이터에 대한 정확도. 여기서 새로운 데이터는 x_test = x_test.reshape((10000,28,28,1))에서.

손글씨 숫자의 경우, 훈련 중 정확도도 높고, 응용했을 때 정확도도 꽤나 높음. --> 일반화가 잘 됨.

만약 훈련 하면서 accuracy가 높지만, 응용할 때 accuracy가 낮으면 과적합. --> 일반화가 안 됨됨.
