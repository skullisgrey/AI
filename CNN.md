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

///작성중....
