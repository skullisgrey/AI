# pytorch에 대해 다룹니다.

## tensorflow에서 내용 참고하면 좋음

## pytorch 설치

**pip install torch-directml** 입력.

만약 다음과 같이 나온다면

<img width="1217" height="809" alt="image" src="https://github.com/user-attachments/assets/0de998e8-48f2-4120-b31e-7ca4d6904ccf" />

**pip install --user torch-directml** 입력하여 관리자 권한으로 설치.

## 1 입력 1 출력 인공신경 구현

```

import torch
import torch.nn as nn
import torch.optim as optim

X = torch.FloatTensor([[2]])
YT = torch.FloatTensor([[10]])
W = torch.FloatTensor([[3]])
B = torch.FloatTensor([1])

model = nn.Sequential(
    nn.Linear(1,1)
)
print(model)

with torch.no_grad():
    model[0].weight = nn.Parameter(W)
    model[0].bias = nn.Parameter(B)
    
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    
    Y = model(X)
    E = loss_fn(Y, YT)
    optimizer.zero_grad()
    E.backward()
    optimizer.step()
    
    if epoch%100 == 99:
        print(epoch, E.item())
        
print(f'W = {model[0].weight.data}')
print(f'B = {model[0].bias.data}')

Y = model(X)
print(Y.data)


```

### pytorch는 입력, 가중치를 2rank tensor(matrix)로 받도록 설계되어있기에 ([[ ]]) 형태로 넣음.<br>단, 편향은 벡터로 넣고, 갯수는 예측값만큼.

### nn.Linear(a, b)에서 a는 입력수 b는 출력수. // Linear말고 RELU, SOFTMAX등 tensorflow에서 쓴<br>활성화 함수를 쓸 수 있음.

### no_grad ==> 이 안은 자동 미분 안 함. --> 가중치, 편향이 autograd에 기록x 

### mm.MSELoss() --> 평균 제곱 오차. 소프트맥스라면 CrossEntropyLoss() 사용

### optimizer --> 가중치 업데이트 방식. 여기서는 SGD사용, lr은 학습률.

## 2입력 1 출력

```

import torch
import torch.nn as nn
import torch.optim as optim

X = torch.FloatTensor([[2,3]])
YT = torch.FloatTensor([[27]])
W = torch.FloatTensor([[3,4]])
B = torch.FloatTensor([1])

model = nn.Sequential(
    nn.Linear(2,1)
)
print(model)

with torch.no_grad():
    model[0].weight = nn.Parameter(W)
    model[0].bias = nn.Parameter(B)
    
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    
    Y = model(X)
    E = loss_fn(Y, YT)
    optimizer.zero_grad()
    E.backward()
    optimizer.step()
    
    if epoch%100 == 99:
        print(epoch, E.item())
        
print(f'W = {model[0].weight.data}')
print(f'B = {model[0].bias.data}')

Y = model(X)
print(Y.data)


```

### 입력이 1개에서 2개로 변경(가중치도 2개로), Linear(입력갯수, 출력갯수) 에서 입력갯수 2로 변경.


## 2입력 2출력

```

import torch
import torch.nn as nn
import torch.optim as optim

X = torch.FloatTensor([[2,3]])
YT = torch.FloatTensor([[27, -30]])
W = torch.FloatTensor([[3,4],[5,6]])
B = torch.FloatTensor([1,2])

model = nn.Sequential(
    nn.Linear(2,2)
)
print(model)

with torch.no_grad():
    model[0].weight = nn.Parameter(W)
    model[0].bias = nn.Parameter(B)
    
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    
    Y = model(X)
    E = loss_fn(Y, YT)
    optimizer.zero_grad()
    E.backward()
    optimizer.step()
    
    if epoch%100 == 99:
        print(epoch, E.item())
        
print(f'W = {model[0].weight.data}')
print(f'B = {model[0].bias.data}')

Y = model(X)
print(Y.data)

```

### 출력이 2개가 되었으므로 결과값이 2개가 됨, 이로 인해 편향 2개로.

### 가중치가 총 4개가 되어야하므로, 2x2행렬이 됨.

### Linear(입력갯수, 출력갯수)에서 출력이 2개이므로 2가 됨.


## 입력2 은닉2 출력2

```

import torch
import torch.nn as nn
import torch.optim as optim

X = torch.FloatTensor([[.05, .10]])
YT = torch.FloatTensor([[.01, .99]])
W = torch.FloatTensor([[.15, .20], [.25, .30]])
B = torch.FloatTensor([.35, .35])
W2 = torch.FloatTensor([[.40, .45], [.50, .55]])
B2 = torch.FloatTensor([.60, .60])

model = nn.Sequential(
    nn.Linear(2,2),
    nn.Linear(2,2)
)
print(model)

with torch.no_grad():
    model[0].weight = nn.Parameter(W)
    model[0].bias = nn.Parameter(B)
    model[1].weight = nn.Parameter(W2)
    model[1].bias = nn.Parameter(B2)
    
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    
    Y = model(X)
    E = loss_fn(Y, YT)
    optimizer.zero_grad()
    E.backward()
    optimizer.step()
    
    if epoch%100 == 99:
        print(epoch, E.item())
    
print(f'W = {model[0].weight.data}')
print(f'B = {model[0].bias.data}')
print(f'W2 = {model[1].weight.data}')
print(f'B2 = {model[1].bias.data}')

Y = model(X)
print(Y.data)

```

### 처음 입력의 출력이 은닉층의 입력으로 들어가서, 출력. 이 값이 최종출력.<br>따라서, Linear(2,2)가 2개.

### 가중치는 2x2 행렬이 2개, 편향은 1x2 벡터가 2개가 됨.

### model 옆의 숫자는 층 수. 0부터 시작하므로 두번째는 1이 됨.

## 활성화 함수 적용(relu, sigmoid)

```

import torch
import torch.nn as nn
import torch.optim as optim

X = torch.FloatTensor([[.05, .10]])
YT = torch.FloatTensor([[.01, .99]])
W = torch.FloatTensor([[.15, .20], [.25, .30]])
B = torch.FloatTensor([.35, .35])
W2 = torch.FloatTensor([[.40, .45], [.50, .55]])
B2 = torch.FloatTensor([.60, .60])

model = nn.Sequential(
    nn.Linear(2,2),
    nn.ReLU(),
    nn.Linear(2,2),
    nn.Sigmoid()
)
print(model)

with torch.no_grad():
    model[0].weight = nn.Parameter(W)
    model[0].bias = nn.Parameter(B)
    model[2].weight = nn.Parameter(W2)
    model[2].bias = nn.Parameter(B2)
    
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    
    Y = model(X)
    E = loss_fn(Y, YT)
    optimizer.zero_grad()
    E.backward()
    optimizer.step()
    
    if epoch%100 == 99:
        print(epoch, E.item())
    
print(f'W = {model[0].weight.data}')
print(f'B = {model[0].bias.data}')
print(f'W2 = {model[2].weight.data}')
print(f'B2 = {model[2].bias.data}')

Y = model(X)
print(Y.data)



```

### 은닉 층에 ReLU, 출력에 Sigmoid 적용.

### 층이 달라졌으므로 model 옆의 값이 달라짐. 유의할 것.


## softmax / crossentropy

```

import torch
import torch.nn as nn
import torch.optim as optim

X = torch.FloatTensor([[.05, .10]])
YT = torch.FloatTensor([[0, 1]])
W = torch.FloatTensor([[.15, .20], [.25, .30]])
B = torch.FloatTensor([.35, .35])
W2 = torch.FloatTensor([[.40, .45], [.50, .55]])
B2 = torch.FloatTensor([.60, .60])

model = nn.Sequential(
    nn.Linear(2,2),
    nn.ReLU(),
    nn.Linear(2,2)
)
print(model)

with torch.no_grad():
    model[0].weight = nn.Parameter(W)
    model[0].bias = nn.Parameter(B)
    model[2].weight = nn.Parameter(W2)
    model[2].bias = nn.Parameter(B2)
    
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100000):
    
    Y = model(X)
    E = loss_fn(Y, YT)
    optimizer.zero_grad()
    E.backward()
    optimizer.step()
    
    if epoch%100 == 99:
        print(epoch, E.item())
    
print(f'W = {model[0].weight.data}')
print(f'B = {model[0].bias.data}')
print(f'W2 = {model[2].weight.data}')
print(f'B2 = {model[2].bias.data}')

Y = model(X)
Y = nn.functional.softmax(Y, dim=1)
print(Y.data)

```

### softmax이므로 결과값은 0 혹은 1, 오차는 CrossEntropyLoss

### Y = nn.functional.softmax(Y, dim=1)로 Y값을 softmax 출력 형태로 변경.

/// 더 작성중



