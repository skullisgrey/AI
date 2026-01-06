# FPGA에 구축하는 MNIST CNN에 대해 다룹니다.


# 프로젝트 목표

## verilog를 통하여 Mnist CNN을 구현, 이를 Test bench에서 시뮬레이션하여 검증

## 사용 툴

Vivado 2022.2를 사용

## CNN 구조

<img width="320" height="390" alt="image" src="https://github.com/user-attachments/assets/1c6992b2-d92d-4bcc-8e20-55364d2d0893" />

conv Input : 1 x 56 x 56 확장된 mnist 데이터 셋 사용, padding : 1 

conv weight : 1 x 3 x 3 

bias : 1 x 1

conv output : 1 x 56 x 56

//////////////////////////////////////////////////////

max pooling input : 1 x 56 x 56

max pooling output : 1 x 28 x 28

//////////////////////////////////////////////////////

fully connected input : 1 x 28 x 28

fully connected weight : 7840

fully connected bias : 10

fully connected output : 10

//////////////////////////////////////////////////////

FPGA에서는 padding 보정을 위해 conv input data를 1 x 58 x 58를 사용.

//////////////////////////////////////////////////////

## Vivado Simulation 분석

Test bench는 다음과 같이 사용. (mem파일은 mnist의 숫자 0에 해당되는 이미지를 58x58(패딩포함)으로 확장시킨 이미지)

```

`timescale 1ns / 1ps

module conv_tb();
    reg clk = 0;
    reg [7:0] data_in = 0;
//    reg rst = 0;
    reg data_valid = 0;
    
    layer1_top top_tb(.clk(clk), .data_in(data_in), .data_valid(data_valid));
    
    always #10 clk = ~clk;
    
    reg [7:0] img_buffer [0:3363]; // 56x56 크기 메모리
    integer i;
    
    initial begin
    // 1. 생성된 메모리 파일 로드
    
    #1000  
    @(posedge clk);
    data_valid <= 1;
    data_in    <= img_buffer[0];   
    $readmemh("mnistreal_0.mem", img_buffer, 0, 3363);
  
    // 3. 데이터를 픽셀 스트림처럼 한 개씩 공급
    for (i = 0; i < 3364; i = i + 1) begin
        @(posedge clk);
        data_in = img_buffer[i];
    end
   @(posedge clk);
    data_valid = 0;
end

endmodule

```

### 이 Test bench set은 AXI_DMA모듈을 이용하여 데이터 스트림을 하는 것과 같은 효과.

### Convolution

<img width="1411" height="634" alt="image" src="https://github.com/user-attachments/assets/0f47e111-d8b2-4449-81e8-96aa9dbda427" />

첫번째 Line buffer의 데이터가 다음 Line buffer에 옮겨지는데 걸리는 시간 : 1160 ns /// 1160 / 20 = 58 --> Line buffer 하나에 픽셀 데이터 58개, 가로 1줄 저장.




<img width="1483" height="675" alt="image" src="https://github.com/user-attachments/assets/95952b76-7138-4664-a3f6-8ceb7b3fe1ae" />

Weight와 곱해져서 나온 값 acc에서, ReLU에 의해 음수는 0으로, 양수는 최대 255까지 출력. // 데이터 오버플로우 문제 방지

<img width="1471" height="640" alt="image" src="https://github.com/user-attachments/assets/890a23cc-1630-4677-819c-b78becfb0b15" />

<img width="1471" height="632" alt="image" src="https://github.com/user-attachments/assets/c17370a4-1f9a-4603-9df2-84c7a8fb70c4" />


out valid 하나의 on time : 1120 ns, 하나의 데이터에 56번의 out valid on /// 1120 / 20 = 56 --> 테두리의 padding 부분을 제외한 데이터. 총 1 x 56 x 56개의 데이터


### Max pooling 2 x 2

<img width="1481" height="622" alt="image" src="https://github.com/user-attachments/assets/8b6f5fbb-ac97-44fb-ae64-09a5ed6606bf" />

Line buffer에 데이터가 들어오는 데 걸리는 시간 : 40 ns /// 2개의 데이터가 들어오면 둘 중 큰 값만 buffer에 반영하기 때문에 2클럭 = 40 ns 소모


<img width="1255" height="592" alt="image" src="https://github.com/user-attachments/assets/a3adcd4f-00de-4f25-aaf4-a7377df0cb5f" />

<img width="1253" height="594" alt="image" src="https://github.com/user-attachments/assets/fe54428a-9de8-4a4a-9cb9-21fa441f758e" />

<img width="1256" height="588" alt="image" src="https://github.com/user-attachments/assets/367492b2-05db-4a37-aa61-6080511b12d5" />


한 패킷의 out valid의 on-off 주기 : 40ns /// 최종적으로 2x2를 1x1로 줄이는데 걸리는 시간.

한 패킷에 28번의 out valid가 on, 28개의 패킷 /// 1 x 28 x 28개의 데이터.


### Fully connected

<img width="1167" height="580" alt="image" src="https://github.com/user-attachments/assets/4c267bd5-4550-4d12-b1e8-e13d1a4016cd" />

<img width="1179" height="586" alt="image" src="https://github.com/user-attachments/assets/33d6100f-7680-4dc6-87c1-1f2ea0cc592f" />

<img width="1180" height="585" alt="image" src="https://github.com/user-attachments/assets/0c9550d5-618d-4edd-8537-a72a759c2488" />

x에 데이터가 쌓이기 시작하고, 모든 데이터가 쌓이면 데이터에 weight를 곱하고 더하기 시작.

acc가 계산이 완료되는데 걸리는 시간 : 156.8μs /// 156800 / 20 = 7840 총 7840번 연산


<img width="1171" height="583" alt="image" src="https://github.com/user-attachments/assets/ebbd16a3-6e26-4c48-a9d6-75522dc24d2e" />

최종적으로 10개의 acc를 비교하여, 가장 큰 값을 고름 --> 최종추론

현재 Test bench에서 숫자 0 데이터를 넣어 추론이 0으로 나옴.

## 간단 검증

<img width="655" height="265" alt="image" src="https://github.com/user-attachments/assets/c40c8569-065f-4229-b2a7-dc9b2f1b6900" />


0 ~ 9까지의 이미지 데이터를 순서대로 넣었을 때, 1과 3을 제외하고 추론과 실제값이 맞아 떨어짐.

각 숫자당 1개씩 총 10번의 검증밖에 없으므로, 표본이 적어 신뢰도가 높지는 않음. 더 많은 데이터를 넣어서 시뮬레이션 해 볼 것을 권장.


## 현재 문제점

각 상태별(Valid)마다 타이밍 여유가 없음. 실제 하드웨어에 구현시에는 오동작 가능.

의도적 지연을 통하여 여유를 줘야함. 현재 구조는 시뮬레이션용 그 이상도 그 이하도 아님.
