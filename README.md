# ✈️ WING IT

> 항공권 가격 흐름을 바탕으로 향후 30일 안에서 유리한 구매 시점을 추천하는 ML 서비스

**"언제 사야 가장 쌀까?"** 라는 질문에 답하기 위해,  
과거 항공권 가격의 시간적 변동 패턴을 학습하고 예상 가격 흐름을 비교해 구매 시점 Top 3를 추천합니다.

---

## Overview

항공권 가격은 예약 시점, 출발일, 수요 변화에 따라 계속 달라집니다.  
사용자 입장에서는 지금 구매할지 조금 더 기다릴지 판단하기 어렵고, 기존 서비스도 대부분 현재 시점의 가격 비교에 집중되어 있습니다.

WING IT은 이 문제를 줄이기 위해 과거 가격 흐름을 학습하고, 향후 30일의 예상 가격을 비교해 구매 시점 Top 3를 추천합니다.

---

## Tech Stack

| 분류 | 기술 |
|------|------|
| ML Pipeline | Python · SageMaker Pipeline · XGBoost · MLflow · S3 |
| Serving | Lambda · API Gateway · SageMaker Endpoint |
| Frontend | HTML · CSS · JavaScript |

---

## Project Structure

```
WINGIT/
├── frontend/
│   └── index.html
├── backend/
│   └── lambda_function.py
├── workflow/
│   ├── pipeline.py
│   ├── config.yaml
│   └── steps/
│       ├── preprocess.py
│       ├── train.py
│       ├── test.py
│       ├── register.py
│       └── deploy.py
└── README.md
```

---

## Request Flow

```
Client → API Gateway → Lambda → SageMaker Endpoint → Response
```

사용자 요청이 들어오면 Lambda에서 입력값을 정리하고 향후 30일 후보 날짜를 만든 뒤,  
SageMaker Endpoint 예측 결과를 바탕으로 가격 흐름과 구매 시점 Top 3를 정리해 응답합니다.

---

## Core Features

### 1. 항공권 구매 시점 추천
향후 30일의 예상 가격을 비교해 상대적으로 유리한 구매 시점 Top 3를 추천합니다.

### 2. 가격 흐름 시각화
예측 결과를 그래프로 함께 보여줘, 지금 구매할지 기다릴지 흐름을 보고 판단할 수 있게 했습니다.

### 3. 실시간 추론 API
API Gateway, Lambda, SageMaker Endpoint를 연결해 사용자 요청에 따라 예측 결과를 바로 반환합니다.

---

## Core Implementation

### 1. Feature Engineering

가격의 최근 흐름이 반영되도록 피처를 구성하고, 미래 정보가 섞이지 않도록 시점을 기준으로 데이터를 나눴습니다.

**주요 피처**

| 피처 | 설명 |
|------|------|
| `days_until_departure` | 출발까지 남은 일수 |
| `purchase_day_of_week` | 구매 요일 |
| `purchase_time_bucket` | 구매 시간대 |
| `is_holiday_season` | 성수기 여부 |
| `is_weekend_departure` | 주말 출발 여부 |
| `stops_count` | 경유 횟수 |
| `route_hash` | 노선 식별자 |
| `prev_fare` | 직전 가격 |
| `min/mean_fare_last_Nd` | 최근 N일 최솟값·평균값 |
| `prev_fare_vs_min/mean_Nd` | 직전 가격 대비 비율 |

**설계 포인트**
- 예측 시점보다 앞선 가격 기록만 사용해 피처를 구성
- 최근 7·14·30일 기준 통계를 추가해 가격 흐름 반영
- target에 `log1p` 변환 적용
- 노선별 시간 흐름을 유지한 상태로 학습·검증·테스트 분할

---

### 2. ML Pipeline

SageMaker Pipeline으로 전처리부터 배포까지의 흐름을 단계별로 구성했습니다.

**Preprocess**  
학습과 추론에 같은 피처 기준을 쓰기 위해 featurizer를 먼저 만들고, 전처리 산출물을 별도로 정리했습니다.

**Train**  
XGBoost로 가격 예측 모델을 학습하고, 전처리 산출물과 route 통계 정보를 함께 묶어 추론에 필요한 모델 패키지로 정리했습니다.

**Test**  
노선별 시간 흐름을 유지해 학습 데이터와 분리된 테스트를 진행했습니다. RMSE와 MAE를 기준으로 성능을 확인했습니다.

**Register**  
배포에 필요한 모델 파일과 전처리 산출물을 각각 정리해 등록했습니다.

**Deploy**  
등록한 모델을 SageMaker Endpoint에 반영하고, Lambda와 연결해 실제 서빙 경로로 이어지도록 구성했습니다.

**Experiment Tracking**  
MLflow에 실행별 파라미터, 메트릭, 아티팩트를 함께 기록해 결과 비교와 재현이 가능하도록 했습니다.

---

### 3. Serving Architecture

입력값 정리와 응답 가공은 Lambda에서, 실제 예측 계산은 SageMaker Endpoint에서 처리하도록 역할을 분리했습니다.

**Serving Flow**

```
1. 사용자 요청(JSON) 수신
2. API Gateway → Lambda
3. Lambda: 입력값 정리 및 후보 날짜 생성
4. SageMaker Endpoint 추론
5. 예측값을 가격 단위로 복원
6. Top 3 추천 결과 및 그래프 응답 반환
```

**Fallback Handling**  
학습에 없던 노선이 들어오면 `route_stats.json`의 global fallback 값을 사용해, 처음 보는 노선에도 응답이 끊기지 않게 했습니다.

**Model Evaluation**
- Metric: RMSE, MAE
- 시간 흐름을 유지한 데이터 분할로 실제 예측 상황과 유사한 조건에서 성능 확인
