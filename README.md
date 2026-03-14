# WING IT ✈️

> 항공권 구매 타이밍을 예측하는 ML 서비스

"언제 사야 가장 쌀까?" — 향후 30일 예상 가격 흐름을 분석해 최적의 구매 시점 Top 3를 추천합니다.

---

## Overview

항공권 가격은 구매 시점에 따라 변동 폭이 크지만, 일반 사용자는 지금 사야 할지 기다려야 할지 판단하기 어렵습니다.  
WING IT은 과거 항공권 가격의 시간적 변동 패턴을 학습해 최적의 구매 타이밍을 예측합니다.

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

---

## Links

- 🌐 Demo:
- 📁 Portfolio:
