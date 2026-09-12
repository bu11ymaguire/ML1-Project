# 🇰🇷 Korean Tourism Revenue Prediction

> 관광·경제·기후·검색 트렌드 데이터를 활용한 **월별 외국인 관광수입 예측 프로젝트**
> Machine Learning 1 Team Project

한국 관광산업과 관련된 다양한 월별 통계 데이터를 통합하고, **Random Forest, XGBoost, LightGBM** 회귀 모델을 이용해 관광수입(`Tourism_Revenue_1K_USD`)을 예측한 머신러닝 프로젝트입니다.

단순한 과거 관광수입뿐 아니라 관광객 수, 입국자 수, 환율, 항공편, 기후, 면세점 소비, Google Trends 등 관광 수요와 관련된 여러 외부 변수를 함께 활용했습니다.

---

## 1. Project Overview

관광수입은 단순한 관광객 수뿐 아니라 다음과 같은 다양한 요인의 영향을 받을 수 있습니다.

* 관광객 및 입국자 규모
* 1인당 평균 소비액
* 환율 및 물가
* 항공 교통량
* 기온 및 강수
* 면세점 소비
* 해외에서의 한국 관광 관련 검색 관심도
* COVID-19과 같은 외부 충격

본 프로젝트에서는 이러한 데이터를 하나의 월별 시계열 데이터셋으로 통합하여 다음 과정을 수행했습니다.

```text
Raw Statistics
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Time-based Train / Validation / Test Split
      ↓
Random Forest / XGBoost / LightGBM
      ↓
MAE / RMSE / R² Evaluation
      ↓
Pandemic Included vs. Excluded Comparison
```

### Prediction Target

```text
Tourism_Revenue_1K_USD
```

즉, **월별 관광수입(1,000 USD 단위)**을 회귀 문제로 모델링했습니다.

---

## 2. Dataset

### Period

원본 데이터:

```text
2015.01 ~ 2024.12
120 months
```

12개월 lag feature 생성 후 초기 결측 행을 제거한 실제 모델링 데이터:

```text
2016.01 ~ 2024.12
108 months
```

### Variables

원본 데이터는 총 **31개 컬럼**으로 구성되며, Feature Engineering 이후 target을 포함해 총 **44개 컬럼**을 사용합니다.

| Category          | Variables              |
| ----------------- | ---------------------- |
| 👥 Tourism        | 관광객 수, 입국자 수           |
| 💰 Revenue        | 관광수입, 1인당 평균 지출        |
| ✈️ Aviation       | 인천국제공항 도착 여객기          |
| 🌡️ Climate       | 서울·부산·제주 평균 기온         |
| 🌧️ Weather       | 지역별 평균 강수일수            |
| 📅 Calendar       | 공휴일                    |
| 🔎 Search Trends  | 미국·일본·태국 Google Trends |
| 💱 Economy        | USD/KRW 환율, 소비자물가지수    |
| 🛍️ Consumption   | 외국인 면세점 결제액 및 이용 인원    |
| 🚇 Infrastructure | 지하철 교통 인프라             |
| 🛢️ Commodity     | WTI 원유 가격              |
| 🦠 External Shock | Pandemic indicator     |

### Google Trends

해외 사용자의 한국 관광 관심도를 반영하기 위해 다음과 같은 검색어의 Google Trends 데이터를 포함했습니다.

```text
United States
- Korea
- Seoul
- Seoul Travel
- Korea Travel
- k-pop

Japan
- Korea Travel
- Seoul Tourism
- Busan Travel

Thailand
- Travel Korea
- Seoul Korea
- Go Korea
```

이를 통해 실제 관광 통계뿐 아니라 **해외에서 형성되는 잠재적인 관광 관심도**도 설명 변수로 활용했습니다.

---

## 3. Data Preprocessing

### Datetime Conversion

월 정보를 `DatetimeIndex`로 변환하여 시계열 순서를 유지했습니다.

```python
df['시간(연도.월)'] = pd.to_datetime(
    df['시간(연도.월)'],
    format='%b-%y'
)

df.set_index('시간(연도.월)', inplace=True)
```

### Numeric Conversion

CSV 내 숫자에 포함된 쉼표를 제거하고 숫자형으로 변환했습니다.

```python
df[col] = pd.to_numeric(
    df[col].astype(str).str.replace(',', ''),
    errors='coerce'
)
```

### Missing Values

결측치는 **Linear Interpolation**을 사용해 보간했습니다.

```python
df.interpolate(method='linear', inplace=True)
```

---

## 4. Feature Engineering

원본 통계 변수 외에 시계열 정보를 모델에 반영하기 위한 파생 변수를 생성했습니다.

### Calendar Features

```python
df['year']
df['month']
df['quarter']
df['day_of_week']
df['day_of_year']
df['week_of_year']
```

### Lag Features

관광수입의 과거 값을 이용해 최근 변화와 계절성을 반영했습니다.

```python
Tourism_Revenue_1K_USD_lag1
Tourism_Revenue_1K_USD_lag3
Tourism_Revenue_1K_USD_lag6
Tourism_Revenue_1K_USD_lag12
```

추가적으로 다음 변수의 1개월 lag도 사용했습니다.

```python
Num_Tourists_lag1
Exchange_Rate_USD_KRW_lag1
```

특히 `lag12`는 전년도 같은 달의 관광수입을 포함함으로써 관광산업의 **연간 계절성**을 모델이 활용할 수 있도록 합니다.

### Rolling Features

최근 관광수입의 단기적인 흐름을 표현하기 위해 3개월 및 6개월 이동평균도 생성했습니다.

```python
Tourism_Revenue_1K_USD_rolling_mean3
Tourism_Revenue_1K_USD_rolling_mean6
```

> **Implementation Note**
>
> 현재 Notebook에서는 다음과 같이 rolling mean을 생성합니다.
>
> ```python
> df[target].rolling(window=3).mean()
> ```
>
> 이 방식은 현재 시점의 target을 rolling feature에 포함시키므로, 순수한 미래 예측 관점에서는 **target leakage가 발생할 수 있습니다.**
>
> 엄밀한 forecasting 실험에서는 다음과 같이 먼저 한 시점 이동시키는 방식이 적절합니다.
>
> ```python
> df[f'{target}_rolling_mean3'] = (
>     df[target]
>     .shift(1)
>     .rolling(window=3)
>     .mean()
> )
> ```
>
> 따라서 현재 Repository의 결과는 기존 프로젝트 구현을 재현한 결과이며, leakage-free forecasting을 위해서는 이 부분을 수정한 뒤 전체 실험을 다시 수행할 필요가 있습니다.

---

## 5. Time-based Data Split

시계열 데이터이기 때문에 `shuffle=True` 방식의 무작위 분할 대신 **시간 순서를 유지한 데이터 분할**을 사용했습니다.

### Experiment A — Pandemic Included

```text
2016.01 ────────────────────────────── 2022.12
                     Train
                    84 months

2023.01 ─────────────── 2023.12
          Validation
          12 months

2024.01 ─────────────── 2024.12
             Test
           12 months
```

| Split      | Period          | Samples |
| ---------- | --------------- | ------: |
| Train      | 2016.01–2022.12 |      84 |
| Validation | 2023.01–2023.12 |      12 |
| Test       | 2024.01–2024.12 |      12 |

각 데이터는 target을 제외한 **43개 feature**로 구성했습니다.

---

### Experiment B — Pandemic Excluded

COVID-19 기간의 급격한 구조적 변동이 모델 학습에 미치는 영향을 확인하기 위해 `Pandemic == 1`인 데이터를 제외한 별도 실험을 수행했습니다.

| Split      | Period          | Samples |
| ---------- | --------------- | ------: |
| Train      | ≤ 2022.12       |      62 |
| Validation | 2023.01–2023.12 |      12 |
| Test       | 2024.01–2024.12 |      12 |

이를 통해 단순한 모델 비교뿐 아니라,

> **Pandemic이라는 비정상적인 외부 충격을 학습 데이터에 포함하는 것이 이후 관광수입 예측에 어떤 영향을 주는가?**

를 함께 확인했습니다.

---

## 6. Models

세 가지 Tree-based Ensemble 모델을 비교했습니다.

### 🌲 Random Forest

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
)
```

다수의 Decision Tree를 결합해 분산을 낮추고 안정적인 예측을 수행합니다.

---

### 🚀 XGBoost

```python
XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
    eval_metric='rmse'
)
```

Gradient Boosting 기반으로 이전 트리의 오차를 순차적으로 보완하며 학습했습니다.

Validation Set을 이용한 **Early Stopping**을 적용했습니다.

---

### 💡 LightGBM

```python
LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    reg_alpha=0.1,
    reg_lambda=0.1
)
```

LightGBM 역시 Validation Set을 기준으로 Early Stopping을 적용했습니다.

---

## 7. Evaluation Metrics

모델은 세 가지 회귀 지표를 사용하여 평가했습니다.

### MAE — Mean Absolute Error

```text
평균적으로 예측값이 실제값에서 얼마나 벗어났는가?
```

낮을수록 좋습니다.

### RMSE — Root Mean Squared Error

큰 오차에 더 큰 패널티를 부여합니다.

낮을수록 좋습니다.

### R² — Coefficient of Determination

모델이 실제 데이터의 변동을 어느 정도 설명하는지를 나타냅니다.

높을수록 좋으며 `1.0`에 가까울수록 실제값을 잘 설명합니다.

---

# 8. Experimental Results

## 8.1 Pandemic Included

COVID-19 기간을 포함한 84개월의 데이터를 Train Set으로 사용했습니다.

### Validation — 2023

| Model         |         MAE |        RMSE |       R² |
| ------------- | ----------: | ----------: | -------: |
| Random Forest |     144,162 |     162,266 |     0.33 |
| **XGBoost**   | **105,066** | **134,865** | **0.54** |
| LightGBM      |     137,021 |     160,434 |     0.35 |

2023 Validation Set에서는 **XGBoost가 모든 지표에서 가장 높은 성능**을 보였습니다.

### Test — 2024

| Model             |         MAE |        RMSE |       R² |
| ----------------- | ----------: | ----------: | -------: |
| **Random Forest** | **101,223** | **137,437** | **0.55** |
| XGBoost           |     102,867 |     138,745 |     0.54 |
| LightGBM          |     149,707 |     173,883 |     0.29 |

2024 Test Set에서는 Random Forest와 XGBoost의 성능이 유사했으며, **Random Forest가 R² 0.55로 가장 높은 성능**을 기록했습니다.

---

## 8.2 Pandemic Excluded

Pandemic indicator가 활성화된 기간을 Train Set에서 제거한 뒤 동일한 Validation / Test 기간을 평가했습니다.

### Validation — 2023

| Model         |         MAE |        RMSE |       R² |
| ------------- | ----------: | ----------: | -------: |
| Random Forest |     133,768 |     149,933 |     0.43 |
| **XGBoost**   | **107,538** | **123,872** | **0.61** |
| LightGBM      |     119,054 |     143,110 |     0.48 |

Validation 기준으로는 **XGBoost가 R² 0.61로 전체 실험 중 가장 높은 값**을 기록했습니다.

### Test — 2024

| Model             |         MAE |        RMSE |       R² |
| ----------------- | ----------: | ----------: | -------: |
| **Random Forest** | **103,453** | **134,061** | **0.58** |
| XGBoost           |     122,230 |     160,154 |     0.39 |
| LightGBM          |     126,582 |     150,551 |     0.46 |

2024 Test Set에서는 **Random Forest가 R² 0.58로 가장 높은 성능**을 기록했습니다.

---

## 9. Pandemic Effect Analysis

두 실험의 2024 Test R²를 비교하면 다음과 같습니다.

| Model         | Pandemic Included | Pandemic Excluded | Difference |
| ------------- | ----------------: | ----------------: | ---------: |
| Random Forest |              0.55 |          **0.58** |      +0.03 |
| XGBoost       |          **0.54** |              0.39 |      -0.15 |
| LightGBM      |              0.29 |          **0.46** |      +0.17 |

Pandemic 기간을 제거했을 때 모든 모델이 일관되게 좋아지지는 않았습니다.

* **Random Forest**: 소폭 개선
* **XGBoost**: 성능 감소
* **LightGBM**: 비교적 큰 폭으로 개선

즉, COVID-19이라는 구조적 충격을 제거하는 것이 항상 유리한 것이 아니라 **모델이 비정상적인 시계열 구간을 어떻게 활용하는지에 따라 성능 영향이 달라질 수 있음**을 확인했습니다.

또한 Validation에서 가장 좋은 모델과 Test에서 가장 좋은 모델이 서로 달랐다는 점은 작은 시계열 데이터셋에서 특정 기간의 성능만으로 모델을 선택할 때 주의가 필요함을 보여줍니다.

---

## 10. Key Findings

이번 프로젝트를 통해 다음 내용을 확인했습니다.

1. 관광수입 예측에는 관광객 통계뿐 아니라 **환율·항공·기후·면세점 소비·검색 트렌드 등 다양한 외부 요인**을 함께 활용할 수 있습니다.

2. Tree-based Ensemble 모델 중 Validation에서는 **XGBoost**, 2024 Test에서는 **Random Forest**가 가장 높은 성능을 보였습니다.

3. COVID-19 기간의 제거 여부가 모델별로 서로 다른 영향을 주었습니다.

4. 시계열 데이터에서는 Random Split보다 **시간 순서를 유지한 평가**가 중요합니다.

5. Feature Engineering 과정에서도 Lag / Rolling Feature가 예측 시점에 실제로 이용 가능한 정보만 포함하는지 확인해야 합니다.

---

## 11. Limitations

### Small Dataset

Feature Engineering 이후 모델 학습에 사용 가능한 데이터는 108개월에 불과합니다.

특히 Pandemic 제외 실험에서는 Train Set이 **62 samples × 43 features**로 작기 때문에 복잡한 Tree Ensemble 모델에서 안정적인 일반화 성능을 확보하기 어렵습니다.

### COVID-19 Structural Break

2020~2021년의 관광산업은 일반적인 계절성과 다른 구조적 변화를 보였기 때문에 단순한 하나의 시계열로 처리하기 어렵습니다.

본 프로젝트에서는 이를 확인하기 위해 Pandemic 포함 / 제외 실험을 별도로 수행했습니다.

### Rolling Feature Leakage

현재 Notebook의 rolling mean feature는 현재 시점의 target을 포함합니다.

따라서 현재 측정된 성능을 엄밀한 **out-of-sample forecasting 성능으로 해석해서는 안 되며**, 향후에는 `shift(1)` 이후 rolling statistics를 생성하고 결과를 다시 평가해야 합니다.

### Availability of Exogenous Variables

현재 모델에는 동일 월의 관광객 수, 입국자 수, 평균 지출 등 여러 설명 변수가 포함되어 있습니다.

따라서 실제 미래 시점의 관광수입을 사전에 예측하려면 해당 변수들이 예측 시점에 실제로 이용 가능한지 고려해야 합니다.

엄밀한 forecasting 시스템에서는

```text
Past observed variables
        +
Known future variables
        +
Forecasted exogenous variables
```

만을 사용하는 구조가 필요합니다.

---

## 12. Future Work

향후에는 다음과 같이 개선할 수 있습니다.

* Leakage-free Rolling Feature 재설계
* Walk-forward Validation 적용
* `TimeSeriesSplit`을 활용한 반복 평가
* 국가별 관광객 수요 모델링
* SARIMA / Prophet 등 전통적 시계열 모델과 비교
* LSTM / Temporal Transformer 기반 모델과 비교
* Hyperparameter Optimization
* SHAP 기반 Feature Importance 분석
* COVID-19 전후 Regime을 분리한 모델링
* 관광수입과 동시에 사용할 수 없는 contemporaneous feature 제거
* 외생 변수 자체를 먼저 예측하는 multi-stage forecasting pipeline 구축

---

## 13. Repository Structure

```text
ML1-Project/
│
├── Final.ipynb
│   └── 데이터 전처리, Feature Engineering,
│       모델 학습 및 평가
│
├── koreaTrip.csv
│   └── 월별 관광·경제·기후·검색 트렌드 데이터
│
├── csv_excel_file.xlsx
│   └── 데이터 수집 및 정리 과정에서 사용한 Excel 파일
│
├── MachinLearning_TeamProjectReport.pdf
│   └── 프로젝트 최종 보고서
│
├── machinLearing.ppt.pptx
│   └── 프로젝트 발표 자료
│
└── README.md
```

---

## 14. How to Run

### Google Colab

Notebook은 Google Colab 환경을 기준으로 작성되었습니다.

Repository를 clone하거나 `Final.ipynb`를 Colab에서 연 뒤 필요한 패키지를 설치합니다.

```python
!pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
```

Notebook의 기본 데이터 경로는 다음과 같습니다.

```python
file_path = '/content/drive/MyDrive/koreaTrip.csv'
```

따라서 Google Drive를 사용하는 경우 `koreaTrip.csv`를 해당 경로에 배치하거나, 실행 환경에 맞게 `file_path`를 수정해야 합니다.

---

## Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Scikit-learn` · `XGBoost` · `LightGBM` · `Google Colab`

---

## Project Summary

> **관광·경제·기후·검색 행동 데이터를 결합해 한국의 월별 관광수입을 예측하고, Random Forest·XGBoost·LightGBM의 성능 및 COVID-19 기간 포함 여부에 따른 차이를 비교한 시계열 머신러닝 프로젝트입니다.**
