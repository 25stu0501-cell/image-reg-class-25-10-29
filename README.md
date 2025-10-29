# image-reg-class-25-10-29
# 🧠 이미지 회귀 (Image Regression) vs 이미지 분류 (Image Classification)

> 머신러닝과 딥러닝에서 **이미지 데이터를 다루는 두 가지 대표적인 문제 유형**입니다.  
> 하나는 **숫자값을 예측(Regression)** 하는 것이고,  
> 다른 하나는 **이미지가 어떤 종류인지 분류(Classification)** 하는 것입니다.

---

## 📊 주요 차이점 비교

| 구분 | **이미지 회귀 (Image Regression)** | **이미지 분류 (Image Classification)** |
|------|--------------------------------------|----------------------------------------|
| **목적** | 이미지를 보고 **연속적인 수치값**을 예측 | 이미지를 보고 **정해진 범주(Class)** 중 하나를 선택 |
| **출력 값** | 실수(예: 나이, 온도, 거리 등) | 클래스 이름 또는 라벨(예: 고양이, 개, 사람 등) |
| **출력 형태** | 단일 숫자 또는 여러 실수 벡터 | 원-핫 벡터(one-hot vector) 또는 확률 분포 |
| **손실 함수 (Loss)** | MSE (Mean Squared Error), MAE 등 | Cross-Entropy Loss |
| **예시 입력** | 사람 얼굴 사진 | 사람 얼굴 사진 |
| **예시 출력** | 얼굴에서 추정한 ‘나이’: 23.5세 | 얼굴의 ‘성별’: 남성 / 여성 |
| **활용 분야** | 얼굴 나이 추정, 거리 측정, 감정 강도 예측, 품질 점수 평가 | 동물 분류, 숫자 인식, 음식 종류 분류, 질병 진단 |
| **출력 레이어 (딥러닝)** | 마지막 레이어: Linear / Dense (활성함수 없음) | 마지막 레이어: Softmax (클래스 확률 계산) |
| **대표 모델 예시** | CNN + Dense(1) | CNN + Dense(# of classes) + Softmax |
| **평가 지표** | RMSE, MAE, R² Score 등 | Accuracy, Precision, Recall, F1 Score |

---

## 🧩 간단한 코드 비교 (TensorFlow 예시)

### 🔹 이미지 회귀 (예: 얼굴 나이 예측)

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # 회귀 → 실수값 1개
])
model.compile(optimizer='adam', loss='mse')
