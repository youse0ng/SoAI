# 🩻 X-ray 기반 소아 Segmentation & Image + Meta Captioning 모델
<img width="844" height="461" alt="image" src="https://github.com/user-attachments/assets/b14011d0-bd0e-4a96-9bd4-795fbf39f550" />

---

## 📋 프로젝트 개요
"AI로 읽는 소아 복부 X-ray — Segmentation과 소아의 X-Ray 이미지와 부가 정보를 이해하는 AI Captioning 모델을 구현"

본 프로젝트는 소아의 이름을 입력하면, 소아 복부 X-ray 영상을 대상으로 **Segmentation**을 수행하여 소아 복부의 구조와 이상 소견을 시각화하고,
영상 데이터와 환자 메타데이터(나이, 진단 라벨 등)를 함께 활용하여 **의료 진단서 형태의 문장(Image + Meta Captioning)**을 자동 생성하는 AI 모델을 개발한 프로젝트입니다.  

의료진에게 영상 판독 및 진단에 대해 보조를 지원하여 정확도를 향상시키고, 추후에 진단 보조 Agent 역할을 수행할 수 있는 것을 목표로 합니다.

---

## 📊 데이터 소개
| 구분        | 내용 |
|-------------|------|
| 분석 대상   | 소아 복부 X-ray 이미지 |
| 라벨 종류   | `Pyloric Stenosis`,`Air-Fluid Level`,`Abdominal Distension`,`Constipation`, `Normal` |
| 메타데이터  | 환아 나이, 촬영 정보, 진단 라벨 |
| 출처        | 자체 수집/가공 데이터 |
| 데이터 구조 | 이미지 파일 + 메타데이터 CSV (ImagePath, Point, Class, Age, PatientName, Caption) |

---

## 🛠️ 기술 스택

| 분류       | 기술 | 용도 |
|------------|------|------|
| 💻 언어    | Python | 전체 프로젝트 개발 |
| 데이터 처리 | Pandas, PyTorch | 데이터 전처리 및 데이터 파이프라인 설계 |
| 이미지 처리 | PIL, PyTorch | 영상 로드/전처리 및 텐서화 |
| 모델 학습  | PyTorch | Segmentation, Cross-Attention 기반 Captioning |
| 딥러닝 구조 | ViT, BERT, Cross-Attention, Text Generator Decoder | Vision-Language Feature 추출 및 두 Feature 연관성 Integrated Feature 추출 및 Generate Text |
| 시각화     | Matplotlib, Streamlit | 학습에 중요한 벡터 시각화, 예측 결과 시각화 |
| 환경       | CUDA, Jupyter Notebook, .Py | GPU 학습 (3060ti) 환경 및 16GB 메모리 필요 그 이하는 OOM(Out Of Memory) 위험 |

---

## 📝 문제 정의
❓ **복부 X-ray 영상과 환자 메타데이터를 기반으로,  
1) 병변 위치를 Segmentation 하고  
2) 진단 지원 문장을 자동 생성하는 것**

---

## 📊 분석·모델링 방법론

1️⃣ **데이터 전처리**
- 원자료로부터 필요한 데이터 파싱 (Point, Image, 메타데이터)
- 이미지 정규화
- 메타데이터 파싱
- 데이터 파이프라인 구현
- 데이터셋 Train/Test 분할

2️⃣ **Segmentation 모델**
- 의료 영역에 맞춘 커스텀 데이터셋(Point, Image) 생성
- DeepLabv3 모델 (PyTorch) 모델 재구성 및 파인 튜닝 진행
- 성능 지표: Pixel Accuracy: 97%
<img width="796" height="455" alt="image" src="https://github.com/user-attachments/assets/61eecc35-8576-405c-b0ee-74236335c6c7" />


3️⃣ **Image + Meta Feature 추출**
- Vision Transformer(ViT)로 이미지 특징 추출
- BERT tokenizer로 메타데이터 임베딩
- Cross-Attention으로 통합 Feature 생성
- 학습 전/후 통합 Feature 벡터 시각화 (Y-Target Levels 별로 통합 Feature 벡터가 잘 분류되었는지 파악하기 위함)
<img width="1130" height="642" alt="image" src="https://github.com/user-attachments/assets/a3c4944a-fc49-430a-bcca-6d2af607aa9a" />


4️⃣ **Captioning 디코더**
- Transformer 기반 Decoder 설계
- 학습: Teacher Forcing + Cross Entropy Loss
- Inference: Greedy Search 채택
<img width="958" height="561" alt="image" src="https://github.com/user-attachments/assets/410d077f-3544-4b2a-8608-9d8de5aed7f8" />

---

## 🔍 주요 결과

### ✅ Segmentation
- Pixel Accuracy: **0.97**
- 병변 영역 정확한 분리 가능

### ✅ Image + Meta Captioning
- 입력: 복부 X-ray + (환자 나이 + 진단 라벨)
- 출력 예시:
`This plain abdominal supine radiograph shows prominent gastric distension with features suggestive of pyloric stenosis.`

---

## 💡 결론 및 제언
- 영상과 메타데이터를 결합하면 Captioning 정확도가 단일 영상 기반 대비 약 **+8%** 향상
- Segmentation 결과를 Captioning 입력에 포함시키면 의료 보고서의 임상적 타당성이 향상
- 차후 실제 임상 데이터에서 테스트 필요

---

## 📈 시각 자료
- Segmentation 예시
<img width="1773" height="773" alt="image" src="https://github.com/user-attachments/assets/61a0bbb7-7ba8-4681-9bd5-1064c1d3204e" />

- Captioning 예측 결과 샘플
<img width="1777" height="486" alt="image" src="https://github.com/user-attachments/assets/e11dd23c-5d34-4185-9650-27312abea501" />



---

## 📝 프로젝트 배운 점
- 의료 이미지와 비정형 텍스트 데이터의 융합 처리 경험
- Vision-Language 모델링과 Cross-Attention 구조 이해
- Segmentation + Captioning 파이프라인 설계 방법 습득
- ONNX 모델 활용 및 PyTorch inference 최적화 경험: 그러나 뽑아내고 사용하지는 않음 (정상적 작동 CPU: 2초 이내 Inference 가능)
---

## 👨‍💻 참여자
황유성


