import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms
from PIL import Image
import streamlit as st
import time
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from segmentation import load_model, preprocess, predict
from dbutils import connect_db, Name_Index
from VISNLP import ViTFeatureExtractor, BertFeatureExtractor, CrossAttention, VISNLPEXTRACTOR, CaptionDecoder, CaptionGenerator, load_Generator, generate_caption
from transformers import AutoTokenizer
from warnings import filterwarnings
filterwarnings('ignore')

def write_stream_caption(text, delay=0.05):
    placeholder = st.empty()
    current_text = ""
    for char in text:
        current_text += char
        # 작고 옅은 텍스트 스타일 (caption 느낌)
        placeholder.markdown(
                    f"<p style='font-size: 20px; color: gray;'>{current_text}</p>",
                    unsafe_allow_html=True)
        time.sleep(delay)

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_WIDTH = 512
IMG_DIM_HEIGHT = 512
PATCH_SIZE = 16
EMBED_DIM = 256
NUM_HEADS = 4
DEPTH = 4
FF_DIM = 2048
NUM_LAYERS = 4
mapping = {'선천성유문협착증':'Pyloric Stenosis','공기액체음영':'air-fluid level',
                        '기복증':'Abdominal distension','변비':'Constipation','정상':'Normal'}

# Load Pretrained Segmetation Model 
segmentation_model = load_model()

# MariaDB와 연동
conn = connect_db()

# Tokenizer 부르기
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load Pretrained Captioning Model
model = CaptionGenerator(
    VISNLPEXTRACTOR(IMG_DIM_HEIGHT, IMG_WIDTH, PATCH_SIZE, EMBED_DIM, NUM_HEADS, DEPTH),
    vocab_size=30522,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    num_layers=NUM_LAYERS
).to(device)

captioning_model = load_Generator(model,'Model_Parameters.pth')
captioning_model.eval()
st.set_page_config(page_title="소아 복부 X-Ray 의료 보조 Agent", layout="wide")
st.title("🩺 소아 복부 X-Ray 의료 보조 Agent")
st.markdown("### 👶 환아의 이름을 입력하세요")

# 이름 입력
name = st.text_input("환아 이름")

# 이름이 입력되면 진행
if name:
    try:
        info = Name_Index(conn, name)
        image_path = info['ImageFile']
        meta_label = mapping[info['ImagePath']]
        # 이미지 로딩 및 전처리
        image = Image.open(image_path)
        image_tensor = preprocess(image)
        # 캡션 생성
        with torch.no_grad():
            caption = generate_caption(captioning_model, image_tensor, meta_label, tokenizer, device=device)
        # 세그멘테이션 수행
        mask = predict(segmentation_model, image_tensor)

        # -------------------------------------
        # Layout - 좌우 분할
        # -------------------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"📸 {info['PatientName']}님의 복부 X-Ray | 나이: {info['Age']}")
            st.image(image, caption=f"{info['PatientName']} (Age: {info['Age']})",  use_container_width=True)

        with col2:
            st.subheader("🧠 Segmentation 결과")
            if isinstance(mask, torch.Tensor):
                mask_np = mask.squeeze().cpu().numpy()
            else:
                mask_np = np.squeeze(mask)
            # 원본 이미지를 numpy로 변환 (RGB)
            image_np = np.array(image.resize((mask_np.shape[1], mask_np.shape[0])))

            # 시각화
            fig, ax = plt.subplots()
            ax.imshow(image_np)  # 원본 이미지
            ax.imshow(mask_np, cmap='jet', alpha=0.5)  # Segmentation Mask overlay
            ax.axis('off')
            st.pyplot(fig)

        st.markdown("---")
        st.subheader("📝 AI 생성형 모델 진단 결과")
        write_stream_caption(caption, delay=0.06)
        
        with st.expander("📂 환자 메타 정보"):
            st.json(info)
    except Exception as e:
        st.error(f"⚠️ 환자 정보를 불러오는 데 실패했습니다: {str(e)}")
else:
    st.warning("좌측 상단에 환자 이름을 입력해주세요.")
