import os
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image
import torch

from core.recommender_core import (
    load_and_align_databases,
    check_db_shapes,
    IntegratedArtworkAnalyzer,
    recommend_artworks,
    resolve_image_path,
    USED_SENTIMENT_QUERIES,
    DATASET_ROOT,
)

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="Artwork Recommendation",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 스타일
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --main-blue: #2563ff;
        --soft-blue: #eef4ff;
        --text-dark: #111827;
        --text-gray: #6b7280;
        --line: #e5e7eb;
        --card-bg: #ffffff;
        --bg: #f3f6fb;
    }

    .stApp {
        background: var(--bg);
    }

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    .pill {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 999px;
        background: var(--soft-blue);
        color: var(--main-blue);
        font-weight: 800;
        font-size: 0.9rem;
        margin-bottom: 14px;
    }

    .card {
        background: var(--card-bg);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
        margin-bottom: 18px;
    }

    .upload-focus {
        border: 2px dashed #cbd5e1;
        border-radius: 18px;
        padding: 22px;
        text-align: center;
        background: #fbfdff;
        color: var(--text-gray);
        margin-bottom: 12px;
    }

    .score-text {
        color: var(--main-blue);
        font-weight: 900;
        text-align: center;
        margin-top: 4px;
        margin-bottom: 8px;
    }

    .small-muted {
        color: var(--text-gray);
        font-size: 0.85rem;
    }

    .metric-row {
        display: flex;
        justify-content: space-between;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .metric-name {
        color: var(--text-dark);
    }

    .metric-value {
        color: var(--main-blue);
    }

    .bar-bg {
        width: 100%;
        height: 9px;
        border-radius: 999px;
        background: #e9edf3;
        overflow: hidden;
        margin-bottom: 14px;
    }

    .bar-fill {
        height: 9px;
        border-radius: 999px;
        background: var(--main-blue);
    }

    .reason-box {
        border-radius: 14px;
        padding: 16px;
        background: #fff1f4;
        border-left: 4px solid #ff4d73;
        margin-bottom: 12px;
    }

    .target-box {
        border-radius: 14px;
        padding: 16px;
        background: #effaff;
        border-left: 4px solid #38bdf8;
        margin-bottom: 12px;
    }

    div[data-testid="stFileUploader"] {
        padding: 12px;
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        background: #fbfdff;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid #dbe4f0;
        background: #ffffff;
        color: #2563ff;
        font-weight: 800;
    }

    div.stButton > button:hover {
        border-color: #2563ff;
        color: #2563ff;
    }

    .block-container {
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 경로 설정
# ============================================================
APP_DIR = Path(__file__).resolve().parent

DEFAULT_VISUAL_DB_PATH = str(APP_DIR / "artwork_db_total.pt")
DEFAULT_SENTIMENT_DB_PATH = str(APP_DIR / "artwork_sentiment_db_total.pt")
DEFAULT_DATASET_ROOT = str(APP_DIR / "wikiart_images")

# ============================================================
# 캐시: DB와 모델은 한 번만 로드
# ============================================================
@st.cache_resource(show_spinner="DB를 불러오는 중입니다...")
def load_db(visual_db_path: str, sentiment_db_path: str):
    aligned_db = load_and_align_databases(
        visual_db_path=visual_db_path,
        sentiment_db_path=sentiment_db_path,
    )
    return aligned_db


@st.cache_resource(show_spinner="CNN / ViT / CLIP 모델을 불러오는 중입니다...")
def load_analyzer(vit_dim: int):
    analyzer = IntegratedArtworkAnalyzer(expected_vit_dim=vit_dim)
    return analyzer


# ============================================================
# 유틸 함수
# ============================================================
def save_uploaded_image(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in [".jpg", ".jpeg", ".png", ".webp"]:
        suffix = ".png"

    temp_dir = Path(tempfile.gettempdir()) / "art_recommender_uploads"
    temp_dir.mkdir(exist_ok=True)

    save_path = temp_dir / f"user_uploaded{suffix}"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(save_path)


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def draw_bar(name: str, value: float):
    value = max(0.0, min(float(value), 1.0))
    st.markdown(
        f"""
        <div class="metric-row">
            <span class="metric-name">{name}</span>
            <span class="metric-value">{value * 100:.1f}%</span>
        </div>
        <div class="bar-bg">
            <div class="bar-fill" style="width:{value * 100:.1f}%"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def top_sentiments_from_vector(vec, top_n=3):
    if torch.is_tensor(vec):
        arr = vec.detach().cpu()
    else:
        arr = torch.tensor(vec)

    k = min(top_n, len(USED_SENTIMENT_QUERIES), arr.numel())
    values, indices = torch.topk(arr.float(), k)

    rows = []
    for v, idx in zip(values.tolist(), indices.tolist()):
        rows.append((USED_SENTIMENT_QUERIES[idx], float(v)))
    return rows


def image_name_from_path(path: str):
    return Path(str(path).replace("\\", "/")).stem.replace("_", " ")


def get_image_path(result, dataset_root):
    return resolve_image_path(result["path"], dataset_root=dataset_root)


def recommendation_summary(result):
    cnn = result["cnn_score"]
    vit = result["vit_score"]
    clip = result["sentiment_score"]
    nov = result["novelty_score"]

    parts = []
    if cnn >= vit:
        parts.append("색감·질감 중심의 시각적 특징이 강하게 맞았습니다.")
    else:
        parts.append("구도·형태 중심의 시각적 구조가 유사하게 나타났습니다.")

    if clip >= 0.7:
        parts.append("업로드 이미지와 감성 라벨 분포도 높게 겹칩니다.")

    if nov >= 0.35:
        parts.append("완전히 동일한 느낌보다는 새로운 후보를 섞는 참신성 보정이 반영되었습니다.")

    return " ".join(parts)


# ============================================================
# 사이드바: 업로드와 파라미터
# ============================================================
with st.sidebar:
    st.markdown('<span class="pill">&lt; 이미지 업로드 &gt;</span>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "추천을 받을 이미지를 업로드하세요",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="upload-focus">
                이미지를 업로드하면<br>
                CNN·ViT·CLIP 분석 후<br>
                추천 결과가 생성됩니다.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.image(uploaded_file, caption="업로드 이미지", use_container_width=True)

    st.markdown("---")
    st.markdown('<span class="pill">&lt; 추천 파라미터 조정 &gt;</span>', unsafe_allow_html=True)

    alpha = st.slider(
        "CNN(화풍/색감) vs ViT(구도/형태)",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        help="값이 높을수록 CNN 비중이 커지고, 낮을수록 ViT 비중이 커집니다.",
    )
    st.caption(f"CNN : ViT = {alpha} : {100 - alpha}")

    beta = st.slider(
        "시각적 유사도 vs 감성적 유사도",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
        help="값이 높을수록 시각 유사도를 더 많이 반영합니다.",
    )
    st.caption(f"시각 : 감성 = {beta} : {100 - beta}")

    size_filter = st.selectbox(
        "작품 크기 필터링",
        ["전체 크기", "작은 작품", "중간 작품", "큰 작품"],
    )

    price_cols = st.columns(2)
    with price_cols[0]:
        min_price = st.number_input("최소 가격", min_value=0, value=0, step=10000)
    with price_cols[1]:
        max_price = st.number_input("최대 가격", min_value=0, value=0, step=10000)

    novelty = st.slider(
        "작품 신선도 가중치",
        min_value=0,
        max_value=100,
        value=30,
        step=5,
        help="값이 높을수록 시각적으로 너무 비슷한 작품보다 감성은 비슷하지만 새로운 작품을 조금 더 반영합니다.",
    )
    novelty_weight = novelty / 100

    top_k = st.slider("추천 개수", min_value=5, max_value=30, value=20, step=5)

    st.markdown("---")
    with st.expander("DB 경로 설정"):
        visual_db_path = st.text_input("이미지 유사도 DB", DEFAULT_VISUAL_DB_PATH)
        sentiment_db_path = st.text_input("감성 유사도 DB", DEFAULT_SENTIMENT_DB_PATH)
        dataset_root = st.text_input("작품 이미지 폴더", DEFAULT_DATASET_ROOT)

    run_button = st.button("추천 실행", type="primary")

# ============================================================
# 메인 화면
# ============================================================
st.markdown("## 🎨 이미지 업로드 기반 작품 추천")
st.caption("업로드 이미지를 기준으로 CNN·ViT 시각 유사도, CLIP 감성 유사도, 참신성 가중치를 결합해 추천합니다.")

if uploaded_file is None:
    left, right = st.columns([0.95, 1.05])
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="pill">&lt; 업로드 대기 &gt;</span>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="upload-focus">
                왼쪽 사이드바에서 JPG / PNG / WEBP 이미지를 업로드하세요.<br><br>
                업로드 후 <b>추천 실행</b>을 누르면 추천 작품 20개가 카드 형태로 표시됩니다.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="pill">&lt; 화면 구성 &gt;</span>', unsafe_allow_html=True)
        st.write(
            """
            이 화면은 업로드를 중심으로 설계되어 있습니다.  
            사용자가 이미지를 넣으면 추천 파라미터를 조정하고, 결과 카드에서 작품을 선택해 상세 분석을 확인하는 흐름입니다.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

if not run_button and "results" not in st.session_state:
    st.info("이미지가 업로드되었습니다. 왼쪽에서 파라미터를 조정한 뒤 **추천 실행**을 눌러주세요.")
    st.stop()

# 추천 실행
if run_button:
    try:
        target_path = save_uploaded_image(uploaded_file)

        aligned_db = load_db(visual_db_path, sentiment_db_path)
        analyzer = load_analyzer(int(aligned_db["vit"].shape[1]))

        with st.spinner("업로드 이미지를 분석하고 추천 결과를 계산하는 중입니다..."):
            results = recommend_artworks(
                target_image_path=target_path,
                analyzer=analyzer,
                aligned_db=aligned_db,
                top_k=top_k,
                cnn_in_visual_weight=alpha / 100,
                vit_in_visual_weight=(100 - alpha) / 100,
                visual_weight=beta / 100,
                sentiment_weight=(100 - beta) / 100,
                novelty_weight=novelty_weight,
                novelty_mode="stable",
                similarity_norm="minmax",
                exclude_same_path=True,
                verbose=False,
            )

            target_sentiment = analyzer.extract_sentiment_probs(target_path)

        st.session_state["results"] = results
        st.session_state["target_path"] = target_path
        st.session_state["target_sentiment"] = target_sentiment
        st.session_state["dataset_root"] = dataset_root
        st.session_state["selected_rank"] = 1
        st.success("추천 결과가 생성되었습니다.")

    except Exception as e:
        st.error("추천 실행 중 오류가 발생했습니다.")
        st.exception(e)
        st.stop()

results = st.session_state["results"]
target_path = st.session_state["target_path"]
target_sentiment = st.session_state["target_sentiment"]
dataset_root = st.session_state.get("dataset_root", DEFAULT_DATASET_ROOT)

# ============================================================
# 결과 그리드
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<span class="pill">&lt; 추천 작품 결과 &gt;</span>', unsafe_allow_html=True)
st.caption("작품 카드를 누르면 아래 상세 분석 영역에 반영됩니다.")

cols_per_row = 5
for row_start in range(0, len(results), cols_per_row):
    cols = st.columns(cols_per_row)
    for col, result in zip(cols, results[row_start: row_start + cols_per_row]):
        with col:
            try:
                img_path = get_image_path(result, dataset_root)
                st.image(Image.open(img_path).convert("RGB"), use_container_width=True)
            except Exception:
                st.warning("이미지 없음")

            st.markdown(
                f'<div class="score-text">유사도 {pct(result["recommend_score"])}</div>',
                unsafe_allow_html=True,
            )
            if st.button(f"{result['rank']}위 선택", key=f"select_{result['rank']}"):
                st.session_state["selected_rank"] = result["rank"]
                st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

selected_rank = st.session_state.get("selected_rank", 1)
selected = next((r for r in results if r["rank"] == selected_rank), results[0])

# ============================================================
# 상세 분석
# ============================================================
detail_left, detail_right = st.columns([0.9, 1.1])

with detail_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="pill">&lt; 업로드 이미지 &gt;</span>', unsafe_allow_html=True)
    st.image(Image.open(target_path).convert("RGB"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="pill">&lt; 선택 작품 &gt;</span>', unsafe_allow_html=True)
    try:
        selected_img_path = get_image_path(selected, dataset_root)
        st.image(Image.open(selected_img_path).convert("RGB"), use_container_width=True)
        st.markdown(f"### {image_name_from_path(selected['path'])}")
    except Exception:
        st.warning("선택한 작품 이미지를 찾을 수 없습니다.")
        st.markdown(f"### {image_name_from_path(selected['path'])}")

    st.caption(f"DB 경로: {selected['path']}")
    st.markdown("</div>", unsafe_allow_html=True)

with detail_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="pill">&lt; 추천 이유 &gt;</span>', unsafe_allow_html=True)

    st.markdown(f"### 종합 추천 점수 {selected['recommend_score'] * 100:.1f}%")
    draw_bar("CNN (COLOR & TEXTURE)", selected["cnn_score"])
    draw_bar("ViT (COMPOSITIONAL STRUCTURE)", selected["vit_score"])
    draw_bar("CLIP (SENTIMENT LABELS)", selected["sentiment_score"])
    draw_bar("VISUAL SCORE", selected["visual_score"])
    draw_bar("NOVELTY", selected["novelty_score"])

    st.markdown("---")
    st.markdown("#### 감성 유사도 분석")

    selected_index = selected["index"]
    try:
        aligned_db = load_db(visual_db_path, sentiment_db_path)
        artwork_sentiment = aligned_db["sentiment"][selected_index]
        artwork_top = top_sentiments_from_vector(artwork_sentiment, top_n=3)
    except Exception:
        artwork_top = []

    user_top = top_sentiments_from_vector(target_sentiment, top_n=3)

    st.markdown('<div class="reason-box">', unsafe_allow_html=True)
    st.markdown("**🎯 추천 작품의 주요 감성 라벨 TOP 3**")
    for i, (label, value) in enumerate(artwork_top, start=1):
        st.write(f'{i}. "{label}" → {value * 100:.1f}%')
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="target-box">', unsafe_allow_html=True)
    st.markdown("**👤 업로드 이미지의 주요 감성 라벨 TOP 3**")
    for i, (label, value) in enumerate(user_top, start=1):
        st.write(f'{i}. "{label}" → {value * 100:.1f}%')
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="pill">&lt; 추천 요약 &gt;</span>', unsafe_allow_html=True)
    st.write(recommendation_summary(selected))
    st.markdown("</div>", unsafe_allow_html=True)
