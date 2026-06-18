# ============================================================
# CNN + ViT + CLIP 감성 + Novelty 제안식 통합 추천 프로그램
# ------------------------------------------------------------
# 특징
# 1) 이미 만들어둔 좌표 파일을 다시 만들지 않고 사용합니다.
# 2) CNN/ViT 내부 가중치와 시각/감성 외부 가중치를 분리합니다.
# 3) 47개 CLIP 감성 중 사용자가 지정한 35개 감성만 사용합니다.
# 4) DB의 ViT 차원이 768이든 1000이든 자동으로 타겟 이미지 ViT 차원을 맞춥니다.
# 5) Novelty = 1 - S_visual 보정항을 안정형 제안식으로 반영합니다.
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import math
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
from transformers import CLIPProcessor, CLIPModel


# -----------------------------
# 1. 기본 경로 설정
# -----------------------------
VISUAL_DB_PATH = "./artwork_db_total.pt"               # CNN/ViT 좌표 파일
SENTIMENT_DB_PATH = "./artwork_sentiment_db_total.pt"  # CLIP 감성 좌표 파일
TARGET_IMAGE_PATH = "./user_upload.jpg"               # 추천받을 이미지
DATASET_ROOT = "./wikiart_images"                     # 이미지 폴더. 결과 이미지 표시용


# -----------------------------
# 2. CLIP 감성 텍스트 쿼리 설정
# -----------------------------
# 원래 CLIP 감성 DB는 아래 47개 쿼리 순서로 만들어졌다고 가정합니다.
# 추천 단계에서는 USED_SENTIMENT_QUERIES에 있는 35개만 선택해서 사용합니다.

ALL_SENTIMENT_QUERIES = [
    "A fluttering excitement like a first love",
    "Evoking warm and gentle comfort",
    "Explosive joy",
    "Awe that fills the heart",
    "A thrilling surge of shivers",
    "Noble and radiant sacrifice",
    "A deep and moving emotional resonance",
    "An aura that feels reverent and radiant",
    "Utterly calm, soothing the mind like a windless stillness",
    "Evoking an untouchable sense of dignity",
    "Overflowing vitality and burning passion",
    "So vivid it feels like it could leap out of the frame",
    "Pure and ideally beautiful in a transparent way",
    "Warm and full of tenderness",
    "A refreshing sense of release that opens up the chest",
    "Dreamlike and surreal, as if in a dream",
    "A fairytale-like fantasy",
    "A faint light in the darkness, evoking a sense of hope",
    "A natural, unembellished scene evoking intimate everyday life",
    "Evoking free and playful innocence",
    "Deep shadows and stillness, evoking quiet solace",
    "A calm and gentle atmosphere, evoking nostalgic intimacy",
    "Carrying an inner gloom",
    "Tense with an unstable composition",
    "Evoking the fragility and futility of life, as if about to crumble",
    "Revealing the harshness of a cold reality",
    "A humorous and satirical depiction with an underlying sense of sorrow",
    "Still and lifeless, as if time has stopped",
    "A fragile precariousness, like thin ice about to break",
    "Cold and sharp lighting, evoking a sense of alienation",
    "A deep sense of longing",
    "An overwhelming and suffocating pressure",
    "Fear as if the heart might stop",
    "A dizzying desire",
    "An empty and barren space, evoking deep solitude",
    "A heart-wrenching sadness",
    "A shock as if everything has come to a halt",
    "A passion that has gone cold",
    "Pain that feels like it pierces the heart",
    "Evoking violent energy",
    "Intense, as if emotions are erupting",
    "Evoking confusion",
    "A refined and balanced composition, evoking elegance",
    "Fragmented and disjointed visual elements, evoking reconstructed memory",
    "A calm and restrained scene, evoking quiet inner tension",
    "An ambiguous and undefined presence, evoking enigma",
    "A fleeting and softened moment, evoking ephemerality",
]

USED_SENTIMENT_QUERIES = [
    "A fluttering excitement like a first love",
    "Evoking warm and gentle comfort",
    "Explosive joy",
    "Awe that fills the heart",
    "A thrilling surge of shivers",
    "Noble and radiant sacrifice",
    "A deep and moving emotional resonance",
    "Utterly calm, soothing the mind like a windless stillness",
    "Evoking an untouchable sense of dignity",
    "Overflowing vitality and burning passion",
    "So vivid it feels like it could leap out of the frame",
    "Pure and ideally beautiful in a transparent way",
    "A refreshing sense of release that opens up the chest",
    "Dreamlike and surreal, as if in a dream",
    "A fairytale-like fantasy",
    "A faint light in the darkness, evoking a sense of hope",
    "A natural, unembellished scene evoking intimate everyday life",
    "Evoking free and playful innocence",
    "Deep shadows and stillness, evoking quiet solace",
    "A calm and gentle atmosphere, evoking nostalgic intimacy",
    "Carrying an inner gloom",
    "Tense with an unstable composition",
    "Evoking the fragility and futility of life, as if about to crumble",
    "Revealing the harshness of a cold reality",
    "A humorous and satirical depiction with an underlying sense of sorrow",
    "Still and lifeless, as if time has stopped",
    "A fragile precariousness, like thin ice about to break",
    "Cold and sharp lighting, evoking a sense of alienation",
    "A deep sense of longing",
    "An overwhelming and suffocating pressure",
    "Fear as if the heart might stop",
    "A dizzying desire",
    "An empty and barren space, evoking deep solitude",
    "A refined and balanced composition, evoking elegance",
    "A calm and restrained scene, evoking quiet inner tension",
]

SELECTED_SENTIMENT_INDICES = [ALL_SENTIMENT_QUERIES.index(q) for q in USED_SENTIMENT_QUERIES]


# -----------------------------
# 3. 공통 유틸 함수
# -----------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _get_first_existing_key(db, candidate_keys, label):
    for key in candidate_keys:
        if key in db:
            return key
    raise KeyError(f"{label} 키를 찾지 못했습니다. 후보={candidate_keys}, 현재 키={list(db.keys())}")


def _find_matrix_tensor_key(db, paths_len, exclude_keys=("paths",), label="tensor"):
    for key, value in db.items():
        if key in exclude_keys:
            continue
        if torch.is_tensor(value) and value.ndim == 2 and value.shape[0] == paths_len:
            return key
    raise KeyError(f"{label}로 사용할 2차원 텐서 키를 찾지 못했습니다. 현재 키={list(db.keys())}")


def path_key(path):
    """
    두 DB의 이미지 경로 표현이 달라도 같은 작품이면 매칭되도록 경로 키를 통일합니다.
    예: C:/.../wikiart_images/genre/file.jpg -> genre/file.jpg
    """
    p = str(path).replace("\\", "/")
    parts = [x for x in p.split("/") if x]

    if "wikiart_images" in parts:
        idx = parts.index("wikiart_images")
        return "/".join(parts[idx + 1:]).lower()

    # wikiart_images가 없으면 뒤쪽 2단계 경로를 우선 사용합니다.
    # 예: genre/file.jpg 형태를 보존하기 위함입니다.
    if len(parts) >= 2:
        return "/".join(parts[-2:]).lower()

    return os.path.basename(p).lower()


def resolve_image_path(raw_path, dataset_root=DATASET_ROOT):
    """DB에 저장된 경로가 현재 컴퓨터에서 바로 안 열릴 때 가능한 경로를 복원합니다."""
    raw = str(raw_path)
    p = raw.replace("\\", "/")
    parts = [x for x in p.split("/") if x]

    candidates = [
        raw,
        p,
        os.path.join(dataset_root, os.path.basename(p)),
        os.path.join(".", "wikiart_images", os.path.basename(p)),
    ]

    if "wikiart_images" in parts:
        idx = parts.index("wikiart_images")
        rel_after_root = os.path.join(*parts[idx + 1:]) if len(parts) > idx + 1 else ""
        candidates.append(os.path.join(dataset_root, rel_after_root))
        candidates.append(os.path.join(".", "wikiart_images", rel_after_root))
    elif len(parts) >= 2:
        candidates.append(os.path.join(dataset_root, parts[-2], parts[-1]))
        candidates.append(os.path.join(".", "wikiart_images", parts[-2], parts[-1]))

    seen = set()
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        if os.path.exists(c):
            return c
    return raw


def min_max_norm(tensor):
    """추천 후보 전체 안에서 0~1 범위로 정규화합니다."""
    tensor = tensor.float()
    t_min = tensor.min()
    t_max = tensor.max()
    return (tensor - t_min) / (t_max - t_min + 1e-8)


def cosine_to_01(tensor):
    """코사인 유사도 -1~1을 0~1 범위로 변환합니다."""
    return ((tensor.float() + 1.0) / 2.0).clamp(0.0, 1.0)


def select_sentiment_dimensions(sentiment_tensor, selected_indices=SELECTED_SENTIMENT_INDICES):
    """
    감성 좌표에서 사용할 차원만 선택합니다.

    - 47차원 DB: 선택한 35개 인덱스만 추출
    - 이미 35차원으로 만들어진 DB: 그대로 사용
    """
    sentiment_tensor = sentiment_tensor.float()
    if sentiment_tensor.ndim == 1:
        dim = sentiment_tensor.shape[0]
        if dim == len(selected_indices):
            return sentiment_tensor
        if max(selected_indices) < dim:
            return sentiment_tensor[selected_indices]
    elif sentiment_tensor.ndim == 2:
        dim = sentiment_tensor.shape[1]
        if dim == len(selected_indices):
            return sentiment_tensor
        if max(selected_indices) < dim:
            return sentiment_tensor[:, selected_indices]

    raise ValueError(
        f"감성 좌표 차원이 예상과 다릅니다. 현재 shape={tuple(sentiment_tensor.shape)}, "
        f"필요 차원 수={len(selected_indices)} 또는 47차원 이상"
    )


def normalize_two_weights(weight_a, weight_b, name_a="weight_a", name_b="weight_b"):
    """두 가중치를 합이 1이 되도록 정규화합니다."""
    weight_a = float(weight_a)
    weight_b = float(weight_b)

    if weight_a < 0 or weight_b < 0:
        raise ValueError(f"{name_a}, {name_b}는 음수가 될 수 없습니다.")

    total = weight_a + weight_b
    if total <= 0:
        raise ValueError(f"{name_a}, {name_b}의 합은 0보다 커야 합니다.")

    return weight_a / total, weight_b / total


def check_db_shapes(aligned_db):
    print("\n=== DB shape 확인 ===")
    print(f"CNN       : {tuple(aligned_db['cnn'].shape)}")
    print(f"ViT       : {tuple(aligned_db['vit'].shape)}")
    print(f"Sentiment : {tuple(aligned_db['sentiment'].shape)}")
    print(f"Paths     : {len(aligned_db['paths']):,}")


# -----------------------------
# 4. 두 좌표 DB 경로 기준 정렬
# -----------------------------
def load_and_align_databases(
    visual_db_path=VISUAL_DB_PATH,
    sentiment_db_path=SENTIMENT_DB_PATH,
    selected_sentiment_indices=SELECTED_SENTIMENT_INDICES,
):
    visual_db = load_pt(visual_db_path)
    sentiment_db = load_pt(sentiment_db_path)

    visual_paths_key = _get_first_existing_key(visual_db, ["paths", "image_paths", "valid_paths"], "시각 DB paths")
    sentiment_paths_key = _get_first_existing_key(sentiment_db, ["paths", "image_paths", "valid_paths"], "감성 DB paths")

    cnn_key = _get_first_existing_key(visual_db, ["cnn", "cnn_features", "resnet", "resnet_features"], "CNN 좌표")
    vit_key = _get_first_existing_key(visual_db, ["vit", "vit_features", "vision_transformer"], "ViT 좌표")

    if "probs" in sentiment_db:
        sentiment_key = "probs"
    elif "sentiment" in sentiment_db:
        sentiment_key = "sentiment"
    elif "clip" in sentiment_db:
        sentiment_key = "clip"
    else:
        sentiment_key = _find_matrix_tensor_key(
            sentiment_db,
            paths_len=len(sentiment_db[sentiment_paths_key]),
            label="감성 좌표",
        )

    visual_paths = visual_db[visual_paths_key]
    sentiment_paths = sentiment_db[sentiment_paths_key]

    visual_map = {}
    for i, p in enumerate(visual_paths):
        visual_map[path_key(p)] = i

    sentiment_map = {}
    for i, p in enumerate(sentiment_paths):
        sentiment_map[path_key(p)] = i

    common_keys = sorted(set(visual_map.keys()) & set(sentiment_map.keys()))
    if not common_keys:
        raise ValueError(
            "두 DB에서 공통 이미지 경로를 찾지 못했습니다. "
            "두 좌표 파일이 같은 이미지 폴더에서 만들어졌는지 확인하세요."
        )

    visual_indices = [visual_map[k] for k in common_keys]
    sentiment_indices = [sentiment_map[k] for k in common_keys]

    sentiment_all = sentiment_db[sentiment_key][sentiment_indices].float()
    sentiment_selected = select_sentiment_dimensions(sentiment_all, selected_sentiment_indices)

    aligned_db = {
        "cnn": visual_db[cnn_key][visual_indices].float(),
        "vit": visual_db[vit_key][visual_indices].float(),
        "sentiment": sentiment_selected,
        "paths": [visual_paths[i] for i in visual_indices],
        "keys": common_keys,
        "used_sentiment_queries": USED_SENTIMENT_QUERIES,
        "selected_sentiment_indices": selected_sentiment_indices,
        "source_keys": {
            "visual_paths": visual_paths_key,
            "sentiment_paths": sentiment_paths_key,
            "cnn": cnn_key,
            "vit": vit_key,
            "sentiment": sentiment_key,
        },
    }

    print("✅ DB 로드 및 정렬 완료")
    print(f" - CNN/ViT DB 이미지 수: {len(visual_paths):,}")
    print(f" - CLIP 감성 DB 이미지 수: {len(sentiment_paths):,}")
    print(f" - 최종 공통 이미지 수: {len(aligned_db['paths']):,}")
    print(f" - 사용한 DB 키: {aligned_db['source_keys']}")
    print(f" - CNN shape: {tuple(aligned_db['cnn'].shape)}")
    print(f" - ViT shape: {tuple(aligned_db['vit'].shape)}")
    print(f" - 원본 Sentiment shape: {tuple(sentiment_all.shape)}")
    print(f" - 사용 Sentiment shape: {tuple(aligned_db['sentiment'].shape)}")

    return aligned_db


# -----------------------------
# 5. 타겟 이미지 좌표 추출 모델
# -----------------------------
class IntegratedArtworkAnalyzer:
    def __init__(
        self,
        all_sentiment_queries=ALL_SENTIMENT_QUERIES,
        selected_sentiment_indices=SELECTED_SENTIMENT_INDICES,
        expected_vit_dim=None,
    ):
        """
        expected_vit_dim은 생략해도 됩니다.
        recommend_artworks() 함수가 aligned_db['vit'].shape[1]을 보고 자동으로 ViT 출력 차원을 맞춥니다.

        - 768  : ViT의 분류 head를 제거한 특징 벡터
        - 1000 : ViT의 ImageNet 분류 logits
        """
        self.device = get_device()
        self.all_sentiment_queries = all_sentiment_queries
        self.selected_sentiment_indices = selected_sentiment_indices
        self.vit_model = None
        self.vit_output_dim = None

        print(f"⚡ 현재 사용 중인 연산 장치: {self.device}")

        print("🚀 ResNet50 로드 중...")
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn_model = nn.Sequential(*list(resnet.children())[:-1]).to(self.device).eval()

        self.visual_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print("🚀 CLIP 로드 중...")
        clip_name = "openai/clip-vit-base-patch32"
        self.clip_processor = CLIPProcessor.from_pretrained(clip_name)
        self.clip_model = CLIPModel.from_pretrained(clip_name).to(self.device).eval()

        if expected_vit_dim is not None:
            self.ensure_vit_dim(expected_vit_dim)

    def ensure_vit_dim(self, expected_vit_dim):
        """DB의 ViT 차원에 맞게 ViT 모델 출력 방식을 자동 설정합니다."""
        expected_vit_dim = int(expected_vit_dim)
        if expected_vit_dim not in (768, 1000):
            raise ValueError(
                f"지원하지 않는 ViT DB 차원입니다: {expected_vit_dim}. "
                "현재 자동 대응은 768 또는 1000차원만 지원합니다."
            )

        if self.vit_model is not None and self.vit_output_dim == expected_vit_dim:
            return

        if self.vit_model is not None:
            del self.vit_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("🚀 ViT-B/16 로드 중...")
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

        if expected_vit_dim == 768:
            print("✅ ViT 출력 모드: 768차원 특징 벡터(heads 제거)")
            vit.heads = nn.Identity()
        else:
            print("✅ ViT 출력 모드: 1000차원 ImageNet logits(heads 유지)")

        self.vit_model = vit.to(self.device).eval()
        self.vit_output_dim = expected_vit_dim

    def extract_visual_features(self, image_path, expected_vit_dim=None):
        if expected_vit_dim is not None:
            self.ensure_vit_dim(expected_vit_dim)
        elif self.vit_model is None:
            raise RuntimeError("ViT 모델이 아직 설정되지 않았습니다. expected_vit_dim을 지정하세요.")

        image = Image.open(image_path).convert("RGB")
        x = self.visual_preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            cnn_vec = self.cnn_model(x).flatten().cpu()
            vit_vec = self.vit_model(x).flatten().cpu()

        return cnn_vec.float(), vit_vec.float()

    def extract_sentiment_probs(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(
            text=self.all_sentiment_queries,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            probs_47 = F.softmax(outputs.logits_per_image, dim=1).flatten().cpu()

        probs_selected = select_sentiment_dimensions(probs_47, self.selected_sentiment_indices)
        return probs_selected.float()

    def extract_all_features(self, image_path, expected_vit_dim=None):
        cnn_vec, vit_vec = self.extract_visual_features(image_path, expected_vit_dim=expected_vit_dim)
        sent_vec = self.extract_sentiment_probs(image_path)
        return {
            "cnn": cnn_vec,
            "vit": vit_vec,
            "sentiment": sent_vec,
        }


# -----------------------------
# 6. 2단계 가중치 + Novelty 제안식 기반 추천 함수
# -----------------------------
def recommend_artworks(
    target_image_path,
    analyzer,
    aligned_db,
    top_k=10,
    # 1단계: CNN과 ViT를 섞어 시각적 유사도를 만드는 내부 가중치
    cnn_in_visual_weight=0.50,
    vit_in_visual_weight=0.50,
    # 2단계: 시각적 유사도와 감성적 유사도를 섞는 외부 가중치
    visual_weight=0.70,
    sentiment_weight=0.30,
    # 3단계: 참신성 보정
    novelty_weight=0.10,
    novelty_mode="stable",  # "stable" 권장. 비교용으로 "additive"도 지원
    similarity_norm="minmax",  # "minmax" 또는 "cosine01"
    exclude_same_path=True,
    verbose=True,
):
    """
    추천 점수 계산 방식

    1) 시각적 유사도
       S_visual = a * S_cnn + (1 - a) * S_vit

    2) 기본 최종 유사도
       S_final = b * S_visual + (1 - b) * S_clip

    3) 참신성
       Novelty = 1 - S_visual

    4) 최종 추천 점수
       additive 모드: S_recommend = S_final + λ * Novelty * S_clip
       stable 모드  : S_recommend = (1 - λ) * S_final + λ * Novelty * S_clip

    이 파일의 기본값은 제안식인 novelty_mode="stable"입니다.
    발표 자료의 덧셈식과 비교하고 싶으면 novelty_mode="additive"를 사용하세요.
    """
    if not os.path.exists(target_image_path):
        raise FileNotFoundError(f"타겟 이미지를 찾을 수 없습니다: {target_image_path}")

    novelty_weight = float(novelty_weight)
    if novelty_weight < 0:
        raise ValueError("novelty_weight는 음수가 될 수 없습니다.")
    if novelty_mode not in ["additive", "stable"]:
        raise ValueError("novelty_mode는 'additive' 또는 'stable'이어야 합니다.")
    if similarity_norm not in ["minmax", "cosine01"]:
        raise ValueError("similarity_norm은 'minmax' 또는 'cosine01'이어야 합니다.")

    cnn_ratio, vit_ratio = normalize_two_weights(
        cnn_in_visual_weight,
        vit_in_visual_weight,
        "cnn_in_visual_weight",
        "vit_in_visual_weight",
    )
    visual_ratio, sentiment_ratio = normalize_two_weights(
        visual_weight,
        sentiment_weight,
        "visual_weight",
        "sentiment_weight",
    )

    db_cnn = aligned_db["cnn"].float()
    db_vit = aligned_db["vit"].float()
    db_sent = aligned_db["sentiment"].float()

    # 핵심 수정:
    # analyzer를 IntegratedArtworkAnalyzer()처럼 아무 인자 없이 만들어도,
    # 여기서 DB의 ViT 차원을 보고 타겟 이미지 출력 차원을 자동으로 맞춥니다.
    expected_vit_dim = db_vit.shape[1]
    analyzer.ensure_vit_dim(expected_vit_dim)

    if verbose:
        print("\n✅ 적용된 가중치")
        print(f" - 시각 내부: CNN {cnn_ratio:.3f} + ViT {vit_ratio:.3f}")
        print(f" - 기본 결합: 시각 {visual_ratio:.3f} + 감성 {sentiment_ratio:.3f}")
        print(f" - 참신성 보정: lambda={novelty_weight:.3f}, mode={novelty_mode}")
        print(f" - 유사도 정규화: {similarity_norm}")
        print(f" - DB ViT 차원에 맞춘 타겟 ViT 차원: {expected_vit_dim}")

    target = analyzer.extract_all_features(target_image_path, expected_vit_dim=expected_vit_dim)

    if target["cnn"].shape[-1] != db_cnn.shape[-1]:
        raise ValueError(
            f"타겟 CNN 차원과 DB CNN 차원이 다릅니다. "
            f"target={tuple(target['cnn'].shape)}, db={tuple(db_cnn.shape)}"
        )
    if target["vit"].shape[-1] != db_vit.shape[-1]:
        raise ValueError(
            f"타겟 ViT 차원과 DB ViT 차원이 다릅니다. "
            f"target={tuple(target['vit'].shape)}, db={tuple(db_vit.shape)}"
        )
    if target["sentiment"].shape[-1] != db_sent.shape[-1]:
        raise ValueError(
            f"타겟 감성 차원과 DB 감성 차원이 다릅니다. "
            f"target={tuple(target['sentiment'].shape)}, db={tuple(db_sent.shape)}"
        )

    # 코사인 유사도 계산
    cnn_sim = F.cosine_similarity(target["cnn"].unsqueeze(0), db_cnn)
    vit_sim = F.cosine_similarity(target["vit"].unsqueeze(0), db_vit)
    sent_sim = F.cosine_similarity(target["sentiment"].unsqueeze(0), db_sent)

    # 서로 스케일이 다르므로 0~1 정규화 후 결합
    if similarity_norm == "minmax":
        cnn_score = min_max_norm(cnn_sim)
        vit_score = min_max_norm(vit_sim)
        sentiment_score = min_max_norm(sent_sim)
    else:
        cnn_score = cosine_to_01(cnn_sim)
        vit_score = cosine_to_01(vit_sim)
        sentiment_score = cosine_to_01(sent_sim)

    # 1단계: CNN/ViT를 합쳐 시각적 유사도 생성
    visual_score = (cnn_ratio * cnn_score) + (vit_ratio * vit_score)

    # 2단계: 시각적 유사도와 감성적 유사도 결합
    base_final_score = (visual_ratio * visual_score) + (sentiment_ratio * sentiment_score)

    # 3단계: 참신성 보정
    novelty_score = (1.0 - visual_score).clamp(0.0, 1.0)
    novelty_bonus = novelty_score * sentiment_score

    if novelty_mode == "additive":
        recommend_score = base_final_score + (novelty_weight * novelty_bonus)
    else:
        if novelty_weight > 1:
            raise ValueError("stable 모드에서는 novelty_weight를 0~1 사이로 두는 것을 권장합니다.")
        recommend_score = ((1.0 - novelty_weight) * base_final_score) + (novelty_weight * novelty_bonus)

    # 타겟 이미지가 DB 안에 있는 경우 자기 자신 추천 방지
    if exclude_same_path:
        target_key = path_key(target_image_path)
        for i, k in enumerate(aligned_db["keys"]):
            if k == target_key:
                recommend_score[i] = -1.0

    top_k = min(int(top_k), len(aligned_db["paths"]))
    top_scores, top_indices = torch.topk(recommend_score, top_k)

    results = []
    for rank, (idx, score) in enumerate(zip(top_indices.tolist(), top_scores.tolist()), start=1):
        results.append({
            "rank": rank,
            "index": idx,
            "path": aligned_db["paths"][idx],
            "recommend_score": float(score),
            "base_final_score": float(base_final_score[idx]),
            "visual_score": float(visual_score[idx]),
            "sentiment_score": float(sentiment_score[idx]),
            "novelty_score": float(novelty_score[idx]),
            "novelty_bonus": float(novelty_bonus[idx]),
            "cnn_score": float(cnn_score[idx]),
            "vit_score": float(vit_score[idx]),
            "cnn_ratio": float(cnn_ratio),
            "vit_ratio": float(vit_ratio),
            "visual_ratio": float(visual_ratio),
            "sentiment_ratio": float(sentiment_ratio),
            "novelty_weight": float(novelty_weight),
            "novelty_mode": novelty_mode,
        })

    return results


# -----------------------------
# 7. 결과 출력 / 시각화 / 저장
# -----------------------------
def print_results(results):
    print("\n=== 통합 추천 결과 ===")
    for r in results:
        print(
            f"{r['rank']:2d}위 | 추천={r['recommend_score']:.4f} "
            f"| 기본={r['base_final_score']:.4f} "
            f"| 시각={r['visual_score']:.4f} "
            f"| 감성={r['sentiment_score']:.4f} "
            f"| 참신성={r['novelty_score']:.4f} "
            f"| 보정={r['novelty_bonus']:.4f} "
            f"| CNN={r['cnn_score']:.4f} "
            f"| ViT={r['vit_score']:.4f} "
            f"| {r['path']}"
        )


def show_results(target_image_path, results, dataset_root=DATASET_ROOT):
    n = len(results)
    cols = min(5, n + 1)
    rows = int(math.ceil((n + 1) / cols))

    plt.figure(figsize=(4 * cols, 4 * rows))

    plt.subplot(rows, cols, 1)
    plt.imshow(Image.open(target_image_path).convert("RGB"))
    plt.title("Target", fontweight="bold")
    plt.axis("off")

    for i, r in enumerate(results, start=2):
        plt.subplot(rows, cols, i)
        img_path = resolve_image_path(r["path"], dataset_root=dataset_root)
        try:
            plt.imshow(Image.open(img_path).convert("RGB"))
            name = os.path.basename(img_path)
            plt.title(
                f"Rank {r['rank']}\n"
                f"Rec {r['recommend_score']:.3f} / Nov {r['novelty_score']:.3f}\n"
                f"{name[:18]}"
            )
        except Exception:
            plt.text(0.5, 0.5, "Image Not Found", ha="center", va="center")
            plt.title(f"Rank {r['rank']}\n{r['recommend_score']:.3f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def save_results_csv(results, save_path="recommendation_results.csv"):
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"✅ 결과 CSV 저장 완료: {save_path}")
    return df



def print_formula_summary():
    """현재 파일에서 사용하는 추천 점수 수식을 출력합니다."""
    print("=== 추천 점수 수식 ===")
    print("1) S_visual = a * S_cnn + (1 - a) * S_vit")
    print("2) S_final  = b * S_visual + (1 - b) * S_clip")
    print("3) Novelty  = 1 - S_visual")
    print("4) 제안식   = (1 - lambda) * S_final + lambda * Novelty * S_clip")
    print("비교용 덧셈식 = S_final + lambda * Novelty * S_clip")


