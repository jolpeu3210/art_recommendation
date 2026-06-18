# 이미지 업로드 기반 작품 추천 Streamlit 앱

이 폴더는 기존 노트북 `통합추천프로그램_참신성_제안식Stable.ipynb`의 추천 로직을 실제 사용자 화면으로 띄우기 위한 Streamlit 앱입니다.

## 폴더 구조

```text
art_recommender_streamlit_app/
├─ app.py
├─ core/
│  ├─ __init__.py
│  └─ recommender_core.py
├─ requirements.txt
└─ README.md
```

## 필요한 파일 배치

아래 파일을 `app.py`와 같은 폴더에 두는 것을 기본으로 가정합니다.

```text
artwork_db_total.pt
artwork_sentiment_db_total.pt
wikiart_images/
```

다른 위치에 있다면 앱 왼쪽의 `DB 경로 설정`에서 직접 경로를 바꾸면 됩니다.

## 실행 방법

```bash
cd art_recommender_streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

## 화면 흐름

1. 사용자가 이미지를 업로드합니다.
2. CNN vs ViT 비율, 시각 vs 감성 비율, 신선도 가중치를 조정합니다.
3. 추천 실행 버튼을 누릅니다.
4. 추천 작품 20개가 카드 형태로 표시됩니다.
5. 작품을 선택하면 업로드 이미지와 선택 작품의 추천 이유, CNN/ViT/CLIP 점수, 감성 라벨 TOP 3가 표시됩니다.

## 현재 코드에서 중요한 점

- 추천 로직은 노트북의 수식과 동일합니다.
- 기본 추천식은 안정형 참신성 수식입니다.

```text
S_visual = a·S_cnn + (1-a)·S_vit
S_final = b·S_visual + (1-b)·S_clip
Novelty = 1 - S_visual
S_recommend = (1-λ)·S_final + λ·Novelty·S_clip
```

## 주의

첫 실행에서는 ResNet50, ViT, CLIP 모델을 불러오기 때문에 시간이 걸릴 수 있습니다.
Streamlit 캐시를 사용하므로 한 번 로드된 뒤에는 같은 세션에서 더 빠르게 동작합니다.
