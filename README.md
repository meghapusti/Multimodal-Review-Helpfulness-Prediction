<div align="center">

# 🧠 Multimodal Review Helpfulness Prediction

**Predicting Amazon review helpfulness using text, images, and metadata — fused into a single ML pipeline.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.44-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.8-FFCC00?style=flat-square)](https://catboost.ai)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21F?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://multimodal-review-prediction.streamlit.app/)

<br/>

> Built as part of the **Computational Data Science** course at **SUTD**

</div>

---

## 🌐 Live Demo

**[→ Try the deployed app on Streamlit Cloud](https://multimodal-review-prediction.streamlit.app/)**

No setup required — select a sample review, explore its extracted features, and get an instant helpfulness prediction.

---

## Overview

When shoppers read Amazon reviews, some reviews feel genuinely useful while others don't land. What makes a review helpful? Is it how the reviewer writes? Whether they included photos? Their rating relative to the product's average?

This project builds an end-to-end ML system that answers exactly that question. Using the **Amazon Reviews 2023 dataset** (All Beauty category), the pipeline ingests raw review text, user-uploaded images, and product metadata — extracts rich multimodal features — and trains a gradient-boosted model to predict a continuous helpfulness score.

The key insight: **text alone isn't enough.** By fusing BERT language embeddings with ResNet/CLIP image embeddings and hand-engineered metadata features, this system substantially outperforms text-only baselines.

---

## Architecture

```
Raw Amazon Data (McAuley-Lab/Amazon-Reviews-2023)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│                  Data Preprocessing                    │
│  • Text cleaning (lemmatization, stopword removal)     │
│  • Image URL extraction & quality scoring             │
│  • Metadata merging (reviews ↔ product metadata)      │
│  • Helpfulness score normalization (log-scaled 0–1)   │
└───────────────────────┬───────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
  ┌──────────┐   ┌──────────────┐  ┌──────────────┐
  │   TEXT   │   │    IMAGE     │  │   METADATA   │
  │          │   │              │  │              │
  │ BERT     │   │ CLIP / ResNet│  │ Rating info  │
  │ 768-dim  │   │ 512-dim      │  │ Verification │
  │ embeddings│  │ embeddings   │  │ Temporal     │
  │ + VADER  │   │ + quality    │  │ Product stats│
  │ sentiment│   │   scoring    │  │              │
  │ + readab.│   │              │  │              │
  └────┬─────┘   └──────┬───────┘  └──────┬───────┘
       │                │                  │
       ▼                ▼                  │
  PCA → 50-dim    PCA → 50-dim             │
       │                │                  │
       └────────────────┴──────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  Feature Matrix       │
            │  ~110-dim vector      │
            │  (50 BERT + 50 img   │
            │   + 10 metadata)      │
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  CatBoost Regressor   │
            │  500 iterations       │
            │  depth=6, lr=0.1     │
            │  Early stopping      │
            └───────────┬───────────┘
                        │
                        ▼
            Helpfulness Score (0–1)
```

---

## Features

### Text Features
| Feature | Method |
|---|---|
| Semantic embeddings | BERT (`bert-base-uncased`) — 768-dim, mean-pooled |
| Sentiment score | VADER (`SentimentIntensityAnalyzer`) compound score |
| Readability | Flesch Reading Ease (`textstat`) |
| Review length | Word count of cleaned text |
| Punctuation count | Period count as proxy for sentence structure |

### Image Features
| Feature | Method |
|---|---|
| Visual embeddings | CLIP (`openai/clip-vit-base-patch32`) — 512-dim |
| Image quality score | Custom metric: Laplacian blur + Sobel sharpness + brightness (weighted composite) |
| Parallel processing | `ThreadPoolExecutor` for concurrent image URL fetching |

### Metadata Features
| Feature | Description |
|---|---|
| `verified_purchase` | Binary flag for verified buyers |
| `average_rating` | Product's mean rating across all reviews |
| `rating_number` | Total number of ratings on the product |
| `rating_deviation` | Reviewer's rating minus product average (engineered feature) |
| `days_since_review` | Temporal feature — age of review |
| `avg_quality_score` | Mean image quality score across all review images |

---

## Project Structure

```
.
├── app.py                      # Streamlit web application (inference + UI)
├── data_clean.py               # Initial data loading & cleaning pipeline
├── data_preprocessing.ipynb    # Feature engineering (BERT, CLIP, image quality)
├── final.ipynb                 # Model training, evaluation & artifact export
├── requirements.txt            # All Python dependencies (pinned versions)
├── sample_reviews.csv          # Pre-processed sample data for demo
├── catboost_model.cbm          # Trained CatBoost model (serialized)
├── bert_pca.pkl                # Fitted PCA transformer for BERT embeddings
├── img_pca.pkl                 # Fitted PCA transformer for image embeddings
└── scaler.pkl                  # Fitted StandardScaler for metadata features
```

---

## Quickstart

### Run the Demo Locally

```bash
git clone https://github.com/meghapusti/Multimodal-Review-Helpfulness-Prediction.git
cd Multimodal-Review-Helpfulness-Prediction
pip install -r requirements.txt
streamlit run app.py
```

The app loads all pre-trained artifacts (`catboost_model.cbm`, `*.pkl`) and runs inference directly — no retraining needed.

### Retrain from Scratch

**1. Load & clean the data**
```bash
python data_clean.py
```
Connects to Hugging Face Datasets, loads `McAuley-Lab/Amazon-Reviews-2023` (All Beauty), handles missing values, normalizes helpfulness scores, and saves train/test CSVs.

**2. Extract multimodal features**

Open and run `data_preprocessing.ipynb`. This notebook:
- Downloads BERT and CLIP model weights from Hugging Face
- Extracts 768-dim BERT embeddings from cleaned review text
- Downloads review images and extracts 512-dim CLIP embeddings
- Computes image quality scores (blur, sharpness, brightness)
- Engineers metadata and temporal features
- Saves the complete feature-rich dataframe

**3. Train the model**

Open and run `final.ipynb`. This notebook:
- Applies PCA (50 components each) to BERT and image embeddings
- Standard-scales all metadata features
- Log-transforms the helpfulness target to handle skew
- Trains a CatBoost regressor with early stopping
- Evaluates on held-out test set (MSE, MAE, R², Pearson r)
- Serializes all model artifacts for deployment

---

## Model Details

### CatBoost Regressor Configuration
```python
CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.1,
    random_seed=42,
    early_stopping_rounds=50,
    task_type='CPU'
)
```

### Target Engineering
Raw `helpful_vote` counts are highly skewed (most reviews have 0–5 votes; a few have hundreds). A log-transform is applied before normalization:

```python
y = log1p(helpful_vote) / log1p(max_votes)   # maps to [0, 1]
```

This prevents the model from being dominated by outlier high-vote reviews.

### Evaluation Metrics
- **MSE** — Mean Squared Error
- **MAE** — Mean Absolute Error
- **R²** — Coefficient of determination
- **Pearson r** — Linear correlation between predictions and ground truth

---

## Tech Stack

| Category | Libraries |
|---|---|
| **Deep Learning** | `torch`, `transformers`, `sentence-transformers` |
| **ML / Gradient Boosting** | `catboost`, `scikit-learn` |
| **NLP** | `nltk` (VADER, lemmatizer, tokenizer), `textstat` |
| **Computer Vision** | `opencv-python`, `scikit-image`, `Pillow` |
| **Data** | `pandas`, `numpy`, `datasets` (HuggingFace) |
| **Dimensionality Reduction** | `sklearn.decomposition.PCA` |
| **Web App** | `streamlit` |
| **Visualization** | `matplotlib`, `seaborn`, `plotly` |
| **Parallelism** | `concurrent.futures.ThreadPoolExecutor` |

---

## Dataset

**[Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)** — McAuley Lab, UCSD

- Category: **All Beauty**
- Filtered to reviews with: at least 1 helpful vote, non-empty text, at least 1 attached image
- Merged with product metadata on `parent_asin`

---

## Acknowledgements

- [McAuley Lab](https://cseweb.ucsd.edu/~jmcauley/) for the Amazon Reviews 2023 dataset
- [Hugging Face](https://huggingface.co) for BERT, CLIP, and the Datasets library
- [CatBoost / Yandex](https://catboost.ai) for the gradient boosting framework
- Built as part of the **Computational Data Science** course at [SUTD](https://www.sutd.edu.sg/)

---

## License

This project is for educational and research purposes.
