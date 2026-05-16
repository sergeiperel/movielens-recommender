# 🎬 MovieLens Recommender System

> End-to-end movie recommendation system based on collaborative filtering (ALS / SVD) with an interactive Streamlit interface and experiment tracking via MLflow.

---

## 📌 Project Overview

Movie recommender system based on matrix factorization (ALS / SVD) trained on [MovieLens dataset](https://grouplens.org/datasets/movielens/). Provides personalized recommendations based on user-selected preferences.


## 🚀 Live Demo

The app is hosted on Streamlit Cloud:
👉 **[Try app here](https://movielens-recsys.streamlit.app/)**


## ⚙️ Installation
1. Clone repository

```
git clone https://github.com/your-username/movielens-recommender.git
cd movielens-recommender
```

2. Install dependencies

```
uv pip install .
```

## ▶️ Run locally

```
streamlit run app.py
```

## 🐳 Run with Docker

```
docker-compose up --build
```

Then open:

```
http://localhost:8501
```

## 🧠 Features

- Collaborative filtering (ALS / SVD)
- Interactive Streamlit UI
- Genre filtering
- Pretrained embeddings inference
- MLflow experiment tracking (optional)


## 📊 Metrics

- Recall@10
- Precision@10
- MAP@10
- NDCG@10