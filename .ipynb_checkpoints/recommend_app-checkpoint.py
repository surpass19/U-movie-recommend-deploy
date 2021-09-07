import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path


@st.cache
def load_data():
    movie = pd.read_pickle("data/movie.pickle")
    movie_enc = joblib.load("model/movie_encoder.pkl")
    movie["movie_id"] = movie_enc.transform(movie.movie_id)
    return movie.sort_values("movie_id").set_index("movie_id")


@st.cache
def load_model(model_path):
    model = joblib.load(model_path)
    return model

st.title("Movie Recommendation")

# load data
data_load_state = st.text("Loading Data...")
movie = load_data().copy()
data_load_state.text("Loading Data...Done!")


# select movies
selections = st.multiselect(
    "select movies",
    movie.index.tolist(),
    format_func=lambda x: movie.loc[x, "movie_title"],
)
st.write("selected movie_id", selections)

# select model
model_options = Path("model").glob("svd_*.pkl")

#radio → ラジオボタン
model_path = st.radio("select model", list(model_options))
model = load_model(model_path)

# predict
pref = np.zeros((1, model.components_.shape[1]))
pref[:, selections] = 10
movie["score"] = model.transform(pref).dot(model.components_).ravel()
movie.sort_values("score", ascending=False, inplace=True)
movie = movie.reindex(columns=['movie_title', 'score', 'genre', 'mean_review_point', 'screening_time', 'number_of_revier'])
st.write("# result", movie[~movie.index.isin(selections)])