# core pkg
import streamlit as st 
import streamlit.components.v1 as stc

# EDA Pkgs
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn import preprocessing

import scipy.sparse
import warnings
warnings.filterwarnings('ignore')

# Components Pkgs
import streamlit.components.v1 as components

@st.cache
def load_data():
    movie = pd.read_pickle('./data/movie.pickle')
    review0_df = pd.read_pickle('./data_reviews/review0.pickle')
    review1_df = pd.read_pickle('./data_reviews/review1.pickle')
    review2_df = pd.read_pickle('./data_reviews/review2.pickle')
    review3_df = pd.read_pickle('./data_reviews/review3.pickle')
    ratings = review0_df.append(review1_df).append(review2_df).append(review3_df)
    
    movie = movie[movie['number_of_revier'] > 10].reset_index(drop=True)
    movie = movie.dropna(subset=['mean_review_point', 'number_of_revier'])
    ratings.point.replace({0: 10}, inplace=True)
    mergeddf = ratings.merge(movie, left_on = 'movie_id', right_on = 'movie_id', suffixes= ['_user', ''])
    mergeddf = mergeddf[['user_id','movie_id','point']]
    mergeddf = mergeddf.drop_duplicates(['user_id','movie_id'])
    user_enc = LabelEncoder()
    movie_enc = LabelEncoder()
    mergeddf["user_id"] = user_enc.fit_transform(mergeddf.user_id)
    mergeddf["movie_id"] = movie_enc.fit_transform(mergeddf.movie_id)
    movie_pivot = mergeddf.pivot(index= 'movie_id',columns='user_id',values='point').fillna(0)
    movie_pivot_sparse = csr_matrix(movie_pivot.values)
    
    # create review matrix
    n_users = mergeddf.user_id.nunique()
    n_movies = mergeddf.movie_id.nunique()
    matrix = scipy.sparse.csr_matrix(
        (mergeddf.point, (mergeddf.user_id, mergeddf.movie_id)), shape=(n_users, n_movies)
    )
    movie_new = pd.read_pickle("./data/movie_drop_duplicates.pickle")
    return movie_pivot, movie_pivot_sparse, movie_new

@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = joblib.load(model_path)
    return model


def give_rec(title, df, sig, indices):
    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores 
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies 
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Movie indices
    movie_indices = [i[0] for i in sig_scores]
    
    juni_list = list(range(1, len(sig)+1))
    
    df['TFIDF_rank'] = 0
    df['TFIDF_rank'].loc[movie_indices] = juni_list.copy()
    
    return df

def get_recommend_movie_df(Movie_ID, movie_pivot, df, movie_pivot_sparse, model_SVD, model_clusterer, model_knn, tfv):
    
    movie_recommend = df.copy()
    
    #k-meansクラスタリング
    c_preds = model_clusterer.predict(movie_pivot_sparse)
    movie_recommend['cluster'] = c_preds
    
    #主成分分析
    pref = np.zeros((1, model_SVD.components_.shape[1]))
    pref[:, Movie_ID] = 10
    score = model_SVD.transform(pref).dot(model_SVD.components_).ravel()
    movie_recommend["svd_score"] = score
    
    Movie_ID = Movie_ID[0]
    
    #knn score
    distance, indice = model_knn.kneighbors(movie_pivot.iloc[movie_pivot.index == Movie_ID].values.reshape(1,-1),n_neighbors=len(df))
    distance_list = distance.tolist()[0]
    indice_list = indice.tolist()[0]
    
    movie_recommend['knn_distance'] = 0
    movie_recommend['knn_distance'].loc[indice_list] = distance_list.copy()
    movie_recommend = movie_recommend.reindex(columns=['movie_title', 'genre', 'mean_review_point', 'screening_time', 'number_of_revier', 'svd_score', 'knn_distance', 'cluster'])
    
    #TFIDF score
    movie_recommend['genre'] = movie_recommend['genre'].fillna('')
    genres_str = movie_recommend['genre'].str.split('|').astype(str)
    tfv_matrix = tfv.fit_transform(genres_str)
    
    # Compute the sigmoid kernel
    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
    indices = pd.Series(movie_recommend.index, index=movie_recommend['movie_title']).drop_duplicates()
    
    movie_recommend = give_rec(Movie_ID, movie_recommend, sig, indices)
    
    #正規化
    mm = preprocessing.MinMaxScaler()
    movie_recommend['svd_score_mm'] = mm.fit_transform(movie_recommend['svd_score'].values.reshape(-1, 1))
    movie_recommend['knn_distance_mm'] = mm.fit_transform(movie_recommend['knn_distance'].values.reshape(-1, 1))
    movie_recommend['TFIDF_rank_mm'] = mm.fit_transform(movie_recommend['TFIDF_rank'].values.reshape(-1, 1))
    
    movie_recommend['rec_score'] =\
        movie_recommend['svd_score_mm'] - movie_recommend['knn_distance_mm'] - movie_recommend['TFIDF_rank_mm']
    
    return movie_recommend

def get_recommend_movie_df_in_recommend(Movie_ID, movie_pivot, df, movie_pivot_sparse, model_SVD, model_clusterer):
    movie_recommend = df.copy()
    
    #k-meansクラスタリング
    c_preds = model_clusterer.predict(movie_pivot_sparse)
    movie_recommend['cluster'] = c_preds
    
    #主成分分析
    pref = np.zeros((1, model_SVD.components_.shape[1]))
    pref[:, Movie_ID] = 10
    score = model_SVD.transform(pref).dot(model_SVD.components_).ravel()
    movie_recommend["svd_score"] = score
    
    return movie_recommend

def get_recommend_movie_df_in_serch(Movie_ID, movie_pivot, df, movie_pivot_sparse, model_clusterer, model_knn, tfv):
    
    movie_recommend = df.copy()
    
    #k-meansクラスタリング
    c_preds = model_clusterer.predict(movie_pivot_sparse)
    movie_recommend['cluster'] = c_preds
    
    #knn score
    distance, indice = model_knn.kneighbors(movie_pivot.iloc[movie_pivot.index == Movie_ID].values.reshape(1,-1),n_neighbors=len(df))
    distance_list = distance.tolist()[0]
    indice_list = indice.tolist()[0]
    
    movie_recommend['knn_distance'] = 0
    movie_recommend['knn_distance'].loc[indice_list] = distance_list.copy()
    movie_recommend = movie_recommend.reindex(columns=['movie_title', 'genre', 'mean_review_point', 'screening_time', 'number_of_revier', 'knn_distance', 'cluster'])
    
    #TFIDF score
    movie_recommend['genre'] = movie_recommend['genre'].fillna('')
    genres_str = movie_recommend['genre'].str.split('|').astype(str)
    tfv_matrix = tfv.fit_transform(genres_str)
    
    # Compute the sigmoid kernel
    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
    indices = pd.Series(movie_recommend.index, index=movie_recommend['movie_title']).drop_duplicates()
    
    movie_recommend = give_rec(Movie_ID, movie_recommend, sig, indices)
    
    #正規化
    mm = preprocessing.MinMaxScaler()
    movie_recommend['knn_distance_mm'] = mm.fit_transform(movie_recommend['knn_distance'].values.reshape(-1, 1))
    movie_recommend['TFIDF_rank_mm'] = mm.fit_transform(movie_recommend['TFIDF_rank'].values.reshape(-1, 1))
    
    movie_recommend['rec_score'] = movie_recommend['knn_distance_mm'] + movie_recommend['TFIDF_rank_mm']
    
    return movie_recommend

######################
# HTML
######################

HTML_BANNER = """
    <div style="background-color:#464e5f;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;">Movie Recommendations App </h1>
    </div>
    """

footer_temp = """
	 <!-- CSS  -->
	  <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
	  <link href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css" type="text/css" rel="stylesheet" media="screen,projection"/>
	  <link href="static/css/style.css" type="text/css" rel="stylesheet" media="screen,projection"/>
	   <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.5.0/css/all.css" integrity="sha384-B4dIYHKNBt8Bc12p+WXckhzcICo0wtJAoU8YZTY5qE0Id1GSseTk6S+L3BlXeVIU" crossorigin="anonymous">
	 <footer class="page-footer grey darken-4">
	    <div class="container" id="aboutapp">
	      <div class="row">
	        <div class="col l6 s12">
	          <h4 class="white-text">About Streamlit Movie Recommendations App</h4>
	          <p class="grey-text text-lighten-4">Using Streamlit</p>
	        </div>
	      
	   <div class="col l3 s12">
	          <h5 class="white-text">Connect With Me</h5>
	          <ul>
	            <a href="https://twitter.com/DXhvw9" target="_blank" class="white-text">
	            <i class="fab fa-twitter-square fa-4x"></i>
	          </a>
	          <a href="https://gh.linkedin.com/in/ukita-ryosuke-b0a9b11a6" target="_blank" class="white-text">
	            <i class="fab fa-linkedin fa-4x"></i>
	          </a>
	          <a href="" target="_blank" class="white-text">
	            <i class="fab fa-youtube-square fa-4x"></i>
	          </a>
	           <a href="https://github.com/surpass19/" target="_blank" class="white-text">
	            <i class="fab fa-github-square fa-4x"></i>
	          </a>
	          </ul>
	        </div>
	      </div>
	    </div>
	    <div class="footer-copyright">
	      <div class="container">
          Data obtained from: <a class="white-text text-lighten-3" href="https://www.jtnews.jp">みんなのシネマレビュー</a><br/>
	      References: <br>
          <a class="white-text text-lighten-3" href="https://speakerdeck.com/amaotone/making-ml-app-with-scrapy-scikit-learn-and-streamlit">Scrapyとscikit-learn、Streamlitで作るかんたん機械学習アプリケーション</a><br/>
	      <a class="white-text text-lighten-3" href="https://www.codexa.net/collaborative-filtering-k-nearest-neighbor/">機械学習を使って630万件のレビューに基づいたアニメのレコメンド機能を作ってみよう（機械学習 k近傍法 初心者向け）</a><br/>
          <a class="white-text text-lighten-3" href="https://youtube.com/playlist?list=PLJ39kWiJXSixyRMcn3lrbv8xI8ZZoYNZU">Streamlit Python Tutorials</a><br/>
	      </div>
	    </div>
	  </footer>
	"""

######################
# Page Title
######################

# image = Image.open('solubility-logo.jpg')
# st.image(image, use_column_width=True)
st.set_page_config(page_title = 'Movie Recommendations App', page_icon="chart_with_upwards_trend",layout="wide",initial_sidebar_state="expanded",)


#-------------------------------------------------------------------------------------------------------------------------------------#

    
def main():
    """Basics on st.beta columns/layout"""

    menu = ["Recommend","Search","(Recommend)","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    stc.html(HTML_BANNER)
    
    # load data
    #data_load_state = st.text("Loading Data...")
    movie_pivot, movie_pivot_sparse, df = load_data()
    #data_load_state.text("Loading Data...Done!")
    
    #-------------------------------------------------------------------#
        
    if choice == '(Recommend)':
        
        # select movies
        selections = st.multiselect(
            "select movies",
            reversed(df.index.tolist()),
            format_func=lambda x: df.loc[x, "movie_title"],
        )
        #st.write("selected movie_id", selections)
        
        Number = st.sidebar.slider("Number of recommend movies",1,30,10)
        
        # select model(SVD)
        model_options = Path("model_svd").glob("svd_*.pkl")
        model_path = st.sidebar.radio("select model", list(model_options), index=3)
        model_SVD = load_model(model_path)
        
        # select model(clusterer:kmeans)
        model_options = Path("model_kmeans").glob("kmeans_*.pkl")
        model_path = st.sidebar.radio("select model", list(model_options))
        model_clusterer = load_model(model_path)
        
        # select model(clusterer:knn)
        model_options = Path("model_knn").glob("knn_*.pkl")
        model_path = st.sidebar.radio("select model", list(model_options), index=7)
        model_knn = load_model(model_path)
        
        # select model(tfv)
        model_options = Path("model_tfv").glob("tfv.pkl")
        model_path = st.sidebar.radio("select model", list(model_options))
        tfv = load_model(model_path)
        
        
        if st.button("Recommend"):
            data_load_state = st.text("Loading Data...")
            
            movie_recommend = get_recommend_movie_df(selections, movie_pivot, df, movie_pivot_sparse, model_SVD, model_clusterer, model_knn, tfv)
            movie_recommend.sort_values("rec_score", ascending=False, inplace=True)
            
            selected_movie_recommend = movie_recommend[~movie_recommend.index.isin(selections)].head(Number)
            selected_movie_recommend = selected_movie_recommend[['movie_title', 'genre', 'screening_time', 'mean_review_point']]
            
            # Filter
            titles = selected_movie_recommend['movie_title'].values
            genres = selected_movie_recommend['genre'].values
            times = selected_movie_recommend['screening_time'].values
            reviews = selected_movie_recommend['mean_review_point'].values
            
            data_load_state.text("Result:")
            st.write(selected_movie_recommend)
            
            # 風船飛ばす
            st.balloons()
            

            c1,c2 = st.columns([1,2])
            with c1:
                with st.expander("movie_title"):
                    for title in titles.tolist():
                        st.success(title)
                        st.write('********************')

            with c2:
                with st.expander("Genre"):
                    for genre in genres.tolist():
                        st.success('・'.join(str(genre).split('|')))
                        st.write('********************')

    #-------------------------------------------------------------------#            
                
    if choice == 'Search':
        
        Number = st.sidebar.slider("Number of recommend movies",1,30,10)
        
        # select model(clusterer:kmeans)
        model_options = Path("model_kmeans").glob("kmeans_*.pkl")
        model_path = st.sidebar.radio("select model", list(model_options))
        model_clusterer = load_model(model_path)
        
        # select model(clusterer:kmeans)
        model_options = Path("model_knn").glob("knn_*.pkl")
        model_path = st.sidebar.radio("select model", list(model_options), index=7)
        model_knn = load_model(model_path)
        
        # select model(clusterer:kmeans)
        model_options = Path("model_tfv").glob("tfv.pkl")
        model_path = st.sidebar.radio("select model", list(model_options))
        tfv = load_model(model_path)
        
        # select movies
        movie_choice = st.selectbox(
            "Select Movie Title",
            reversed(df.index.tolist()),
            format_func=lambda x: df.loc[x, "movie_title"],
        )
        if st.button("Search"):
            #with st.expander('Movies DF',expanded=False):
            selected_df = df[df.index == movie_choice]
            selected_df = selected_df[['movie_title', 'genre', 'screening_time', 'mean_review_point']]

            # Filter
            title = selected_df['movie_title'].values[0]
            genres = selected_df['genre'].values
            time = selected_df['screening_time'].values[0]
            review = selected_df['mean_review_point'].values[0]

            c1,c2,c3,c4 = st.columns([1.5,1,1,0.8])

            with c1:
                with st.expander("Title"):
                    st.success(title)

            with c2:
                with st.expander("Review"):
                    st.success(str(round(review,1)) + ' / 10.0')

            with c3:
                with st.expander("Genre"):
                    for genre in str(genres.tolist()[0]).split('|'):
                        st.success(genre)
            with c4:
                with st.expander("Time"):
                    st.success(str(int(time)) + '分')
            
            st.write('Similar Movies...')
            
            movie_close = get_recommend_movie_df_in_serch(movie_choice, movie_pivot, df, movie_pivot_sparse, model_clusterer, model_knn, tfv)
            movie_close.sort_values("rec_score", ascending=True, inplace=True)
            movie_close = movie_close[~(movie_close.index == (movie_choice))].head(Number)
            movie_close = movie_close[['movie_title', 'genre', 'screening_time', 'mean_review_point']]

            st.dataframe(movie_close)
            
    #-------------------------------------------------------------------#
        
    if choice == 'Recommend':
        st.subheader("Recommend")
        
        # select movies
        selections = st.multiselect(
            "select movies",
            reversed(df.index.tolist()),
            format_func=lambda x: df.loc[x, "movie_title"],
        )
        #st.write("selected movie_id", selections)
        
        Number = st.sidebar.slider("Number of recommend movies",1,30,10)
        
        # select model(SVD)
        model_options = Path("model_svd").glob("svd_*.pkl")
        model_path = st.sidebar.radio("select model", list(model_options), index=3)
        model_SVD = load_model(model_path)
        #st.write('保たれている情報:累積寄与率: {0}'.format(sum(model_SVD.explained_variance_ratio_)))
        
        # select model(clusterer:kmeans)
        model_options = Path("model_kmeans").glob("kmeans_*.pkl")
        model_path = st.sidebar.radio("select model", list(model_options))
        model_clusterer = load_model(model_path)
        
        if st.button("Recommend"):
            data_load_state = st.text("Loading Data...")
            
            movie_recommend = get_recommend_movie_df_in_recommend(selections, movie_pivot, df, movie_pivot_sparse, model_SVD, model_clusterer)
            
            movie_recommend.sort_values("svd_score", ascending=False, inplace=True)
            
            selected_movie_recommend = movie_recommend[~movie_recommend.index.isin(selections)].head(Number)
            selected_movie_recommend = selected_movie_recommend[['movie_title', 'genre', 'screening_time', 'mean_review_point']]
            
            # Filter
            titles = selected_movie_recommend['movie_title'].values
            genres = selected_movie_recommend['genre'].values
            times = selected_movie_recommend['screening_time'].values
            reviews = selected_movie_recommend['mean_review_point'].values
            
            data_load_state.text("Result:")
            st.write(selected_movie_recommend)
            
            # 風船飛ばす
            st.balloons()
            

            c1,c2 = st.columns([1,2])
            with c1:
                with st.expander("movie_title"):
                    for title in titles.tolist():
                        st.success(title)
                        st.write('********************')

            with c2:
                with st.expander("Genre"):
                    for genre in genres.tolist():
                        st.success('・'.join(str(genre).split('|')))
                        st.write('********************')
                        
    if choice == 'About':
        components.html(footer_temp,height=500)
  
        
if __name__ == '__main__':
	main()




