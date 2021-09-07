import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path

# core pkg
import streamlit as st 
import streamlit.components.v1 as stc

# EDA Pkgs
import pandas as pd

HTML_BANNER = """
    <div style="background-color:#464e5f;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;">Movie Directory App </h1>
    </div>
    """


def main():
    """Basics on st.beta columns/layout"""

    menu = ["Home","Search","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    stc.html(HTML_BANNER)

    df = pd.read_csv("final_movielens_500_db.csv")
    # Change Year to Datetime
    df['year'] = pd.to_datetime(df['year'])


    if choice == 'Home':
        st.subheader("Home")

        movies_title_list = df['title'].tolist()
        movies_title_list = reversed(movies_title_list)
        movie_choice = st.selectbox("Movie Title",movies_title_list)

        with st.expander('Movies DF',expanded=False):
            st.dataframe(df[df.title == movie_choice])

            # Filter
            img_link = df[df['title'] == movie_choice]['img_link'].values[0]
            title = df[df['title']== movie_choice]['title'].values
            genre = df[df['title']== movie_choice]['genres'].values


        # Layout
            # st.write(img_link)
            # st.image(img_link)
        c1,c2,c3 = st.columns([1,2,1])

        with c1:
            with st.expander("Title"):
                st.success(title)


        with c2:
            with st.expander("Image"):
                st.image(img_link)


        with c3:
            with st.expander("Genre"):
                st.write(str(genre).split('|'))

    elif choice == "Search":
        st.subheader("Search Movies")

        with st.expander("Search By Year"):
            movie_year = st.number_input("Year",1995, 2020)

            df_for_year = df[df['year'].dt.year == movie_year]
            st.dataframe(df_for_year)

        col1,col2,col3  = st.columns([1,2,1])

        with col1:
            with st.expander("Title"):
                for t in df_for_year['title'].tolist():
                    st.success(t)


        with col2:
            with st.expander("Images"):
                for i in df_for_year['img_link'].tolist():
                    st.image(i)


        with col3:
            with st.expander("Genre"):
                for g in df_for_year['genres'].tolist():
                    st.write(g)



    else:
        st.subheader("About")
        st.text("Built with Streamlit")
        st.text("Jesus Saves @JCharisTech")



if __name__ == '__main__':
	main()

