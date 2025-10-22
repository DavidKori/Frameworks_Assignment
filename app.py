# app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

@st.cache_data
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, low_memory=False)
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    df['year'] = df['publish_time'].dt.year
    df['abstract_word_count'] = df['abstract'].fillna("").apply(lambda x: len(x.split()))
    return df

def main():
    st.title("CORD‑19 Data Explorer")
    st.write("A simple interactive exploration of COVID‑19 research metadata.")
    
    filepath = os.path.join("data", "metadata.csv")
    df = load_data(filepath)
    
    min_year = int(df['year'].dropna().min())
    max_year = int(df['year'].dropna().max())
    year_range = st.slider("Select publication year range", min_year, max_year, (min_year, max_year))
    
    # Filter
    mask = (df['year'] >= year_range[0]) & (df['year'] <= year_range[1])
    df_filtered = df.loc[mask]
    
    st.subheader(f"Filtered dataset: {year_range[0]} to {year_range[1]}")
    st.write(f"Number of papers: {df_filtered.shape[0]}")
    st.dataframe(df_filtered[['title', 'journal', 'year']].head(10))
    
    # Visualization 1: Publications by Year
    st.subheader("Publications by Year")
    counts = df_filtered['year'].value_counts().sort_index()
    fig1, ax1 = plt.subplots(figsize=(8,5))
    sns.barplot(x=counts.index, y=counts.values, ax=ax1, color='skyblue')
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of Publications")
    st.pyplot(fig1)
    
    # Visualization 2: Top Journals
    st.subheader("Top Journals (in filtered set)")
    top10 = df_filtered['journal'].value_counts().head(10)
    fig2, ax2 = plt.subplots(figsize=(8,5))
    sns.barplot(x=top10.values, y=top10.index, ax=ax2, palette="viridis")
    ax2.set_xlabel("Number of Papers")
    ax2.set_ylabel("Journal")
    st.pyplot(fig2)
    
    # Visualization 3: Histogram of Abstract Word Counts
    st.subheader("Distribution of Abstract Word Count")
    fig3, ax3 = plt.subplots(figsize=(8,5))
    ax3.hist(df_filtered['abstract_word_count'].dropna(), bins=30, color='orange', edgecolor='black')
    ax3.set_xlabel("Abstract Word Count")
    ax3.set_ylabel("Frequency")
    st.pyplot(fig3)
    
    # Visualization 4: Scatter – Abstract Word Count vs Year
    st.subheader("Abstract Word Count vs Year")
    fig4, ax4 = plt.subplots(figsize=(8,5))
    sns.scatterplot(x='year', y='abstract_word_count', data=df_filtered, alpha=0.3, ax=ax4)
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Abstract Word Count")
    st.pyplot(fig4)

if __name__ == "__main__":
    main()
