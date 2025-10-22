# analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from collections import Counter

def load_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Data loaded. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"ERROR: File not found at {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading data: {e}")
        sys.exit(1)

def explore_data(df: pd.DataFrame):
    print("\n=== Head ===")
    print(df.head())
    
    print("\n=== Info ===")
    print(df.info())
    
    print("\n=== Missing values per column ===")
    print(df.isnull().sum())

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Convert publish_time to datetime
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    # Create year column
    df['year'] = df['publish_time'].dt.year
    
    # For simplicity: drop rows missing title or year
    before = df.shape[0]
    df = df.dropna(subset=['title', 'year'])
    after = df.shape[0]
    print(f"Dropped {before - after} rows missing title or year.")
    
    # Optional: Create word count for abstract
    df['abstract_word_count'] = df['abstract'].fillna("").apply(lambda x: len(x.split()))
    
    return df

def basic_analysis(df: pd.DataFrame):
    print("\n=== Statistical Summary ===")
    print(df[['year', 'abstract_word_count']].describe())
    
    print("\n=== Publications by Year ===")
    counts_year = df['year'].value_counts().sort_index()
    print(counts_year.head(10))
    
    print("\n=== Top Journals ===")
    top_journals = df['journal'].value_counts().head(10)
    print(top_journals)

    # Most frequent words in titles (simple)
    all_titles = df['title'].dropna().str.lower().str.split()
    words = Counter([word for title in all_titles for word in title if len(word) > 5])
    most_common = words.most_common(10)
    print("\nMost common title words:", most_common)

def plot_publications_by_year(df: pd.DataFrame, output_path: str = "pubs_by_year.png"):
    plt.figure(figsize=(10, 6))
    counts = df['year'].value_counts().sort_index()
    sns.barplot(x=counts.index, y=counts.values, color='skyblue')
    plt.title("Number of Publications by Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Publications")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_top_journals(df: pd.DataFrame, output_path: str = "top_journals.png"):
    plt.figure(figsize=(10, 6))
    top10 = df['journal'].value_counts().head(10)
    sns.barplot(x=top10.values, y=top10.index, palette="viridis")
    plt.title("Top 10 Journals Publishing COVID‑19 Research")
    plt.xlabel("Number of Papers")
    plt.ylabel("Journal")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_histogram_abstract_words(df: pd.DataFrame, output_path: str = "hist_abstract_wordcount.png"):
    plt.figure(figsize=(8,5))
    plt.hist(df['abstract_word_count'].dropna(), bins=30, color='orange', edgecolor='black')
    plt.title("Distribution of Abstract Word Count")
    plt.xlabel("Abstract Word Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_scatter_sepal_vs_abstract(df: pd.DataFrame, output_path: str = "scatter_abstract_vs_year.png"):
    # Example: scatter abstract length vs year
    plt.figure(figsize=(8,5))
    sns.scatterplot(x='year', y='abstract_word_count', data=df, alpha=0.3)
    plt.title("Abstract Word Count vs Publication Year")
    plt.xlabel("Year")
    plt.ylabel("Abstract Word Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")

def main():
    filepath = os.path.join("data", "metadata.csv")
    df = load_data(filepath)
    explore_data(df)
    df_clean = clean_data(df)
    basic_analysis(df_clean)
    # Create plots directory if needed
    os.makedirs("plots", exist_ok=True)
    plot_publications_by_year(df_clean, output_path=os.path.join("plots","pubs_by_year.png"))
    plot_top_journals(df_clean, output_path=os.path.join("plots","top_journals.png"))
    plot_histogram_abstract_words(df_clean, output_path=os.path.join("plots","hist_abstract_wordcount.png"))
    plot_scatter_sepal_vs_abstract(df_clean, output_path=os.path.join("plots","scatter_abstract_vs_year.png"))

if __name__ == "__main__":
    main()
