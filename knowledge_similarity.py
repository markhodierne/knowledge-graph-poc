import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from neo4j import GraphDatabase
import numpy as np
import matplotlib.pyplot as plt

# --- Neo4j connection setup ---
from dotenv import load_dotenv
import os

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# --- Functions ---
def fetch_embeddings(embedding_type="conceptEmbedding", limit=500):
    with driver.session() as session:
        result = session.run(f"""
            MATCH (c:Concept)<-[:TEACHES]-(l:Lesson)<-[:HAS_LESSON]-(v:Variant)
            WHERE c.{embedding_type} IS NOT NULL
            WITH c.name AS name, collect(DISTINCT v.subjectTitle)[0] AS subject, c.{embedding_type} AS embedding
            RETURN name, subject, embedding
            LIMIT $limit
        """, limit=limit).data()
    return pd.DataFrame(result)

def compute_similarity_pairs(df, threshold=0.9):
    names = df['name'].tolist()
    vectors = np.array(df['embedding'].tolist())
    sim_matrix = cosine_similarity(vectors)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sim = sim_matrix[i][j]
            if sim >= threshold:
                pairs.append((names[i], names[j], sim))
    return pd.DataFrame(pairs, columns=["Concept A", "Concept B", "Similarity"]).sort_values(by="Similarity", ascending=False)

def merge_concepts(source, target):
    merge_query = """
    MATCH (s:Concept {name: $source})
    MATCH (t:Concept {name: $target})

    OPTIONAL MATCH (l:Lesson)-[r:TEACHES]->(s)
    FOREACH (_ IN CASE WHEN r IS NULL THEN [] ELSE [1] END |
        MERGE (l)-[r2:TEACHES]->(t)
        SET r2.importance = r.importance
        DELETE r
    )
    WITH s, t

    OPTIONAL MATCH (s)-[r1:REQUIRES]->(x)
    FOREACH (_ IN CASE WHEN r1 IS NULL THEN [] ELSE [1] END |
        MERGE (t)-[:REQUIRES]->(x)
        DELETE r1
    )
    WITH s, t

    OPTIONAL MATCH (x)-[r2:REQUIRES]->(s)
    FOREACH (_ IN CASE WHEN r2 IS NULL THEN [] ELSE [1] END |
        MERGE (x)-[:REQUIRES]->(t)
        DELETE r2
    )
    WITH s

    DETACH DELETE s
    """
    with driver.session() as session:
        session.run(merge_query, source=source, target=target)

def plot_2d_projection_colored(df):
    names = df['name'].tolist()
    embeddings = np.array(df['embedding'].tolist())
    subjects = df['subject'].fillna("Unknown").tolist()

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    unique_subjects = sorted(set(subjects))
    color_map = {subject: plt.cm.tab10(i % 10) for i, subject in enumerate(unique_subjects)}

    fig, ax = plt.subplots(figsize=(10, 6))
    for subject in unique_subjects:
        idx = [i for i, s in enumerate(subjects) if s == subject]
        ax.scatter(coords[idx, 0], coords[idx, 1], label=subject, alpha=0.7)

    for i, name in enumerate(names):
        ax.annotate(name, (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.6)

    ax.set_title("2D PCA Projection of Concept Embeddings (Colored by Subject)")
    ax.legend(title="Subject", fontsize=8)
    st.pyplot(fig)

# --- Streamlit UI ---
st.set_page_config("🧠 Concept Merge Tool", page_icon="🔗")
st.title("🔗 Concept Similarity & Merge Tool")

embedding_type = st.selectbox(
    "Select embedding type to compare:",
    options=["conceptEmbedding", "textEmbedding", "hybridEmbedding"],
    format_func=lambda x: {
        "conceptEmbedding": "Graph Embedding (FastRP)",
        "textEmbedding": "Text Embedding (SentenceTransformers)",
        "hybridEmbedding": "Hybrid Embedding (Graph + Text)"
    }[x]
)

threshold = st.slider("Similarity threshold", min_value=0.7, max_value=0.99, value=0.9, step=0.01)
embedding_limit = st.number_input("Max number of concepts to compare", value=250, step=50)

if st.button("🔍 Find Similar Concepts"):
    df = fetch_embeddings(embedding_type=embedding_type, limit=embedding_limit)
    st.success(f"Fetched {len(df)} concept embeddings using '{embedding_type}'")
    similar_df = compute_similarity_pairs(df, threshold=threshold)
    st.session_state["similar_df"] = similar_df
    st.dataframe(similar_df.style.format({"Similarity": "{:.4f}"}))
    
if "similar_df" in st.session_state:
        similar_df = st.session_state["similar_df"]
        
        if not similar_df.empty:
            st.subheader("👁️ Review & Merge Concepts")
            rows_to_drop = []
            
            for idx, row in similar_df.iterrows():
                with st.expander(f"{row['Concept A']} ↔ {row['Concept B']} ({row['Similarity']:.4f})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Concept A:** {row['Concept A']}")
                        if st.button(f"Merge into B", key=f"merge_a_{idx}"):
                            merge_concepts(row['Concept A'], row['Concept B'])
                            rows_to_drop.append(idx)

                    with col2:
                        st.markdown(f"**Concept B:** {row['Concept B']}")
                        if st.button(f"Merge into A", key=f"merge_b_{idx}"):
                            merge_concepts(row['Concept B'], row['Concept A'])
                            rows_to_drop.append(idx)
                            
            if rows_to_drop:
                st.session_state["similar_df"] = similar_df.drop(index=rows_to_drop).reset_index(drop=True)
                st.rerun()
                
if st.button("📊 Visualize Embeddings in 2D"):
    df = fetch_embeddings(embedding_type=embedding_type, limit=embedding_limit)
    plot_2d_projection_colored(df)
