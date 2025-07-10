import streamlit as st
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os
from neo4j import GraphDatabase
from typing import Tuple
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

from utils.db_utils import get_lesson_plans_by_id, get_samples

# --- Load environment variables ---
load_dotenv()

# --- Postgres config ---
POSTGRES_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# --- Neo4j config ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_openai_embedding(text: str) -> list:
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"OpenAI embedding failed: {e}")
        return None
    
    
# --- Load lesson plans from PostgreSQL ---
def load_lesson_plans_from_postgres(table_name: str) -> pd.DataFrame:
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# --- Get knowledge concepts and prior requirements from Neo4j ---
def fetch_lesson_knowledge_details(
    lesson_plan_json: str, 
    driver,  # no need to specify GraphDatabase.driver type here
    top_k: int = 3
):
    # 1. Encode lesson plan to get its embedding
    query_embedding = np.array(text_model.encode(lesson_plan_json)).reshape(1, -1)

    # 2. Pull all existing lesson embeddings and metadata from Neo4j
    with driver.session() as session:
        results = session.run("""
            MATCH (l:Lesson)
            WHERE l.lessonPlanEmbedding IS NOT NULL
            RETURN l.lessonId AS lesson_id, l.lessonTitle AS title, l.lessonPlanEmbedding AS embedding
        """).data()

    if not results:
        raise ValueError("No lessonPlanEmbedding found in Neo4j.")

    lesson_ids = [row["lesson_id"] for row in results]
    titles = [row["title"] for row in results]
    embeddings = np.array([row["embedding"] for row in results])

    # 3. Compute cosine similarity using sklearn
    similarities = cosine_similarity(query_embedding, embeddings).flatten()

    lessons = [
        {
            "lesson_id": lid,
            "title": title,
            "similarity": sim
        }
        for lid, title, sim in zip(lesson_ids, titles, similarities)
    ]

    # 4. Sort by similarity
    top_lessons = sorted(lessons, key=lambda x: x["similarity"], reverse=True)[:top_k]
    top_lesson_ids = [lesson["lesson_id"] for lesson in top_lessons]

    # 5. Fetch TEACHES and REQUIRES concepts for top matches
    with driver.session() as session:
        result = session.run("""
            MATCH (l:Lesson)-[:TEACHES]->(c:Concept)
            WHERE l.lessonId IN $lesson_ids
            OPTIONAL MATCH (c)-[:REQUIRES]->(prereq:Concept)
            RETURN l.lessonId AS lesson_id, c.name AS concept, prereq.name AS prerequisite
        """, lesson_ids=top_lesson_ids)

    # 6. Organize into a dictionary
    knowledge_map = defaultdict(lambda: {"concepts": set(), "prerequisites": set()})

    for record in result:
        lid = record["lesson_id"]
        if record["concept"]:
            knowledge_map[lid]["concepts"].add(record["concept"])
        if record["prerequisite"]:
            knowledge_map[lid]["prerequisites"].add(record["prerequisite"])

    # 7. Add concepts and prerequisites to top lessons
    for lesson in top_lessons:
        lesson_id = lesson["lesson_id"]
        lesson["concepts"] = sorted(knowledge_map[lesson_id]["concepts"])
        lesson["prerequisites"] = sorted(knowledge_map[lesson_id]["prerequisites"])

    return top_lessons

def find_most_similar_concepts(input_text, top_k=5, min_similarity=0.6):
    input_embedding = text_model.encode(input_text).tolist()

    with driver.session() as session:
        concepts = session.run("""
            MATCH (c:Concept)
            WHERE c.textEmbedding IS NOT NULL
            RETURN c.name AS name, c.textEmbedding AS embedding
        """).data()

    similarities = []
    for c in concepts:
        score = cosine_similarity([input_embedding], [c["embedding"]])[0][0]
        if score >= min_similarity:
            similarities.append((c["name"], score))

    top_concepts = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    return top_concepts

def get_most_relevant_lesson_for_concept(concept_name):
    with driver.session() as session:
        result = session.run("""
            MATCH (l:Lesson)-[:TEACHES]->(c:Concept {name: $concept})
            RETURN l.lessonTitle AS lesson_title
            ORDER BY rand()  // or use metadata to rank
            LIMIT 1
        """, concept=concept_name).single()
    return result["lesson_title"] if result else None

def get_prerequisite_concepts(concept_name, max_hops=2):
    with driver.session() as session:
        result = session.run(f"""
            MATCH path = (c:Concept {{name: $concept}})-[:REQUIRES*1..{max_hops}]->(prereq:Concept)
            OPTIONAL MATCH (l:Lesson)-[:TEACHES]->(prereq)
            RETURN prereq.name AS prereq_name, prereq.definition AS definition,
                collect(DISTINCT l.lessonTitle)[0] AS source_lesson,
                length(path) AS depth
            ORDER BY depth ASC
        """, concept=concept_name).data()
    return result


# --- Streamlit UI ---
st.set_page_config("National Curriculum Mapping", page_icon="📘")
st.title("National Curriculum Mapping")

# -----------------------------------
# Select dataset
# -----------------------------------
st.subheader("Dataset selection")

# Load and prepare samples
samples_data = (
    get_samples()
    .sort_values(by="created_at", ascending=False)
    .assign(samples_options=lambda df: df["sample_title"] + " (" + df["number_of_lessons"].astype(str) + ")")
)

# Select sample
sample_option = st.selectbox(
    "Select a sample of lessons plans to query:",
    samples_data["samples_options"].tolist(),
    help="(Number of Lesson Plans in the Sample)"
)

# Get selected sample
selected_sample_row = samples_data[samples_data["samples_options"] == sample_option]
sample_id = selected_sample_row["id"].iloc[0]

# Display lesson plans for selected sample
lesson_plans = get_lesson_plans_by_id(sample_id)
total_lesson_plans = len(lesson_plans)
st.dataframe(lesson_plans)

# Set limit on lesson plans
limit = st.number_input(
    "(Optional) Set a limit for the number of lesson plans to query:",
    min_value=1,
    max_value=total_lesson_plans,
    value=total_lesson_plans,
    step=100
)

# Get lesson texts with limit
lessons_raw = lesson_plans["lesson_json_str"].dropna().unique().tolist()[:limit]

# Build preview-to-full mapping
lesson_options = {
    f"{text[:100].replace(chr(10), ' ')}...": text
    for text in lessons_raw
}

if 'lesson_selection' not in st.session_state:
    st.session_state.lesson_selection = list(lesson_options.keys())[0] if lesson_options else None

def on_lesson_selection_change():
    st.session_state.lesson_selection = st.session_state.lesson_selectbox

if st.session_state.lesson_selection in lesson_options:
    default_index = list(lesson_options.keys()).index(st.session_state.lesson_selection)
else:
    default_index = 0
    st.session_state.lesson_selection = list(lesson_options.keys())[0] if lesson_options else None

selected_preview = st.selectbox(
    "Select a lesson plan to query:",
    options=list(lesson_options.keys()),
    index=default_index,
    key="lesson_selectbox",
    on_change=on_lesson_selection_change
)

# -------------------------------
# 🔍 Match Lesson to Topics
# -------------------------------
st.subheader("Match Lesson Plan to Topics")

topic_min_similarity = st.slider("Minimum topic similarity threshold", 0.0, 1.0, 0.6, 0.05)
topic_top_k = st.slider("How many top matching topics to show?", 1, 20, 5)

def find_similar_topics(lesson_text, min_similarity=0.6, top_k=5):
    # Encode lesson text
    lesson_embedding = get_openai_embedding(lesson_text)
    
    if not lesson_embedding:
        return []

    with driver.session() as session:
        # Fetch topics with embeddings
        topics = session.run("""
            MATCH (t:Topic)
            WHERE t.topicEmbedding IS NOT NULL
            RETURN t.topic AS name, t.topicEmbedding AS embedding
        """).data()

    if not topics:
        st.warning("No topics with embeddings found in Neo4j.")
        return []

    # Compute cosine similarity
    topic_similarities = []
    for t in topics:
        score = cosine_similarity([lesson_embedding], [t["embedding"]])[0][0]
        if score >= min_similarity:
            topic_similarities.append((t["name"], score))

    # Sort by similarity and return top_k
    top_topics = sorted(topic_similarities, key=lambda x: x[1], reverse=True)[:top_k]
    return top_topics

if st.session_state.get("lesson_selection"):
    selected_lesson_plan = lesson_options[st.session_state.lesson_selection]

    if st.button("Find Matching Topics"):
        with st.spinner("Finding similar topics using vector embeddings..."):
            topic_matches = find_similar_topics(selected_lesson_plan, topic_min_similarity, topic_top_k)

            if topic_matches:
                st.markdown("### Matching Topics")
                for name, score in topic_matches:
                    st.markdown(f"- **{name}** (similarity: {score:.3f})")
            else:
                st.warning("❗ No topics found above the similarity threshold.")
else:
    st.info("⬆️ Please select a lesson plan first.")




