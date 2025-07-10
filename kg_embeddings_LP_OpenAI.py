import os
import pandas as pd

import openai
import streamlit as st
from dotenv import load_dotenv
from neo4j_utils import Neo4jClient

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = openai.OpenAI(api_key=openai_api_key)

# Initialize Neo4j client
neo4j_client = Neo4jClient()

st.set_page_config(page_title="KG Embeddings", page_icon="🔎")
st.title("Add Aila Lesson Plan Embeddings")
st.write("Uses OpenAI to create embeddings of the `content` property of `AilaLessonPlan` nodes. The embeddings are saved to a property called `contentEmbedding`.")


def remove_embeddings():
    query = """
    MATCH (lp:AilaLessonPlan)
    WHERE lp.contentEmbedding IS NOT NULL
    REMOVE lp.contentEmbedding
    RETURN COUNT(lp) AS nodesUpdated
    """
    results = neo4j_client.run_query(query)

    if results is None:  # Ensure results is not None
        st.error("Error: No data returned from Neo4j. Check if your database is running and has AilaLessonPlan nodes.")
        return

    if not results:  # Ensure results is not empty
        st.warning("No lesson plan encodings found.")
        return

    st.success(f"{results[0].get('nodesUpdated', 0)} lesson plan embeddings removed.")


def get_embedding(text):
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    #return response["data"][0]["embedding"]
    return response.data[0].embedding

# Fetch lesson plans from Neo4j
def encode_and_store_embeddings():
    query = """
    MATCH (a:AilaLessonPlan)
    WHERE a.contentEmbedding IS NULL
    RETURN a.lessonPlanId AS lesson_id, a.content AS content
    """
    results = neo4j_client.run_query(query)

    if results is None:  # Ensure results is not None
        st.error("Error: No data returned from Neo4j. Check if your database is running and has AilaLessonPlan nodes.")
        return

    if not results:  # Ensure results is not empty
        st.warning("No new lesson plans found to encode.")
        return

    for record in results:
        lesson_id = record.get("lesson_id")
        content = record.get("content")

        if not content:
            st.warning(f"Skipping lesson {lesson_id}: Content is missing.")
            continue  # Skip empty content

        embedding = get_embedding(content)

        # Store embedding in Neo4j
        update_query = """
        MATCH (a:AilaLessonPlan {lessonPlanId: $lesson_id})
        SET a.contentEmbedding = $embedding
        """
        neo4j_client.run_query(update_query, {"lesson_id": lesson_id, "embedding": embedding})

    st.success("Lesson plan embeddings updated.")


def show_indexes():
    query = "SHOW INDEXES;"
    
    try:
        results = neo4j_client.run_query(query)

        if not results:
            st.warning("No indexes found in the database.")
            return
        
        df = pd.DataFrame(results)
        st.write("### Existing Indices in Neo4j:")
        st.dataframe(df)

    except Exception as e:
        st.error(f"Error retrieving indexes: {e}")


def delete_vector_search_index():
    query = '''
        DROP INDEX lesson_content_embedding_index IF EXISTS;
    '''
    try:
        result = neo4j_client.run_query(query)
        st.success("Index successfully deleted.")

    except Exception as e:
        # Print full traceback for debugging
        error_details = traceback.format_exc()
        st.error(f"Error deleting vector index: {e}")
        st.error(f"Detailed error traceback: {error_details}")


def create_vector_search_index():
    query = '''
        CREATE VECTOR INDEX lesson_content_embedding_index
        FOR (a:AilaLessonPlan)
        ON (a.contentEmbedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }};
    '''
    try:
        result = neo4j_client.run_query(query)
        st.success("Index successfully created.")

    except Exception as e:
        # Print full traceback for debugging
        error_details = traceback.format_exc()
        st.error(f"Error creating vector index: {e}")
        st.error(f"Detailed error traceback: {error_details}")

st.write("### Manage Embeddings & Indices")
if st.button("Remove Existing Embeddings"):
    remove_embeddings()
if st.button("Generate and Store Embeddings"):
    encode_and_store_embeddings()
if st.button("Show Current Indices in Neo4j"):
    show_indexes()
if st.button("Delete Vector Search Index in Neo4j"):
    delete_vector_search_index()
if st.button("Create Vector Search Index in Neo4j"):
    create_vector_search_index()
