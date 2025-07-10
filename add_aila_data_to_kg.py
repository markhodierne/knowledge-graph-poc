import os
import pandas as pd
import streamlit as st
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from dotenv import load_dotenv

INPUT_DIR = "data"
USER = "Oak"

# Load environment variables
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not (NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD):
    raise EnvironmentError("Neo4j credentials not set in the environment.")

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Streamlit page configuration
st.set_page_config("Aila Data to KG", page_icon="📚")


def add_user(driver, user_name):
    with driver.session() as session:
        try:
            session.run("""
            MERGE (u:User {userName:'Oak'})
            """)

        except Neo4jError as e:
            st.error(f"Error adding user: {user_name}")
            return
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return


def add_lesson_plans(driver, df, batch_size=500):
    """
    Processes a dataframe and adds 'lessonPlan' nodes and relationships 
    to the Neo4j graph.

    Args:
        driver (neo4j.Driver): The Neo4j driver object.
        df (pd.DataFrame): The dataframe with columns 
            'lessonPlanId', 'oak_lesson_id', and 'content'.
        batch_size (int): The number of rows to process in each batch 
            for performance optimization.
    """
    # Validate the DataFrame
    required_columns = {"lessonPlanId", "oak_lesson_id", "content"}
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required CSV columns: {', '.join(missing_columns)}")
        return
    else:
        st.success("Correct CSV format confirmed.")
        st.write(df.head())
    
    # Convert DataFrame to a list of dictionaries
    rows = df.to_dict(orient="records")

    # Process in batches
    with driver.session() as session:
        with st.spinner("Processing... Please wait!"):
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                
                try:
                    session.run("""
                    UNWIND $batch AS row
                    MATCH (u:User {userName:'Oak'})
                    MATCH (l:Lesson {lessonId: row.oak_lesson_id}) 
                    MERGE (lp:AilaLessonPlan {lessonPlanId: row.lessonPlanId}) 
                    ON CREATE SET lp.content = row.content
                    MERGE (l)-[:HAS_LESSON_PLAN]->(lp)
                    MERGE (u)-[:HAS_CREATED]->(lp)
                    """, {"batch": batch})

                except Neo4jError as e:
                    st.error(f"Error processing batch {i // batch_size + 1}: {e}")
                    return
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    return

    st.success("Lesson plans added to graph successfully!")

def add_lesson_plan_parts(driver, df, batch_size=500):
    """
    Processes a dataframe and adds 'lessonPlanPart' nodes and relationships 
    to the Neo4j graph.

    Args:
        driver (neo4j.Driver): The Neo4j driver object.
        df (pd.DataFrame): The DataFrame with columns 
            'lessonPlanPartId', 'lessonPlanId', 'key' and 'content'.
        batch_size (int): The number of rows to process in each batch 
            for performance optimization.
    """
    # Validate the DataFrame
    required_columns = {"lessonPlanPartId", "lessonPlanId", "key", "content"}
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required CSV columns: {', '.join(missing_columns)}")
        return
    else:
        st.success("Correct CSV format confirmed.")
        st.write(df.head())
    
    # Convert DataFrame to a list of dictionaries
    rows = df.to_dict(orient="records")

    # Process in batches
    with driver.session() as session:
        with st.spinner("Processing... Please wait."):
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]  # Get the current batch
                try:
                    session.run("""
                    UNWIND $batch AS row
                    MATCH (lp:AilaLessonPlan {lessonPlanId: row.lessonPlanId}) 
                    MERGE (lpp:AilaLessonPlanPart {lessonPlanPartId: row.lessonPlanPartId}) 
                    ON CREATE SET lpp.part = row.key, lpp.content = row.content
                    MERGE (lp)-[:HAS_PART]->(lpp)
                    """, {"batch": batch})

                except Neo4jError as e:
                    st.error(f"Error processing batch {i // batch_size + 1}: {e}")
                    return
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    return

    st.success("Lesson plan parts added to graph successfully!")

def assign_uuids():
    """Assign UUIDs to all nodes without an ID."""
    query = """
    MATCH (n)
    WHERE n.id IS NULL
    SET n.id = apoc.create.uuid()
    """
    with driver.session() as session:
        session.run(query)

def streamlit_ui():
    st.title("Add Aila Lesson Data to KG")

    # File upload section
    os.makedirs(INPUT_DIR, exist_ok=True)
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

    if csv_files:
        selected_file = st.selectbox("Select a CSV file to upload:", csv_files)
        file_path = os.path.join(INPUT_DIR, selected_file)
        
        with st.spinner("Checking CSV... Please wait."):
            df = pd.read_csv(file_path)
            target_column = "oak_lesson_id"
            if target_column in df.columns:
                df[target_column] = df[target_column].fillna(0).astype(int)

        if st.button("Add lesson plans to KG"):
            try:
                add_user(driver, USER)
                add_lesson_plans(driver, df)
                assign_uuids()
            except Exception as e:
                st.error(f"Error adding lesson plans: {e}")
                
        if st.button("Add lesson plan parts to KG"):
            try:
                add_lesson_plan_parts(driver, df)
                assign_uuids()
            except Exception as e:
                st.error(f"Error adding lesson plan parts: {e}")
    else:
        st.warning(f"No CSV files found in the '{INPUT_DIR}' directory.")
    # Close the Neo4j driver
    driver.close()
    

streamlit_ui()