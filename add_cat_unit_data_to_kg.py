import os
import pandas as pd
import numpy as np
import streamlit as st
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from dotenv import load_dotenv

INPUT_DIR = "data"

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
st.set_page_config("Unit Data to KG", page_icon="📚")


def add_cat_unit_data(driver, df, batch_size=500):
    """
    Processes a dataframe and adds nodes and relationships 
    to the Neo4j graph.

    Args:
        driver (neo4j.Driver): The Neo4j driver object.
        df (pd.DataFrame): The dataframe.
        batch_size (int): The number of rows to process in each batch 
            for performance optimization.
    """
    # Validate the DataFrame
    required_columns = {
        'futureUnit', 'priorUnit', 'plannedNumberOfLessons', 
        'priorKnowledgeRequirements', 'unitId', 'unitTitle',
        'unitDescription'
    }
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

                    MATCH (priorUnit:Unit {unitId: row.priorUnit})
                    MATCH (currentUnit:Unit {unitId: row.unitId, subjectTitle: row.subjectTitle})
                    SET currentUnit.plannedNumLessons = row.planned_number_of_lessons,
                        currentUnit.unitDescription = row.unitDescription
                        
                    WITH currentUnit, collect(DISTINCT priorUnit) AS priorUnits

                    CALL {
                        WITH currentUnit, priorUnits

                        // Case 1: Exactly one prior node
                        WITH currentUnit, priorUnits
                        WHERE size(priorUnits) = 1
                        UNWIND priorUnits AS singlePriorUnit
                        MERGE (currentUnit)-[:HAS_PRIOR_KNOWLEDGE]->(singlePriorUnit)

                        UNION

                        // Case 2: Multiple prior nodes
                        WITH currentUnit, priorUnits
                        UNWIND priorUnits AS p
                        WITH currentUnit, p
                        WHERE size(priorUnits) > 1 AND p.subjectTitle = currentUnit.subjectTitle
                        MERGE (currentUnit)-[:HAS_PRIOR_KNOWLEDGE]->(p)
                    }
                    """, {"batch": batch})

                except Neo4jError as e:
                    st.error(f"Error processing batch {i // batch_size + 1}: {e}")
                    return
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    return

    st.success("CAT Unit data added to graph successfully!")

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
    st.title("Add CAT Unit Data to KG")

    # File upload section
    os.makedirs(INPUT_DIR, exist_ok=True)
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

    if csv_files:
        selected_file = st.selectbox("Select a CSV file to upload:", csv_files)
        file_path = os.path.join(INPUT_DIR, selected_file)
        
        with st.spinner("Checking CSV... Please wait."):
            df = pd.read_csv(file_path)
            df = df.dropna(subset=["unitId"])
            df["priorUnit"] = df["priorUnit"].replace({np.nan: None})
            df = df.drop_duplicates()

        if st.button("Add CAT Unit data to KG"):
            try:
                add_cat_unit_data(driver, df)
                assign_uuids()
            except Exception as e:
                st.error(f"Error adding lesson plans: {e}")
    else:
        st.warning(f"No CSV files found in the '{INPUT_DIR}' directory.")
    # Close the Neo4j driver
    driver.close()
    

streamlit_ui()