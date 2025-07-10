import os
import pandas as pd
import re
import textwrap
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
st.set_page_config("NC Data to KG", page_icon="📚")


def clean_multiline_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace('\u00A0', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('•', '-')
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'").replace('–', '-').replace('—', '-')
    text = re.sub(r"[^a-zA-Z0-9.,:;!?'\"()\-\s]", '', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def add_nc_data(driver, df, batch_size=500):
    """
    Processes a dataframe and adds 'primaryFocus' and 'topic' nodes and relationships 
    to the Neo4j graph.

    Args:
        driver (neo4j.Driver): The Neo4j driver object.
        df (pd.DataFrame): The dataframe with columns 
            'principalFocus', 'focusStat', 'focusNonStat', 'topic', 
            'statutoryRequirements', and 'notesGuidance'.
        batch_size (int): The number of rows to process in each batch 
            for performance optimization.
    """
    # Validate the DataFrame
    required_columns = {
        'keyStageTitle', 'principalFocus', 'focusStat', 'focusNonStat', 
        'subjectTitle', 'yearTitle', 'topic', 'statutoryRequirements', 
        'notesGuidance', 'unitId', 'unitTitle'
    }
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required CSV columns: {', '.join(missing_columns)}")
        return
    else:
        for col in [
            "yearTitle", "focusStat", "focusNonStat",
            "statutoryRequirements", "notesGuidance",
            "principalFocus", "subjectTitle"
        ]:
            df[col] = df[col].fillna("").apply(clean_multiline_text)
        
        df["unitId"] = (
            pd.to_numeric(df["unitId"], errors="coerce")
            .astype("Int64")
        )
        
        df = df.where(pd.notnull(df), None)
        
        st.success("Correct CSV format confirmed.")

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
                    WITH
                        row.subjectTitle AS subjectTitle,
                        row.yearTitle AS yearTitle,
                        row.principalFocus AS principalFocus,
                        row.topic AS topic,
                        row.statutoryRequirements AS statutoryRequirements,
                        row.notesGuidance AS notesGuidance,
                        row.focusStat AS focusStat,
                        row.focusNonStat AS focusNonStat,
                        row.unitId AS unitId,
                        apoc.util.md5([row.principalFocus]) AS focusHash,
                        apoc.util.md5([row.focusStat]) AS statHash,
                        apoc.util.md5([row.focusNonStat]) AS nonStatHash

                    MATCH (s:Subject)
                    WHERE toLower(trim(s.subjectTitle)) = toLower(trim(subjectTitle))
                    OPTIONAL MATCH (y:Year {yearTitle: yearTitle})

                    MERGE (f:PrincipalFocus {id: focusHash})
                    ON CREATE SET f.focusDescription = principalFocus

                    MERGE (t:Topic {topic: topic, yearTitle: coalesce(yearTitle,'')})
                    ON CREATE
                    SET t.statutoryRequirements = statutoryRequirements,
                        t.notesGuidance = notesGuidance
                        
                    MERGE (s)-[:HAS_PRINCIPAL_FOCUS]->(f)
                    MERGE (f)-[:HAS_TOPIC]->(t)

                    WITH y, f, statHash, focusStat, nonStatHash, focusNonStat, t, unitId
                    
                    // Relate Year to Principal Focus
                    CALL {
                        WITH y, f
                        WITH y, f WHERE y IS NOT NULL
                        MERGE (y)-[:HAS_PRINCIPAL_FOCUS]->(f)
                        RETURN 0 AS dummy1
                    }
                    WITH f, statHash, focusStat, nonStatHash, focusNonStat, t, unitId

                    // Create Statutory Requirements
                    CALL {
                        WITH f, statHash, focusStat
                        WITH f, statHash, focusStat
                        WHERE focusStat IS NOT NULL AND focusStat <> ''
                        MERGE (sr:StatutoryRequirements {id: statHash})
                        ON CREATE SET sr.statDescription = focusStat
                        MERGE (f)-[:HAS_REQUIREMENTS]->(sr)
                        RETURN 0 AS dummy2
                    }
                    WITH f, nonStatHash, focusNonStat, t, unitId

                    // Create Non Statutory Requirements
                    CALL {
                        WITH f, nonStatHash, focusNonStat
                        WITH f, nonStatHash, focusNonStat
                        WHERE focusNonStat IS NOT NULL AND focusNonStat <> ''
                        MERGE (nsr:NonStatutoryRequirements {id: nonStatHash})
                        ON CREATE SET nsr.nonStatDescription = focusNonStat
                        MERGE (f)-[:HAS_REQUIREMENTS]->(nsr)
                        RETURN 0 AS dummy3
                    }
                    WITH t, unitId  

                    // Map Topics to Units
                    CALL {
                        WITH t, unitId
                        WITH t, unitId WHERE unitId IS NOT NULL
                        MATCH (u:Unit {unitId: unitId})
                        WITH t, u
                        MERGE (t)-[:MAPS_TO]->(u)
                        RETURN 0 AS dummy4
                    }
                    WITH 1 AS done
                    RETURN done
                    """, {"batch": batch}).consume()

                except Neo4jError as e:
                    st.error(f"Error processing batch {i // batch_size + 1}: {e}")
                    return
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    return

    st.success("National Curriculum data added to graph successfully!")

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
    st.title("Add NC Data to KG")
    st.write(
        "Upload a National Curriculum CSV and push its content into the graph. "
        "Topics are mapped to Lessons and Units (links require SL review)."
    )

    # File upload section
    os.makedirs(INPUT_DIR, exist_ok=True)
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

    if not csv_files:
        st.warning(f"No CSV files found in '{INPUT_DIR}'.")
        return

    selected_file = st.selectbox("Select a CSV file to upload:", csv_files)
    file_path = os.path.join(INPUT_DIR, selected_file)
        
    with st.spinner("Loading CSV..."):
        df = pd.read_csv(file_path, dtype={"unitId": "Int64"})

    st.write("Dataset preview:")
    st.dataframe(df.head())

    if st.button("Add NC data to KG"):
        try:
            add_nc_data(driver, df)
            assign_uuids()
        except Exception as e:
            st.error(f"Error adding lesson plans: {e}")



try:
    streamlit_ui()
finally:
    # Close the Neo4j driver
    driver.close()