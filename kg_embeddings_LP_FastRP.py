import pandas as pd
import traceback

import streamlit as st
from neo4j_utils import Neo4jClient


# Initialize Neo4j client
neo4j_client = Neo4jClient()

st.set_page_config(page_title="KG Embeddings", page_icon="🔎")
st.title("Add Aila Lesson Plan Embeddings")
st.write("- Uses the FastRP algorithm from Neo4j's GDS library to create embeddings for `AilaLessonPlan` nodes.")
st.write("- The embeddings are saved to a property called `lessonPlanEmbedding`.")
st.write("- A graph projection has to be created to run the FastRP algorithm.")

def projection_exists(graph_name="fullGraph"):
    query = f"CALL gds.graph.exists('{graph_name}') YIELD exists"
    result = neo4j_client.run_query(query)
    
    return result and result[0].get("exists", False)


def remove_embeddings():
    query = """
    MATCH (lp:AilaLessonPlan)
    WHERE lp.lessonPlanEmbedding IS NOT NULL
    REMOVE lp.lessonPlanEmbedding
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
        DROP INDEX aila_lesson_plan_vector_index IF EXISTS;
    '''
    try:
        result = neo4j_client.run_query(query)
        st.success("Index successfully deleted.")

    except Exception as e:
        error_details = traceback.format_exc()
        st.error(f"Error deleting vector index: {e}")
        st.error(f"Detailed error traceback: {error_details}")


def show_projections():
    query = "CALL gds.graph.list();"
    
    try:
        results = neo4j_client.run_query(query)

        if not results:
            st.warning("No projections found.")
            return
        
        df = pd.DataFrame(results)
        st.write("### Existing Graph Projections:")
        st.dataframe(df)

    except Exception as e:
        st.error(f"Error retrieving indexes: {e}")


def delete_projection():
    query = "CALL gds.graph.drop('fullGraph') YIELD graphName;"
    
    try:
        result = neo4j_client.run_query(query)
        st.success("Projection successfully deleted.")

    except Exception as e:
        # Print full traceback for debugging
        error_details = traceback.format_exc()
        st.error(f"Error deleting projection: {e}")
        st.error(f"Detailed error traceback: {error_details}")


def verify_projection():
    st.write("### Verifying Graph Projection")
    
    # Check if projection exists
    if not projection_exists():
        st.error("No projection named 'fullGraph' exists.")
        return False
    
    # Get projection info including schema
    info_query = """
    CALL gds.graph.list('fullGraph')
    YIELD graphName, nodeCount, relationshipCount, schema
    RETURN graphName, nodeCount, relationshipCount, schema
    """
    
    try:
        info = neo4j_client.run_query(info_query)
        if not info or len(info) == 0:
            st.error("No information returned for 'fullGraph' projection.")
            return False
            
        result = info[0]
        st.success(f"Found projection: {result['graphName']}")
        st.write(f"Total nodes: {result['nodeCount']}")
        st.write(f"Total relationships: {result['relationshipCount']}")
        
        # Check if AilaLessonPlan is in the schema
        schema = result['schema']

        node_labels = list(schema.get('nodes', {}).keys())
        
        if 'AilaLessonPlan' in node_labels:
            st.success(f"AilaLessonPlan nodes found in projection")
            return True
        else:
            st.error("AilaLessonPlan nodes not found in projection")
            st.info("Available labels: " + ", ".join(node_labels))
            return False
            
    except Exception as e:
        st.error(f"Error verifying projection: {e}")
        st.code(traceback.format_exc())
        return False


def project_into_GDS():
    if projection_exists():
        st.info("Graph projection 'fullGraph' already exists. Deleting for a fresh projection…")
        try:
            neo4j_client.run_query("CALL gds.graph.drop('fullGraph') YIELD graphName;")
        except Exception as e:
            st.warning(f"Could not drop existing graph. This might be okay. Error: {e}")

    st.write("Creating graph projection in GDS…")
    
    query_lesson_plans = """
    CALL gds.graph.project(
        'fullGraph',
        ['AilaLessonPlan', 'AilaLessonPlanPart'],
        {
            HAS_PART: { orientation: 'UNDIRECTED' }
        }
    )
    YIELD graphName, nodeCount, relationshipCount
    """
    
    try:
        result = neo4j_client.run_query(query_lesson_plans)
        if not result:
            st.error("Failed to create graph projection. This may mean the source data resulted in an empty graph.")
            return

        projection_info = result[0]
        nodeCount = projection_info.get("nodeCount", 0)
        relCount  = projection_info.get("relationshipCount", 0)
        st.success(f"Graph projection '{projection_info.get('graphName')}' created with {nodeCount} nodes and {relCount} relationships.")

        st.write("Verifying projection contents…")

        verify_query = """
        CALL gds.graph.list('fullGraph')
        YIELD schemaWithOrientation          // or just schema
        WITH schemaWithOrientation.nodes AS n
        RETURN keys(n) AS labels             // list of label names
        """

        verify_result = neo4j_client.run_query(verify_query)
        if not verify_result:
            st.error("Could not retrieve label information from the created projection.")
            return

        labels = verify_result[0]["labels"]

        if "AilaLessonPlan" in labels:
            st.success("Verified AilaLessonPlan nodes are in the projection.")
        else:
            st.warning("Could not verify AilaLessonPlan nodes in projection. ")

    except Exception as e:
        st.error(f"An error occurred during the GDS process: {e}")
        st.code(traceback.format_exc())


def compute_embeddings():
    # First, verify that the projection exists
    if not projection_exists():
        st.error("Graph projection 'fullGraph' doesn't exist. Please create it first.")
        return
    
    # Check if AilaLessonPlan nodes exist in database
    check_query = """
    MATCH (n:AilaLessonPlan) 
    RETURN count(n) as nodeCount
    """
    count_result = neo4j_client.run_query(check_query)
    
    if not count_result or count_result[0].get("nodeCount", 0) == 0:
        st.error("No AilaLessonPlan nodes found in the database.")
        return
    else:
        st.info(f"Found {count_result[0]['nodeCount']} AilaLessonPlan nodes in database.")
    
    # Check if AilaLessonPlan nodes exist in projection
    proj_check_query = """
    CALL gds.graph.list('fullGraph')
    YIELD schemaWithOrientation          // or just schema
    WITH schemaWithOrientation.nodes AS n
    RETURN keys(n) AS labels             // list of label names
    """
    
    try:
        check_result = neo4j_client.run_query(proj_check_query)
        
        if not check_result:
            st.error("Could not retrieve label information from the created projection.")
            return

        labels = check_result[0]["labels"]

        if "AilaLessonPlan" in labels:
            st.success("Verified AilaLessonPlan nodes are in the projection.")
        else:
            st.warning("Could not verify AilaLessonPlan nodes in projection. ")
            
    except Exception as e:
        st.error(f"Error checking projection: {e}")
        st.code(traceback.format_exc())
        return
    
    # Now attempt to run FastRP
    st.info("Computing FastRP Embeddings...")
    write_query = """
    CALL gds.fastRP.write(
        'fullGraph',
        {
            embeddingDimension: 1024,
            iterationWeights: [1, 1, 1, 1],
            normalizationStrength: 0.0,
            writeProperty: 'lessonPlanEmbedding'
        }
    ) YIELD nodePropertiesWritten
    RETURN nodePropertiesWritten;
    """
    
    try:
        # Execute the query and capture the result
        result = neo4j_client.run_query(write_query)
        
        if result is None:
            st.error("FastRP query returned None. Check Neo4j logs for errors.")
        elif not result:
            st.error("FastRP query returned empty result. Check configuration and Neo4j logs.")
        elif result[0].get("nodePropertiesWritten", 0) > 0:
            st.success(f"Embeddings computed and stored for {result[0]['nodePropertiesWritten']} nodes.")
        else:
            st.warning("⚠️ FastRP ran but no node properties were written.")
    except Exception as e:
        st.error(f"Error executing FastRP: {e}")
        st.code(traceback.format_exc())


def create_vector_index():
    st.spinner("Creating Vector Index for Embeddings...")
    query = """
    CREATE VECTOR INDEX aila_lesson_plan_vector_index
    FOR (l:AilaLessonPlan)
    ON (l.lessonPlanEmbedding)
    OPTIONS {indexConfig: {
        `vector.dimensions`: 1024,
        `vector.similarity_function`: 'cosine'
    }};
    """
    neo4j_client.run_query(query)
    st.success("Vector index created.")


st.write("### Graph projections")
if st.button("Verify Graph Projection"):
    verify_projection()
if st.button("Show Current Graph Projections"):
    show_projections()
if st.button("Delete Graph Projection"):
    delete_projection()
if st.button("Create Graph Projection"):
    project_into_GDS()

st.write("### Manage Embeddings & Indexes")
if st.button("Remove Existing Embeddings"):
    remove_embeddings()
if st.button("Generate and Store Embeddings"):
    compute_embeddings()
if st.button("Show Current Indices in Neo4j"):
    show_indexes()
if st.button("Delete Vector Search Index in Neo4j"):
    delete_vector_search_index()
if st.button("Create Vector Search Index in Neo4j"):
    create_vector_index()
