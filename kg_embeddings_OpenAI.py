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
st.title("Add Node Property Embeddings")
st.write("Uses OpenAI to create embeddings of selected properties from selected node types. The embeddings are saved to a property with suffix `Embedding`.")


def get_available_node_labels():
    """Get all available node labels from the Neo4j database"""
    query = "CALL db.labels() YIELD label RETURN label ORDER BY label"
    try:
        results = neo4j_client.run_query(query)
        if results:
            return [record["label"] for record in results]
        return []
    except Exception as e:
        st.error(f"Error fetching node labels: {e}")
        return []


def get_node_properties(node_label):
    """Get text-like properties for a given node label, excluding embedding properties."""
    query = f"""
    MATCH (n:{node_label})
    UNWIND keys(n) AS prop
    WITH prop, n[prop] AS value
    WHERE NOT prop ENDS WITH 'Embedding'
        AND value IS NOT NULL
        AND toString(value) <> ""
    RETURN DISTINCT prop ORDER BY prop
    LIMIT 50
    """
    try:
        results = neo4j_client.run_query(query)
        if results:
            return [record["prop"] for record in results]
        return []
    except Exception as e:
        st.error(f"Error fetching properties for {node_label}: {e}")
        return []


def get_sample_content(node_label, property_name, limit=3):
    """Get sample content for preview"""
    query = f"""
    MATCH (n:{node_label})
    WHERE n.{property_name} IS NOT NULL AND toString(n.{property_name}) <> ""
    RETURN n.{property_name} AS content
    LIMIT {limit}
    """
    try:
        results = neo4j_client.run_query(query)
        if results:
            return [record["content"] for record in results]
        return []
    except Exception as e:
        st.error(f"Error fetching sample content: {e}")
        return []


def remove_embeddings(node_label, property_name):
    """Remove existing embeddings for specified node and property"""
    embedding_property = f"{property_name}Embedding"
    query = f"""
    MATCH (n:{node_label})
    WHERE n.{embedding_property} IS NOT NULL
    REMOVE n.{embedding_property}
    RETURN COUNT(n) AS nodesUpdated
    """
    try:
        results = neo4j_client.run_query(query)
        if results:
            count = results[0].get('nodesUpdated', 0)
            st.success(f"{count} {node_label} node embeddings removed from property '{embedding_property}'.")
        else:
            st.warning("No embeddings found to remove.")
    except Exception as e:
        st.error(f"Error removing embeddings: {e}")


def get_embedding(text):
    """Get embedding from OpenAI"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=str(text)
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error getting embedding: {e}")
        return None


def encode_and_store_embeddings(node_label, property_name, identifier_property=None):
    """Fetch nodes and store embeddings"""
    embedding_property = f"{property_name}Embedding"
    
    # Try to find a suitable identifier property if not provided
    if not identifier_property:
        common_ids = ['id', 'nodeId', f'{node_label.lower()}Id', 'name', 'title']
        for prop in common_ids:
            sample_query = f"""
            MATCH (n:{node_label})
            WHERE n.{prop} IS NOT NULL
            RETURN n.{prop} AS prop_value
            LIMIT 1
            """
            try:
                test_result = neo4j_client.run_query(sample_query)
                if test_result:
                    identifier_property = prop
                    break
            except:
                continue
    
    # If still no identifier found, use a generic approach
    if not identifier_property:
        query = f"""
        MATCH (n:{node_label})
        WHERE n.{embedding_property} IS NULL AND n.{property_name} IS NOT NULL AND toString(n.{property_name}) <> ""
        RETURN elementId(n) AS node_id, n.{property_name} AS content
        """
        id_field = "node_id"
    else:
        query = f"""
        MATCH (n:{node_label})
        WHERE n.{embedding_property} IS NULL AND n.{property_name} IS NOT NULL AND toString(n.{property_name}) <> ""
        RETURN n.{identifier_property} AS node_id, n.{property_name} AS content
        """
        id_field = "node_id"

    try:
        results = neo4j_client.run_query(query)

        if not results:
            st.warning(f"No new {node_label} nodes found to encode for property '{property_name}'.")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()
        total_nodes = len(results)
        
        for i, record in enumerate(results):
            node_id = record.get(id_field)
            content = record.get("content")

            if not content:
                st.warning(f"Skipping node {node_id}: Content is missing.")
                continue

            status_text.text(f"Processing node {i+1}/{total_nodes}: {node_id}")
            
            embedding = get_embedding(content)
            if embedding is None:
                continue

            # Store embedding in Neo4j
            if identifier_property:
                update_query = f"""
                MATCH (n:{node_label} {{`{identifier_property}`: $node_id}})
                SET n.{embedding_property} = $embedding
                """
            else:
                update_query = f"""
                MATCH (n:{node_label})
                WHERE elementId(n) = $node_id
                SET n.{embedding_property} = $embedding
                """
            
            neo4j_client.run_query(update_query, {"node_id": node_id, "embedding": embedding})
            
            progress_bar.progress((i + 1) / total_nodes)

        status_text.text("Complete!")
        st.success(f"{node_label} node embeddings updated for property '{property_name}'.")

    except Exception as e:
        st.error(f"Error encoding and storing embeddings: {e}")


def show_indexes():
    """Display current indexes"""
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


def create_vector_search_index(node_label, property_name):
    """Create vector search index for the specified node and property"""
    embedding_property = f"{property_name}Embedding"
    index_name = f"{node_label.lower()}_{property_name.lower()}_embedding_index"
    
    query = f'''
        CREATE VECTOR INDEX {index_name}
        FOR (n:{node_label})
        ON (n.{embedding_property})
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }}}};
    '''
    try:
        neo4j_client.run_query(query)
        st.success(f"Vector index '{index_name}' successfully created.")
    except Exception as e:
        st.error(f"Error creating vector index: {e}")


def delete_vector_search_index(index_name):
    """Delete specified vector search index"""
    query = f'DROP INDEX {index_name} IF EXISTS;'
    try:
        neo4j_client.run_query(query)
        st.success(f"Index '{index_name}' successfully deleted.")
    except Exception as e:
        st.error(f"Error deleting vector index: {e}")


# Main UI
st.write("### Select Node Type and Property")

# Get available node labels
node_labels = get_available_node_labels()

if not node_labels:
    st.error("No node labels found in the database. Please check your Neo4j connection.")
    st.stop()

# Node selection
selected_node = st.selectbox("Select Node Type:", node_labels)

# Property selection
if selected_node:
    properties = get_node_properties(selected_node)
    
    if not properties:
        st.warning(f"No text properties found for {selected_node} nodes.")
        st.stop()
    
    selected_property = st.selectbox("Select Property to Embed:", properties)
    
    # Show sample content
    if selected_property:
        st.write("### Sample Content Preview")
        samples = get_sample_content(selected_node, selected_property)
        for i, sample in enumerate(samples, 1):
            with st.expander(f"Sample {i}"):
                st.text(str(sample)[:500] + "..." if len(str(sample)) > 500 else str(sample))

# Optional identifier property
st.write("### Optional: Specify Identifier Property")
identifier_prop = st.text_input("Identifier Property (leave empty for auto-detection):", 
                                help="Property used to identify nodes (e.g., 'id', 'nodeId', 'name')")

st.write("### Actions")

if st.button("Remove Existing Embeddings", type="secondary"):
    if selected_node and selected_property:
        remove_embeddings(selected_node, selected_property)

if st.button("Generate and Store Embeddings", type="primary"):
    if selected_node and selected_property:
        encode_and_store_embeddings(selected_node, selected_property, identifier_prop if identifier_prop else None)

st.write("### Index Management")

if st.button("Show Current Indices"):
    show_indexes()
    
if st.button("Create Vector Index"):
    if selected_node and selected_property:
        create_vector_search_index(selected_node, selected_property)

index_to_delete = st.text_input("Index name to delete:")
if st.button("Delete Vector Index"):
    if index_to_delete:
        delete_vector_search_index(index_to_delete)