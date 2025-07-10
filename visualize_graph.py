import os
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit as st
from dotenv import load_dotenv
import streamlit.components.v1 as components


class Neo4jGraphVisualizer:
    def __init__(self, environment="Neo4j Aura"):
        # Load environment variables
        load_dotenv()

        # Credential options
        self.credential_options = {
            "Neo4j Aura": {
                "uri": os.getenv("NEO4J_URI"),
                "username": os.getenv("NEO4J_USERNAME"),
                "password": os.getenv("NEO4J_PASSWORD"),
            },
            "Neo4j Local": {
                "uri": os.getenv("NEO4J_LOCAL"),
                "username": os.getenv("NEO4J_LOCAL_USR"),
                "password": os.getenv("NEO4J_LOCAL_PWD"),
            },
        }

        if environment not in self.credential_options:
            raise ValueError(f"Invalid environment: {environment}. Choose 'Neo4j Aura' or 'Neo4j Local'.")

        credentials = self.credential_options[environment]

        if not all(credentials.values()):
            raise EnvironmentError(f"Missing Neo4j credentials for {environment} in environment variables.")

        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(credentials["uri"], auth=(credentials["username"], credentials["password"]))

        # Color scheme for different node types
        self.color_mapping = {
            "KeyStage": "#FFF300",
            "Year": "#e76f51",
            "Subject": "#f4a261",
            "Unit": "#40e0d0",
            "Variant": "#e9c46a",
            "Lesson": "#BEF2BD",
            "ExamBoard": "#a68ad7",
            "Unknown": "#808080",
        }

    def fetch_graph_data(self):
        query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->(m)
        WITH collect(distinct n) as nodes, collect(distinct r) as relationships
        RETURN nodes, relationships
        """
        with self.driver.session() as session:
            result = session.run(query)
            record = result.single()
            
            if not record:
                return None, None

            nodes = self._process_nodes(record["nodes"])
            edges = self._process_edges(record["relationships"])

            return nodes, edges

    def _process_nodes(self, node_records):
        nodes = []
        for node in node_records:
            if node:
                nodes.append({
                    "id": node.id,
                    "labels": list(node.labels),
                    "properties": dict(node)
                })
        return nodes

    def _process_edges(self, edge_records):
        edges = []
        for rel in edge_records:
            edges.append({
                "source": rel.start_node.id,
                "target": rel.end_node.id,
                "type": rel.type,
                "properties": dict(rel)
            })
        return edges

    def build_graph(self, nodes, edges):
        net = Network(height="750px", width="100%", notebook=True)
        
        # Add nodes
        for node in nodes:
            node_type = node["labels"][0] if node["labels"] else "Unknown"
            color = self.color_mapping.get(node_type, "gray")

            label = f"{node_type}: {node['properties'].get('title', node['id'])}"
            title = "<br>".join([f"{k}: {v}" for k, v in node['properties'].items()])
            
            net.add_node(node["id"], label=label, color=color, title=title)

        # Add edges
        for edge in edges:
            title = edge["type"]
            if "examBoardTitle" in edge["properties"]:
                title += f' ({edge["properties"]["examBoardTitle"]})'
            net.add_edge(edge["source"], edge["target"], title=title, arrows="to")

        net.show("graph.html")
        with open("graph.html", "r", encoding="utf-8") as html_file:
            source_code = html_file.read()
        components.html(source_code, height=800, width=800)

    def check_graph_exists(self):
        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            return result.single()["count"]

    def cleanup(self):
        self.driver.close()


def streamlit_ui():
    st.set_page_config("Neo4j Graph Viewer", page_icon="🔍", layout="wide")
    st.title("Neo4j Graph Visualization")

    # Select environment
    environment = st.radio(
        "Choose Neo4j Connection",
        options=["Neo4j Aura", "Neo4j Local"]
    )

    # Initialize the visualizer with the selected environment
    visualizer = Neo4jGraphVisualizer(environment=environment)

    # Check if graph exists
    node_count = visualizer.check_graph_exists()
    if node_count == 0:
        st.warning(f"The {environment} database is empty. Please populate it with data first.")
        visualizer.cleanup()
        return

    # Fetch and display graph
    nodes, edges = visualizer.fetch_graph_data()
    if not nodes:
        st.warning("No data found in the graph.")
        visualizer.cleanup()
        return

    if st.button(f"Show Graph"):
        st.write("Creating visualization...")
        visualizer.build_graph(nodes, edges)

    visualizer.cleanup()


streamlit_ui()