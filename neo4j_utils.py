import os
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Neo4j Aura instance
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

#Neo4j desktop instance
#NEO4J_URI = os.getenv("NEO4J_LOCAL")
#NEO4J_USERNAME = os.getenv("NEO4J_LOCAL_USR")
#NEO4J_PASSWORD = os.getenv("NEO4J_LOCAL_PWD")

if not (NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD):
    raise EnvironmentError("Neo4j credentials not set in the environment.")


class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    def close(self):
        if self.driver:
            self.driver.close()

    def run_query(self, query, parameters=None):
        """
        Runs a Cypher query and returns the result.

        Args:
            query (str): The Cypher query to execute.
            parameters (dict, optional): The parameters for the query.

        Returns:
            list: Query result as a list of dictionaries.
        """
        parameters = parameters or {}
        try:
            with self.driver.session() as session:
                return session.run(query, parameters).data()
        except Neo4jError as e:
            print(f"Neo4j query error: {e}")
            return None

    def fetch_all(self, query: str, params: dict = None):
        """Execute a query and return all results."""
        with self.driver.session() as session:
            return list(session.run(query, params or {}))

    def fetch_single(self, query: str, params: dict = None):
        """Execute a query and return a single row."""
        with self.driver.session() as session:
            result = session.run(query, params or {})
            record = result.single()
            return record.data() if record else None

    def run_batch_query(self, query, batch):
        """
        Runs a Cypher query in batch mode.

        Args:
            query (str): The Cypher query with $batch as a parameter.
            batch (list): List of dictionaries to pass as batch data.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with self.driver.session() as session:
                session.run(query, {"batch": batch})
            return True
        except Neo4jError as e:
            print(f"Batch query error: {e}")
            return False
