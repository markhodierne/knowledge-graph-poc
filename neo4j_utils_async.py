import os
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import Neo4jError
from dotenv import load_dotenv


# Load environment variables
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not (NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD):
    raise EnvironmentError("Neo4j credentials not set in the environment.")


class AsyncNeo4jClient:
    def __init__(self):
        self.driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    async def close(self):
        await self.driver.close()

    async def run_query(self, query, parameters=None):
        """
        Asynchronously runs a Cypher query and returns the result.

        Args:
            query (str): The Cypher query to execute.
            parameters (dict, optional): The parameters for the query.

        Returns:
            list: Query result as a list of dictionaries.
        """
        parameters = parameters or {}
        try:
            async with self.driver.session() as session:
                result = await session.run(query, parameters)
                return [record.data() async for record in result]
        except Neo4jError as e:
            print(f"Neo4j query error: {e}")
            return None

    async def run_batch_query(self, query, batch):
        """
        Asynchronously runs a Cypher query in batch mode.

        Args:
            query (str): The Cypher query with $batch as a parameter.
            batch (list): List of dictionaries to pass as batch data.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            async with self.driver.session() as session:
                await session.run(query, {"batch": batch})
            return True
        except Neo4jError as e:
            print(f"Batch query error: {e}")
            return False
