"""
CAT Knowledge Graph Builder.
This module provides functionality to create, visualize and manage a 
Neo4j knowledge graph from CAT (Curriculum and Assessment Tool) data.


Optimization Recommendations

Use Neo4j's Bulk Insert API:

Instead of individual MERGE operations, consider using Neo4j's bulk import tools.


Connection Pooling:

Implement proper connection pooling instead of opening/closing connections frequently.


Optimize Data Preprocessing:

Preprocess data once at startup and cache the results.
Simplify the preprocessing logic for better performance.


Batch Transaction Optimization:

Use explicit transactions for batches of operations.
Consider increasing the batch size if memory allows.


Query Optimization:

Use parameterized queries instead of constructing new query strings for each operation.
Consider using CREATE instead of MERGE when uniqueness is already guaranteed.


Streamlit-specific Optimizations:

Use Streamlit's caching mechanisms to avoid redundant processing.
Separate the UI logic from the data processing logic.



Implementing these optimizations would likely result in significant performance improvements for this knowledge graph builder application.


"""

import os
import unicodedata
import re
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from functools import lru_cache

import pandas as pd
import numpy as np
import streamlit as st
from ast import literal_eval
from dotenv import load_dotenv
from neo4j import GraphDatabase, Transaction
from neo4j_utils import Neo4jClient



st.set_page_config("Lesson Data to KG", page_icon=":tropical_fish:")
st.title("Add CAT Lesson Data to KG")


@dataclass
class Config:
    """Application configuration settings."""

    NEO4J_URI: str
    NEO4J_USERNAME: str 
    NEO4J_PASSWORD: str
    DATA_DIR: Path = Path('data')
    CAT_DATA_CSV: Path = DATA_DIR / 'cat_data.csv'
    BATCH_SIZE: int = 5000

    @classmethod
    def from_env(cls) -> 'Config':
        """Create config from environment variables."""
        load_dotenv()
        required_vars = ['NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}"
            )
        return cls(
            NEO4J_URI=os.getenv('NEO4J_URI'),
            NEO4J_USERNAME=os.getenv('NEO4J_USERNAME'),
            NEO4J_PASSWORD=os.getenv('NEO4J_PASSWORD')
        )


class Neo4jConnection:
    """Manage Neo4j database connection."""

    def __init__(self, config: Config):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )

    def execute_query(
        self, 
        query: str, 
        parameters: Optional[Dict] = None
    ) -> Any:
        """Execute a Neo4j query with parameters."""
        with self.driver.session() as session:
            result = session.run(query, parameters)
            data = [record.data() for record in result]
            return data
        
    def fetch_single(
        self, 
        query: str, 
        parameters: Optional[Dict] = None
    ) -> Dict:
        """Fetch a single result from Neo4j."""
        with self.driver.session() as session:
            result = session.run(query, parameters)
            record = result.single()
            return record.data() if record else None

    def close(self):
        """Close the database connection."""
        self.driver.close()


class GraphBuilder:
    """Handle creation and management of Neo4j knowledge graph."""

    def __init__(self, connection: Neo4jConnection, batch_size: int = 1000):
        """Initialize graph builder with database connection.
        Args:
            connection: Neo4j database connection
            batch_size: Number of rows to process in each batch
        """
        self.connection = connection
        self.batch_size = batch_size
    
    def _generate_lesson_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        df_relationships = df[[
            "unitVariantId", 
            "tierTitle", 
            "examBoardTitle", 
            "lessonId", 
            "lessonOrderInUnit"
        ]].copy()
        df_relationships = df_relationships.sort_values(by=[
            "unitVariantId", 
            "tierTitle", 
            "examBoardTitle", 
            "lessonOrderInUnit"
        ])
        df_relationships = df_relationships.drop_duplicates()
        df_relationships["toLessonId"] = df_relationships.groupby([
            "unitVariantId", 
            "tierTitle", 
            "examBoardTitle"
        ])["lessonId"].shift(-1)
        df_relationships = df_relationships.dropna(subset=["toLessonId"])

        return df_relationships

    def create_knowledge_graph(self, df: pd.DataFrame) -> None:
        """
        Create Neo4j knowledge graph from DataFrame using batch processing.

        Args:
            df (pd.DataFrame): The input data to create the graph from.
        """
        graph_parts = [
            "key_stage_year_subject",
            "unit",
            "unit_sequence",
            "variant_examboard",
            "variant_sequence",
            "lesson",
            "lesson_sequence",
            "keyword",
            "learning_points",
            "content_guidance",
            "slide_content",
            "exit_quiz",
            "starter_quiz",
            "threads",
            "equip_resources"
            "misconceptions",
            "pupil_outcomes",
            "supervision_levels",
            "teacher_tips",
        ]

        for part in graph_parts:
            self.create_graph_part(part, df)
            st.success(f"Completed processing for {part}.")

    def create_graph_part(self, part: str, df: pd.DataFrame) -> None:
        """Execute a specific part of the graph creation process."""
        try:
            if part == "lesson_sequence":
                df = self._generate_lesson_relationships(df)
                
            total_rows = len(df)
            expected_columns = set(df.columns)
            
            for start_idx in range(0, total_rows, self.batch_size):
                end_idx = min(start_idx + self.batch_size, total_rows)
                batch_df = df.iloc[start_idx:end_idx]

                batch_params = [
                    self._prepare_node_parameters(row, expected_columns)
                    for _, row in batch_df.iterrows()
                    if pd.notna(row["lessonId"])
                ]

                if not batch_params:
                    continue

                # Dynamically resolve query method
                query_method_name = f"_get_{part}_query"
                query_method = getattr(self, query_method_name, None)

                if not query_method or not callable(query_method):
                    raise AttributeError(
                        f"Query method '{query_method_name}' not found or not callable."
                    )

                # Execute the query
                self.connection.execute_query(
                    query_method(),
                    {"batch": batch_params}
                )

        except Exception as e:
            raise RuntimeError(
                f"Error creating graph part '{part}': {e}"
            )

    @staticmethod
    def _get_key_stage_year_subject_query() -> str:
        """Create KeyStage, Year, Subject nodes and relationships."""
        return """
        UNWIND $batch as row

        // Create KeyStage
        MERGE (k:KeyStage {keyStageTitle: row.keyStageTitle})
        ON CREATE SET k.phaseTitle = row.phaseTitle

        // Create Year and connect to KeyStage
        MERGE (y:Year {yearTitle: row.yearTitle})
        MERGE (k)-[:HAS_YEAR]->(y)

        // Create Subject
        MERGE (s:Subject {subjectTitle: row.subjectTitle})
        """

    @staticmethod
    def _get_unit_query() -> str:
        """Create Unit nodes and connect them to Year and Subject."""
        return """
        UNWIND $batch as row
        
        // Skip rows with missing unitId
        WITH row WHERE row.unitId IS NOT NULL
        
        // Match the Year and Subject first
        MATCH (y:Year {yearTitle: row.yearTitle})
        MATCH (s:Subject {subjectTitle: row.subjectTitle})
        
        // Create Unit and connect to Subject and Year
        MERGE (u:Unit {
            unitId: row.unitId,
            unitTitle: row.unitTitle,
            subjectTitle: row.subjectTitle
        })
        ON CREATE SET u.unitOrder = row.unitOrder, u.yearTitle = row.yearTitle
        ON MATCH SET  u.unitOrder = row.unitOrder, u.yearTitle = row.yearTitle
        
        // Ensure subjectTitle matches before creating relationships
        WITH s, u, y
        MERGE (s)-[:HAS_UNIT]->(u)
        MERGE (y)-[:HAS_UNIT]->(u)
        """
        
    @staticmethod
    def _get_unit_sequence_query() -> str:
        """Create sequencing relationships between Unit nodes."""
        return """
        MATCH (s:Subject)-[:HAS_UNIT]->(u:Unit)<-[:HAS_UNIT]-(y:Year)
        WHERE u.unitOrder IS NOT NULL
        WITH s.subjectTitle AS subject, y.yearTitle AS year, u
        ORDER BY u.unitOrder
        WITH subject, year, collect(u) AS units
        UNWIND range(0, size(units) - 2) AS i
        WITH units[i] AS currentUnit, units[i + 1] AS potentialNextUnit
        
        // Ensure a NEXT_UNIT relationship exists
        MERGE (currentUnit)-[:NEXT_UNIT]->(potentialNextUnit)
        
        // Identify units with multiple potential next units
        WITH currentUnit, potentialNextUnit
        
        // First, remove any duplicate or redundant NEXT_UNIT relationships
        WITH currentUnit, 
            COLLECT(DISTINCT potentialNextUnit) AS uniqueNextUnits
        
        UNWIND uniqueNextUnits AS nextUnit
        
        // Merge the NEXT_UNIT relationship if it doesn't already exist
        MERGE (currentUnit)-[:NEXT_UNIT]->(nextUnit)
        """

    @staticmethod
    def _get_variant_examboard_query() -> str:
        """Create Variant and ExamBoard nodes and their 
        relationships.
        """
        return """
        UNWIND $batch as row
        
        // Match the Unit with the correct subjectTitle
        MATCH (u:Unit {
            unitTitle: row.unitTitle, 
            subjectTitle: row.subjectTitle
        })

        // Create Variant
        MERGE (v:Variant {
            unitVariantId: COALESCE(row.unitVariantId, ""), 
            variantTier: COALESCE(row.tierTitle, ""),
            subjectTitle: row.subjectTitle
        })
        ON CREATE SET v.variantOrder = row.unitOrder
        ON MATCH SET v.variantOrder = row.unitOrder
        MERGE (u)-[:HAS_VARIANT]->(v)

        // Create ExamBoard if exists
        WITH v, row
        WHERE row.examBoardTitle IS NOT NULL
        MERGE (e:ExamBoard {examBoardTitle: row.examBoardTitle})
        MERGE (e)-[:OFFERS]->(v)
        """

    @staticmethod
    def _get_variant_sequence_query() -> str:
        """Create sequencing relationships between Variant nodes."""
        return """
        // Match Variants grouped by Subject, Year, ExamBoard, and Tier
        MATCH (s:Subject)-[:HAS_UNIT]->(u:Unit)<-[:HAS_UNIT]-(y:Year)
        MATCH (u)-[:HAS_VARIANT]->(v:Variant)
        OPTIONAL MATCH (v)<-[:OFFERS]-(e:ExamBoard)
        WHERE v.variantOrder IS NOT NULL
        
        // Group by subject, year, examBoard, and tier, with defaults 
        // for missing properties, ordered by variantOrder 
        WITH 
            s.subjectTitle AS subject,
            y.yearTitle AS year,
            COALESCE(e.examBoardTitle, 'No Exam Board') AS examBoard,
            COALESCE(v.variantTier, 'No Tier') AS tier, 
            v
        ORDER BY v.variantOrder

        // Collect variants into groups
        WITH subject, year, examBoard, tier, collect(v) AS variants
        
        // Loop through consecutive Variants within each group
        UNWIND range(0, size(variants) - 2) AS i
        WITH variants[i] AS currentVariant, variants[i + 1] AS nextVariant
        
        // Create NEXT_VARIANT relationships for consecutive Variants 
        // in the ordered sequence
        MERGE (currentVariant)-[:NEXT_VARIANT]->(nextVariant)
        """

    @staticmethod
    def _get_lesson_query() -> str:
        """Create Lesson and connect it to Variant."""
        return """
        UNWIND $batch as row
        
        // Match the Variant first
        MATCH (v:Variant {unitVariantId: row.unitVariantId})

        // Create Lesson
        MERGE (l:Lesson {lessonId: row.lessonId})
        WITH v, l, row
        WHERE row.lessonTitle IS NOT NULL
        //SET l.lessonOrderInUnit = row.globalOrder,
        //    l.lessonTitle = row.lessonTitle
        SET l.lessonTitle = row.lessonTitle
        
        // Create relationship
        MERGE (v)-[:HAS_LESSON]->(l)
        """
    
    @staticmethod
    def _get_lesson_sequence_query() -> str:
        """Create lesson-to-lesson sequencing relationships using CASE 
        for relationship type."""
        return """
        UNWIND $batch as row
        
        MATCH (l1:Lesson {lessonId: row.lessonId})
        MATCH (l2:Lesson {lessonId: row.toLessonId})
        
        WITH l1, l2, row,
        CASE row.tierTitle 
            WHEN 'Higher' THEN 'NEXT_LESSON_H_TIER'
            WHEN 'Foundation' THEN 'NEXT_LESSON_F_TIER'
            ELSE 'NEXT_LESSON'
        END as relType
        
        CALL apoc.create.relationship(
            l1,
            relType,
            {
                variantId: row.unitVariantId,
                examBoard: row.examBoardTitle,
                tier: row.tierTitle
            },
            l2
        ) YIELD rel
        
        RETURN COUNT(rel) as relationshipsCreated
        """
    
    @staticmethod
    def _get_keyword_query() -> str:
        """Create LessonKeywords."""
        return """
        UNWIND $batch as row

        // Match the Lesson node
        MATCH (l:Lesson {lessonId: row.lessonId})
        WHERE row.lessonKeywords IS NOT NULL

        WITH l, row.lessonKeywords AS keywordMap
        UNWIND keys(keywordMap) AS keyword

        WITH l, keyword, keywordMap[keyword] AS properties
        WHERE keyword IS NOT NULL AND properties IS NOT NULL

        // Create or merge the LessonKeyword node
        MERGE (kw:LessonKeyword {keyword: keyword})

        // Dynamically set properties
        FOREACH (propKey IN keys(properties) |
            SET kw[propKey] = properties[propKey]
        )

        // Ensure the relationship exists
        MERGE (l)-[:HAS_KEYWORD]->(kw)
        """
    
    @staticmethod
    def _get_learning_points_query() -> str:
        """Create KeyLearningPoints."""
        return """
        UNWIND $batch as row

        // Match the Lesson node
        MATCH (l:Lesson {lessonId: row.lessonId})
        WHERE row.keyLearningPoints IS NOT NULL

        // Unwind keyLearningPoints list
        UNWIND row.keyLearningPoints AS learningPoint

        WITH l, learningPoint
        WHERE learningPoint IS NOT NULL

        // Create or merge the KeyLearningPoint node
        MERGE (klp:KeyLearningPoint {keyLearningPoint: learningPoint})

        // Ensure the relationship exists
        MERGE (l)-[:HAS_LEARNING_POINT]->(klp)
        """
    
    @staticmethod
    def _get_content_guidance_query() -> str:
        """Create Content Guidance."""
        return """
        UNWIND $batch as row

        // Match the Lesson first
        MATCH (l:Lesson {lessonId: row.lessonId})
        WITH l, row.contentGuidance AS guidance
        WHERE guidance IS NOT NULL AND guidance <> ""

        MERGE (cg:ContentGuidance {contentGuidance: guidance})
        MERGE (l)-[:HAS_CONTENT_GUIDANCE]->(cg)
        """
    
    @staticmethod
    def _get_slide_content_query() -> str:
        """Create Slide Content."""
        return """
        UNWIND $batch as row

        // Match the Lesson first
        MATCH (l:Lesson {lessonId: row.lessonId})
        WITH l, row.slideContent AS slides
        WHERE slides IS NOT NULL AND slides <> ""

        MERGE (s:SlideContent {slideContent: slides})
        MERGE (l)-[:HAS_SLIDE_CONTENT]->(s)
        """
    
    @staticmethod
    def _get_exit_quiz_query() -> str:
        """Create Exit Quiz."""
        return """
        UNWIND $batch as row

        // Match the Lesson first
        MATCH (l:Lesson {lessonId: row.lessonId})
        WITH l, row.exitQuiz AS eQuiz, row.exitQuizId AS eQuizId
        WHERE eQuiz IS NOT NULL AND eQuiz <> ""

        MERGE (q:ExitQuiz {exitQuiz: eQuiz})
        ON CREATE SET q.exitQuizId = eQuizId
        MERGE (l)-[:HAS_EXIT_QUIZ]->(q)
        """
    
    @staticmethod
    def _get_starter_quiz_query() -> str:
        """Create Starter Quiz."""
        return """
        UNWIND $batch as row

        // Match the Lesson first
        MATCH (l:Lesson {lessonId: row.lessonId})
        WITH l, row.starterQuiz AS sQuiz, row.starterQuizId AS sQuizId
        WHERE sQuiz IS NOT NULL AND sQuiz <> ""

        MERGE (q:StarterQuiz {starterQuiz: sQuiz})
        ON CREATE SET q.starterQuizId = sQuizId
        MERGE (l)-[:HAS_STARTER_QUIZ]->(q)
        """
    
    @staticmethod
    def _get_threads_query() -> str:
        """Create Threads."""
        return """
        UNWIND $batch as row

        // Match the Unit first
        MATCH (u:Unit {unitId: row.unitId})
        WITH u, row.threadTitle AS threadTitle, row.threadId AS threadId
        WHERE threadTitle IS NOT NULL AND threadTitle <> ""

        MERGE (t:Thread {threadId: threadId})
        ON CREATE SET t.threadTitle = threadTitle
        MERGE (t)-[:HAS_UNIT]->(u)
        """

    @staticmethod
    def _get_equip_resources_query() -> str:
        """Create Lesson Equipment and Resources."""
        return """
        UNWIND $batch as row

        // Match the Lesson first
        MATCH (l:Lesson {lessonId: row.lessonId})
        WITH l, row.lessonEquipmentAndResources AS equipment
        WHERE equipment IS NOT NULL AND equipment <> ""

        MERGE (e:EquipmentResources {equipmentResources: equipment})
        MERGE (l)-[:HAS_EQUIPMENT_AND_RESOURCES]->(e)
        """
    
    @staticmethod
    def _get_misconceptions_query() -> str:
        """Create Misconceptions and Common Mistakes."""
        return """
        UNWIND $batch as row

        // Match the Lesson first
        MATCH (l:Lesson {lessonId: row.lessonId})
        WITH l, row.misconceptionsAndCommonMistakes AS misconceptions
        WHERE misconceptions IS NOT NULL AND misconceptions <> ""

        MERGE (m:MisconceptionsMistakes {misconceptionsMistakes: misconceptions})
        MERGE (l)-[:HAS_MISCONCEPTIONS_MISTAKES]->(m)
        """
    
    @staticmethod
    def _get_pupil_outcomes_query() -> str:
        """Create Pupil Lesson Outcomes."""
        return """
        UNWIND $batch as row

        // Match the Lesson first
        MATCH (l:Lesson {lessonId: row.lessonId})
        WITH l, row.pupilLessonOutcome AS outcome
        WHERE outcome IS NOT NULL AND outcome <> ""

        MERGE (o:PupilLessonOutcome {pupilLessonOutcome: outcome})
        MERGE (l)-[:HAS_PUPIL_OUTCOME]->(o)
        """
    
    @staticmethod
    def _get_supervision_levels_query() -> str:
        """Create Supervision Levels."""
        return """
        UNWIND $batch as row

        // Match the Lesson first
        MATCH (l:Lesson {lessonId: row.lessonId})
        WITH l, row.supervisionLevel AS level
        WHERE level IS NOT NULL AND level <> ""

        MERGE (s:SupervisionLevel {supervisionLevel: level})
        MERGE (l)-[:HAS_SUPERVISION_LEVEL]->(s)
        """
    
    @staticmethod
    def _get_teacher_tips_query() -> str:
        """Create Teacher Tips."""
        return """
        UNWIND $batch as row

        // Match the Lesson first
        MATCH (l:Lesson {lessonId: row.lessonId})
        WITH l, row.teacherTips AS tip
        WHERE tip IS NOT NULL AND tip <> ""

        MERGE (t:TeacherTips {teacherTips: tip})
        MERGE (l)-[:HAS_TEACHER_TIP]->(t)
        """

    @staticmethod
    def _prepare_node_parameters(row: pd.Series, expected_columns: set) -> Dict:
        return {
            "keyStageTitle": row.get("keyStageTitle") 
                if "keyStageTitle" in expected_columns else None,
            "phaseTitle": row.get("phaseTitle") 
                if "phaseTitle" in expected_columns else None,
            "yearTitle": row.get("yearTitle") 
                if "yearTitle" in expected_columns else None,
            "subjectTitle": row.get("subjectTitle") 
                if "subjectTitle" in expected_columns else None,
            "unitTitle": row.get("unitTitle") 
                if "unitTitle" in expected_columns else None,
            "unitOrder": int(row["unitOrder"]) 
                if "unitOrder" in expected_columns 
                and pd.notna(row.get("unitOrder")) else None,
            "unitVariantId": int(row["unitVariantId"]) 
                if "unitVariantId" in expected_columns 
                and pd.notna(row.get("unitVariantId")) else None,
            "tierTitle": row.get("tierTitle") 
                if "tierTitle" in expected_columns 
                and pd.notna(row.get("tierTitle")) else None,
            "lessonId": int(row["lessonId"]) 
                if "lessonId" in expected_columns 
                and pd.notna(row.get("lessonId")) else None,
            "toLessonId": int(row["toLessonId"]) 
                if "toLessonId" in expected_columns 
                and pd.notna(row.get("toLessonId")) else None,
            "lessonTitle": row.get("lessonTitle") 
                if "lessonTitle" in expected_columns else None,
            "examBoardTitle": row.get("examBoardTitle") 
                if "examBoardTitle" in expected_columns 
                and pd.notna(row.get("examBoardTitle")) else None,
            "lessonKeywords": row.get("lessonKeywords") 
                if "lessonKeywords" in expected_columns else None,
            "keyLearningPoints": row.get("keyLearningPoints") 
                if "keyLearningPoints" in expected_columns else None,
            "unitId": int(row["unitId"]) 
                if "unitId" in expected_columns 
                and pd.notna(row.get("unitId")) else None,
            "contentGuidance": row.get("contentGuidance")
                if "contentGuidance" in expected_columns else None,
            "slideContent": row.get("slideContent")
                if "slideContent" in expected_columns else None,
            "exitQuiz": row.get("exitQuiz")
                if "exitQuiz" in expected_columns else None,
            "exitQuizId": row.get("exitQuizId")
                if "exitQuizId" in expected_columns else None,
            "starterQuiz": row.get("starterQuiz")
                if "starterQuiz" in expected_columns else None,
            "starterQuizId": row.get("starterQuizId")
                if "starterQuizId" in expected_columns else None,
            "threadId": row.get("threadId")
                if "threadId" in expected_columns else None,
            "threadTitle": row.get("threadTitle")
                if "threadTitle" in expected_columns else None,
            "lessonEquipmentAndResources": row.get("lessonEquipmentAndResources")
                if "lessonEquipmentAndResources" in expected_columns else None,
            "misconceptionsAndCommonMistakes": row.get("misconceptionsAndCommonMistakes")
                if "misconceptionsAndCommonMistakes" in expected_columns else None,
            "pupilLessonOutcome": row.get("pupilLessonOutcome")
                if "pupilLessonOutcome" in expected_columns else None,
            "supervisionLevel": row.get("supervisionLevel")
                if "supervisionLevel" in expected_columns else None,
            "teacherTips": row.get("teacherTips")
                if "teacherTips" in expected_columns else None,
        }

class StreamlitApp:
    """Main Streamlit application class."""

    def __init__(self):
        """Initialize Streamlit application."""
        self.config = Config.from_env()
        self.connection = Neo4jConnection(self.config)
        self.graph_builder = GraphBuilder(self.connection)

        if "graph_exists" not in st.session_state:
            st.session_state.graph_exists = self._check_graph_exists()
        if "cat_data" not in st.session_state:
            st.session_state.cat_data = self._load_data()
    
    def _load_data(self) -> pd.DataFrame:
        """Load and preprocess CAT data from the configured CSV file."""
        try:
            with st.spinner("Loading CAT data..."):
                df = pd.read_csv(self.config.CAT_DATA_CSV)
                processed_df = self._preprocess_dataframe(df)
                st.session_state.cat_data = processed_df
                st.success("Data loaded successfully.")
                
            return st.session_state.cat_data
        except FileNotFoundError:
            st.error("The data file is missing. Please upload it.")
            raise
        except pd.errors.EmptyDataError:
            st.error("The data file is empty. Please check the file.")
            raise
    
    @staticmethod
    def _clean_mathematical_text(text: str) -> str:
        """
        Clean mathematical text for Neo4j by converting special 
        characters to proper Unicode.
        
        Args:
            text: Input text containing mathematical symbols
            
        Returns:
            Cleaned text with proper Unicode encoding
        """
        if not isinstance(text, str):
            return text
            
        # Mathematical symbol mappings
        symbol_map = {
            # Basic arithmetic
            '−': '\u2212',  # minus sign
            '=': '\u003D',  # equals sign
            '+': '\u002B',  # plus sign
            '×': '\u00D7',  # multiplication sign
            '÷': '\u00F7',  # division sign
            
            # Superscripts
            '²': '\u00B2',  # superscript 2
            '³': '\u00B3',  # superscript 3
            
            # Greek letters (commonly used in physics/math)
            'Δ': '\u0394',  # Delta
            'α': '\u03B1',  # alpha
            'β': '\u03B2',  # beta
            'θ': '\u03B8',  # theta
            
            # Other mathematical symbols
            '∑': '\u2211',  # sum
            '∫': '\u222B',  # integral
            '∞': '\u221E',  # infinity
            '≈': '\u2248',  # approximately equal
            '≠': '\u2260',  # not equal
            '≤': '\u2264',  # less than or equal
            '≥': '\u2265',  # greater than or equal
        }
        
        # Replace symbols with their Unicode equivalents
        for symbol, unicode_char in symbol_map.items():
            text = text.replace(symbol, unicode_char)
        
        # Normalize Unicode composition
        text = unicodedata.normalize('NFC', text)
        
        # Remove any remaining problematic characters
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        
        return text

    def _clean_string(self, data):
        """
        Recursively clean and validate data structures.
        - Cleans strings inside lists and dictionaries without converting them into JSON strings.
        - Ensures JSON-like structures remain as valid Python lists/dictionaries.

        Args:
            data: The input data to clean. Can be a dictionary, list, string, or other types.

        Returns:
            The cleaned and validated data.
        """
        if isinstance(data, dict):
            return {k: self._clean_string(v) for k, v in data.items()}  # Preserve dictionary structure

        elif isinstance(data, list):
            return [self._clean_string(v) for v in data]  # Preserve list structure

        elif isinstance(data, str):
            try:
                # Remove control characters but preserve valid JSON format
                data = self._clean_mathematical_text(data)
                data = re.sub(r'[\x00-\x1F\x7F]', '', data)
                #data = data.replace('\\', '\\\\').replace('"', '\\"')  # Escape characters correctly
                data.encode('utf-8')  # Ensure valid UTF-8 encoding
                return data

            except UnicodeEncodeError:
                return ''.join(char for char in data if not (0xD800 <= ord(char) <= 0xDFFF))

        return data  # Return non-string values unchanged

    def _parse_and_clean(self, x, extract_keys=None, join_as_text=False):
        """
        Extracts key-value pairs from a list of dictionaries or JSON string.
        Returns either a dictionary or a list, depending on the input.

        Args:
            x (str, list, dict): Input data, which could be a JSON string, a list of dictionaries, or None.
            extract_keys (list): A list of keys to extract from each dictionary.
            join_as_text (bool): Whether to return a concatenated string or a dictionary/list.

        Returns:
            dict, list, or str: 
            - Returns a dictionary if multiple keys are extracted.
            - Returns a list if only one key is extracted.
            - Returns a concatenated string if `join_as_text=True`.
        """
        if extract_keys is None:
            extract_keys = []

        # Handle None, empty lists, and NumPy arrays safely
        if x is None or (isinstance(x, (list, np.ndarray)) and len(x) == 0):
            return [] if len(extract_keys) == 1 else {}

        if isinstance(x, np.ndarray):  
            x = x.tolist()  # Convert NumPy array to a Python list

        extracted_values = {}

        primary_key = None
        if len(extract_keys) > 0:
            primary_key = extract_keys[0]

        # If input is already a list of dictionaries, process it directly
        if isinstance(x, list):
            for item in x:
                if isinstance(item, dict):
                    primary_key = extract_keys[0]  # First key is the unique identifier
                    
                    # If extracting only ONE key, return a LIST
                    if len(extract_keys) == 1:
                        if primary_key in item:
                            extracted_values.setdefault(primary_key, []).append(item[primary_key])
                    else:
                        unique_id = item.get(primary_key, "").strip()
                        if unique_id:
                            extracted_values[unique_id] = {
                                k: item[k] for k in extract_keys if k in item
                            }

            return list(extracted_values.get(primary_key, [])) if len(extract_keys) == 1 else extracted_values

        # If input is a **string**, check if it's JSON and parse it
        try:
            if isinstance(x, str):
                x = x.strip()

                # If already a valid JSON list, parse and recursively process it
                if x.startswith("[") and x.endswith("]"):  
                    parsed = json.loads(x)
                    
                    # If parsed result is a list, pass it back into `_parse_and_clean()`
                    if isinstance(parsed, list):
                        return self._parse_and_clean(parsed, extract_keys, join_as_text)

        except json.JSONDecodeError as e:
            st.write(f"🚨 Error parsing JSON: {str(e)} - Raw value: {x}")

        return [] if len(extract_keys) == 1 else {}


    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for graph creation."""
        df = df.replace({np.nan: None})

        # Convert JSON-like strings into lists BEFORE cleaning
        def json_to_list(x):
            if isinstance(x, list):  
                return x  # Already a list, return unchanged
            
            if isinstance(x, str):
                try:
                    # Ensure valid JSON format before parsing
                    x = x.strip()
                    if x.startswith("[") and x.endswith("]"):
                        parsed = json.loads(x)  # Convert JSON string to list
                        if isinstance(parsed, list):
                            return parsed  # Correctly parsed list
                except json.JSONDecodeError as e:
                    st.write(f"🚨 JSONDecodeError: {e} for value: {x}")
                    return []  # Return empty list if parsing fails
            
            return []  # Default to empty list

        # Apply json_to_list() to keyLearningPoints
        df["keyLearningPoints"] = df["keyLearningPoints"].apply(json_to_list)
        
        # Apply deep cleaning but keep list format
        def clean_list(x):
            if isinstance(x, list):
                return [self._clean_string(v) for v in x]  # Clean each element
            return x  # If not a list, return unchanged

        df["keyLearningPoints"] = df["keyLearningPoints"].apply(clean_list)
        
        # Ensure `keyLearningPoints` is **always** a list of strings
        def extract_learning_points(x):
            if isinstance(x, list):
                return [str(item["keyLearningPoint"]) if isinstance(item, dict) and "keyLearningPoint" in item else str(item) for item in x]
            return []  # Default to empty list

        df["keyLearningPoints"] = df["keyLearningPoints"].apply(extract_learning_points)

        # Continue with normal string column cleaning
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            df[col] = df[col].apply(lambda x: self._clean_string(x) if isinstance(x, str) else x)

        # Apply parsing function to ensure correct data structure
        df["lessonKeywords"] = df["lessonKeywords"].apply(
            lambda x: self._parse_and_clean(
                x, 
                extract_keys=["keyword", "description"], 
                join_as_text=False
            )
        )
        
        return df.astype({
            'contentGuidance': 'object',
            'examBoardTitle': 'string',
            'exitQuiz': 'string',
            'exitQuizId': 'Int64',
            'keyLearningPoints': 'object',
            'keyStageTitle': 'string',
            'lessonEquipmentAndResources': 'string',
            'lessonId': 'Int64',
            'lessonKeywords': 'object',
            'lessonOrderInUnit': 'Int64',
            'lessonTitle': 'string',
            'misconceptionsAndCommonMistakes': 'string',
            'pupilLessonOutcome': 'string',
            'starterQuiz': 'string',
            'starterQuizId': 'Int64',
            'subjectTitle': 'string',
            'supervisionLevel': 'string',
            'teacherTips': 'string',
            'tierTitle': 'string',
            'unitOrder': 'Int64',
            'unitTitle': 'string',
            'unitVariantId': 'Int64',
            'yearTitle': 'string',
            'phaseTitle': 'string',
            'unitId': 'Int64',
            'threadTitle': 'string',
            'threadId': 'Int64',
            #'slideContent': 'string',
        })


    def _check_graph_exists(self) -> bool:
        """Check if the knowledge graph exists in the database."""
        query = "MATCH (n) RETURN COUNT(n) as count"
        result = self.connection.fetch_single(query)
        return result["count"] > 0

    def run(self):
        """Run the Streamlit application."""
        if st.button("Build Entire Knowledge Graph"):
            self._handle_graph_creation()

        st.subheader("Build knowledge graph by adding data in parts:")
        if st.button("Create KeyStage, Year, Subject"):
            self._create_key_stage_year_subject()
        if st.button("Create KeyStage, Year, Subject (FAST)"):
            self._create_key_stage_year_subject_fast()
        if st.button("Create Units"):
            self._create_units()
        if st.button("Create Variants and ExamBoards"):
            self._create_variants_examboards()
        if st.button("Create Lessons"):
            self._create_lessons()
        if st.button("Create Keywords"):
            self._create_keywords()
        if st.button("Create Key Learning Points"):
            self._create_learning_points()
        if st.button("Create Content Guidance"):
            self._create_content_guidance()
        if st.button("Create Slide Content"):
            self._create_slide_content()
        if st.button("Create Exit Quiz"):
            self._create_exit_quiz()
        if st.button("Create Starter Quiz"):
            self._create_starter_quiz()
        if st.button("Create Threads"):
            self._create_threads()
        if st.button("Create Equipment and Resources"):
            self._create_equip_and_resources()
        if st.button("Create Misconceptions and Common Mistakes"):
            self._create_misconceptions()
        if st.button("Create Pupil Lesson Outcomes"):
            self._create_pupil_lesson_outcomes()
        if st.button("Create Supervision Levels"):
            self._create_supervision_levels()
        if st.button("Create Teacher Tips"):
            self._create_teacher_tips()

        if st.session_state.graph_exists:
            if st.button("Delete Current Knowledge Graph"):
                self._delete_knowledge_graph()

        self.connection.close()

    def _handle_graph_creation(self):
        """Handle graph creation."""
        with st.spinner("Building Knowledge Graph..."):
            if "cat_data" not in st.session_state:
                self._load_data()
            df = st.session_state.cat_data
            self.graph_builder.create_knowledge_graph(df)
            st.success("Knowledge Graph created successfully.")
            st.session_state.graph_exists = True

    def _create_key_stage_year_subject(self):
        """Handle creation of KeyStage, Year, Subject."""
        with st.spinner("Creating KeyStage, Year, Subject..."):
            if "cat_data" not in st.session_state:
                self._load_data()
            df = st.session_state.cat_data
            self.graph_builder.create_graph_part("key_stage_year_subject", df)
            st.success("KeyStage, Year, Subject created.")
            st.session_state.graph_exists = True
    
    def _create_key_stage_year_subject_fast(self):
        """Handle creation of KeyStage, Year, Subject."""
        with st.spinner("Creating KeyStage, Year, Subject (FAST)..."):
            query = """
            CALL apoc.periodic.iterate(
            "LOAD CSV WITH HEADERS FROM 'file:///cat_data.csv' AS row RETURN row",
            \"
                MERGE (k:KeyStage {keyStageTitle: row.keyStageTitle})
                ON CREATE SET k.phaseTitle = row.phaseTitle

                MERGE (y:Year {yearTitle: row.yearTitle})
                MERGE (k)-[:HAS_YEAR]->(y)
                
                MERGE (s:Subject {subjectTitle: row.subjectTitle})
            \",
            {batchSize: 1000, parallel: true}
            )
            """
            
            self.connection.execute_query(query)
            st.success("KeyStage, Year, Subject created (via APOC).")
            st.session_state.graph_exists = True

    def _create_units(self):
        """Handle creation of Units."""
        with st.spinner("Creating Units..."):
            if "cat_data" not in st.session_state:
                self._load_data()
            df = st.session_state.cat_data
            self.graph_builder.create_graph_part("unit", df)
            # self.graph_builder.create_graph_part("unit_sequence", df)  # temp disable NEXT_UNIT relationships
            st.success("Units created.")
            st.session_state.graph_exists = True

    def _create_variants_examboards(self):
        """Handle creation of Variants and ExamBoards."""
        with st.spinner("Creating Variants and ExamBoards..."):
            if "cat_data" not in st.session_state:
                self._load_data()
            df = st.session_state.cat_data
            self.graph_builder.create_graph_part("variant_examboard", df)
            self.graph_builder.create_graph_part("variant_sequence", df)
            st.success("Variants and ExamBoards created.")
            st.session_state.graph_exists = True

    def _create_lessons(self):
        """Handle creation of Lessons."""
        with st.spinner("Creating Lessons..."):
            if "cat_data" not in st.session_state:
                self._load_data()
            df = st.session_state.cat_data
            self.graph_builder.create_graph_part("lesson", df)
            self.graph_builder.create_graph_part("lesson_sequence", df)
            st.success("Lessons created.")
            st.session_state.graph_exists = True

    def _create_keywords(self):
        """Handle creation of Keywords."""
        with st.spinner("Creating Keywords..."):
            if "cat_data" not in st.session_state:
                self._load_data()
            df = st.session_state.cat_data
            self.graph_builder.create_graph_part("keyword", df)
            st.success("Keywords created.")
            st.session_state.graph_exists = True

    def _create_learning_points(self):
        """Handle creation of KeyLearningPoints."""
        with st.spinner("Creating Key Learning Points..."):
            if "cat_data" not in st.session_state:
                self._load_data()
            df = st.session_state.cat_data
            self.graph_builder.create_graph_part("learning_points", df)
            st.success("Key Learning Points created.")
            st.session_state.graph_exists = True
    
    def _create_content_guidance(self):
        """Handle creation of Content Guidance."""
        with st.spinner("Creating Content Guidance..."):
            if "cat_data" not in st.session_state:
                self._load_data()
            df = st.session_state.cat_data
            self.graph_builder.create_graph_part("content_guidance", df)
            st.success("Content Guidance created.")
            st.session_state.graph_exists = True
    
    def _create_slide_content(self):
        """Handle creation of Slide Content."""
        with st.spinner("Creating Slide Content..."):
            if "cat_data" not in st.session_state:
                self._load_data()
            df = st.session_state.cat_data
            self.graph_builder.create_graph_part("slide_content", df)
            st.success("Slide Content created.")
            st.session_state.graph_exists = True
    
    def _create_exit_quiz(self):
        """Handle creation of Exit Quiz."""
        with st.spinner("Creating Exit Quiz..."):
            if "cat_data" not in st.session_state:
                self._load_data()
            df = st.session_state.cat_data
            self.graph_builder.create_graph_part("exit_quiz", df)
            st.success("Exit Quiz created.")
            st.session_state.graph_exists = True
            
    def _create_starter_quiz(self):
        """Handle creation of Starter quiz."""
        with st.spinner("Creating Starter Quiz..."):
            if "cat_data" not in st.session_state:
                self._load_data()
            df = st.session_state.cat_data
            self.graph_builder.create_graph_part("starter_quiz", df)
            st.success("Starter Quiz created.")
            st.session_state.graph_exists = True
            
    def _create_threads(self):
        """Handle creation of Starter quiz."""
        with st.spinner("Creating Threads..."):
            if "cat_data" not in st.session_state:
                self._load_data()
            df = st.session_state.cat_data
            self.graph_builder.create_graph_part("threads", df)
            st.success("Threads created.")
            st.session_state.graph_exists = True
            
    def _create_equip_and_resources(self):
        """Handle creation of Lesson Equipment and Resources."""
        with st.spinner("Creating Lesson Equipment and Resources..."):
            if "cat_data" not in st.session_state:
                self._load_data()
            df = st.session_state.cat_data
            self.graph_builder.create_graph_part("equip_resources", df)
            st.success("Lesson Equipment and Resources created.")
            st.session_state.graph_exists = True
    
    def _create_misconceptions(self):
        """Handle creation of Misconceptions and Common Mistakes."""
        with st.spinner("Creating Misconceptions and Common Mistakes..."):
            if "cat_data" not in st.session_state:
                self._load_data()
            df = st.session_state.cat_data
            self.graph_builder.create_graph_part("misconceptions", df)
            st.success("Misconceptions and Common Mistakes created.")
            st.session_state.graph_exists = True
    
    def _create_pupil_lesson_outcomes(self):
        """Handle creation of Pupil Lesson Outcomes."""
        with st.spinner("Creating Pupil Lesson Outcomes..."):
            if "cat_data" not in st.session_state:
                self._load_data()
            df = st.session_state.cat_data
            self.graph_builder.create_graph_part("pupil_outcomes", df)
            st.success("Pupil Lesson Outcomes created.")
            st.session_state.graph_exists = True
    
    def _create_supervision_levels(self):
        """Handle creation of Supervision Levels."""
        with st.spinner("Creating Supervision Levels..."):
            if "cat_data" not in st.session_state:
                self._load_data()
            df = st.session_state.cat_data
            self.graph_builder.create_graph_part("supervision_levels", df)
            st.success("Supervision Levels created.")
            st.session_state.graph_exists = True
            
    def _create_teacher_tips(self):
        """Handle creation of Teacher Tips."""
        with st.spinner("Creating Teacher Tips..."):
            if "cat_data" not in st.session_state:
                self._load_data()
            df = st.session_state.cat_data
            self.graph_builder.create_graph_part("teacher_tips", df)
            st.success("Teacher Tips created.")
            st.session_state.graph_exists = True

    def _delete_knowledge_graph(self):
        """Delete all nodes and relationships in the graph."""
        with st.spinner("Deleting Knowledge Graph..."):
            self.connection.execute_query("MATCH (n) DETACH DELETE n")
            st.success("Knowledge Graph deleted successfully.")
            st.session_state.graph_exists = False


def main():
    """Main entry point for the application."""
    app = StreamlitApp()
    app.run()


main()