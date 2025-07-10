"""
CAT Data Import Module.

This module provides functionality to import CAT (Curriculum and 
Assessment Tool) data from a Hasura GraphQL API and save it to CSV 
format.
"""

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv


st.set_page_config("Import Lesson Data", page_icon="😺")
st.title("Import CAT Lesson Data to CSV")
    
    
@dataclass
class Config:
    """Application configuration settings."""

    HASURA_URL: str
    DATA_DIR: Path = Path('data')
    CAT_DATA_CSV: Path = DATA_DIR / 'cat_data.csv'
    SLIDE_DATA_CSV: Path = DATA_DIR / 'slide_cycle_data_final.csv'

    @classmethod
    def from_env(cls) -> 'Config':
        """Create config from environment variables."""
        load_dotenv()
        hasura_url = os.getenv('HASURA_URL')
        if not hasura_url:
            raise EnvironmentError("HASURA_URL environment variable not set")
        return cls(HASURA_URL=hasura_url)


class HasuraClient:
    """Client for interacting with Hasura GraphQL API."""

    def __init__(self, url: str, token: str):
        """Initialize client with API URL and auth token."""
        self.url = url
        self.headers = {
            'Content-Type': 'application/json',
            'Hasura-Client-Name': 'hasura-console',
            'hasura-collaborator-token': token.strip()
        }

    def execute_query(
        self, 
        query: str, 
        variables: Optional[Dict] = None
    ) -> Dict:
        """Execute GraphQL query and return response."""
        try:
            response = requests.post(
                self.url,
                json={'query': query, 
                'variables': variables},
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"API Request Error: {e}")
            return {}


class DataProcessor:
    """Process and transform CAT data."""

    @staticmethod
    def unify_column_types(df: pd.DataFrame) -> pd.DataFrame:
        """Convert list/dict columns to canonical JSON strings."""
        json_cols = [
            col for col in df.columns
            if df[col].apply(lambda x: isinstance(x, (list, dict))).any()
        ]
        for col in json_cols:
            df[col] = df[col].apply(
                lambda x: (
                    json.dumps(x, sort_keys=True) if isinstance(x, (list, dict))
                    else str(x) if pd.notna(x) else None
                )
            )
        return df


class CATDataImporter:
    """Main class for importing CAT data."""

    def __init__(self, config: Config):
        """Initialize importer with config."""
        self.config = config
        self.processor = DataProcessor()
        self.client: Optional[HasuraClient] = None

    def set_auth_token(self, token: str) -> None:
        """Set the Hasura authentication token."""
        self.client = HasuraClient(self.config.HASURA_URL, token)

    def get_subject_list(self) -> List:
        """Return the list of available subjects."""
        return [
            "Biology", "Chemistry", "Combined Science", "English", "Geography",
            "History", "Maths", "Physics", "Science"
        ]

    def get_key_stage_list(self) -> List:
        """Return the list of key stages."""
        return ["KS1", "KS2", "KS3", "KS4"]

    def get_year_list(self) -> List:
        """Return the list of years."""
        return [
            "Year 1", "Year 2", "Year 3", "Year 4", "Year 5", "Year 6",
            "Year 7", "Year 8", "Year 9", "Year 10", "Year 11"
        ]
    
    def import_data(
        self,
        subjects: List[str],
        key_stages: Optional[List[str]] = None,
        years: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """Import CAT data based on selected filters."""
        if not self.client:
            raise RuntimeError("Auth token not set")

        subject_slugs = [s.lower().replace(" ", "-") for s in subjects]
        ks_slugs = (
            [ks.lower().replace(" ", "-") for ks in key_stages]
            if key_stages else None
        )
        year_slugs = (
            [yr.lower().replace(" ", "-") for yr in years]
            if years else None
        )
        data = self._query_lesson_data(subject_slugs, ks_slugs, year_slugs)
        if not data:
            return None

        df = pd.DataFrame(data)
        if df.empty:
            return None
        return self._process_dataframe(df)

    def _query_lesson_data(
        self,
        subjects: List[str],
        key_stages: Optional[List[str]],
        years: Optional[List[str]]
    ) -> List[Dict]:
        """Query lesson data from Hasura."""
        subjects = ', '.join(f'"{sub}"' for sub in subjects)
        filters = []
        if key_stages:
            filters.append(f'keyStageSlug: {{_in: [{",".join(key_stages)}]}}')
        if years:
            filters.append(f'yearSlug: {{_in: [{",".join(years)}]}}')

        query = self._build_lesson_query(subjects, ', '.join(filters))
        response = self.client.execute_query(query)
        return (response.get('data', {})
                .get('published_mv_lesson_openapi_1_2_3', []))
        
    def _df_fixes(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[
            (df["unitVariantId"] == 1080)
            & (df["lessonId"] == 5048),
            "lessonOrderInUnit"
        ] = 6
        df.loc[
            (df["unitVariantId"] == 1080)
            & (df["lessonId"] == 5049),
            "lessonOrderInUnit"
        ] = 7
        df.loc[
            (df["unitVariantId"] == 1080)
            & (df["lessonId"] == 5050),
            "lessonOrderInUnit"
        ] = 8
        df.loc[
            (df["unitVariantId"] == 1080)
            & (df["lessonId"] == 5051),
            "lessonOrderInUnit"
        ] = 10
        df.loc[
            (df["unitVariantId"] == 1080)
            & (df["lessonId"] == 8141),
            "lessonOrderInUnit"
        ] = 11
        df.loc[
            (df["unitVariantId"] == 1080)
            & (df["lessonId"] == 8142),
            "lessonOrderInUnit"
        ] = 12

        return df
        

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the raw dataframe into final format."""
        df.rename(columns={'examBoard': 'examBoardTitle'}, inplace=True)
        df['programmeSlug'] = df['programmeSlug'].str.replace(
            'combined-science', 
            'combinedscience', 
            regex=False
        )
        df['programmeSlug'] = df['programmeSlug'].astype(str)
        df['phaseTitle'] = (
            df['programmeSlug']
            .str.split('-', expand=True)[1]
            .str.capitalize()
        )
        df['examBoardTitle'] = df['examBoardTitle'].fillna("NoBoard")

        df = self._add_unit_id(df)
        df.drop(columns=['programmeSlug', 'unitSlug'], inplace=True)
        df = self.processor.unify_column_types(df)
        
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()

        float_cols = df.select_dtypes(include='float').columns
        df[float_cols] = df[float_cols].round(2)
    
        df.drop_duplicates(inplace=True)
        
        df = self._df_fixes(df)
        df = self._add_thread_data(df)
        df = self._add_slide_data(df)

        return df
    
    def _build_unit_query(self) -> str:
        """Build the GraphQL query for unit data."""
        return """
            query MyQuery {
                published_mv_openapi_unit_curriculum_content_1_0_2 {
                    unitId
                    unitSlug
                }
            }
        """

    def _build_thread_query(self) -> str:
        """Build the GraphQL query for thread data."""
        return """
            query MyQuery {
                threads_aggregate {
                    nodes {
                        title
                        thread_id
                        slug
                        thread_units {
                            unit_id
                        }
                    }
                }
            }
        """

    def _process_thread_data(self, thread_data: Dict) -> pd.DataFrame:
        """Process thread data into a flattened DataFrame."""
        flattened_data = []
        for node in thread_data['data']['threads_aggregate']['nodes']:
            for thread_unit in node['thread_units']:
                flattened_data.append({
                    'threadTitle': node['title'],
                    'threadId': node['thread_id'],
                    'unitId': thread_unit['unit_id']
                })
        return pd.DataFrame(flattened_data)
    
    def _add_unit_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add unitId to the dataframe."""
        unit_response = self.client.execute_query(self._build_unit_query())
        if not unit_response or 'data' not in unit_response:
            return df
        unit_data = unit_response['data'][
            'published_mv_openapi_unit_curriculum_content_1_0_2'
        ]
        lookup_df = pd.DataFrame(unit_data)
        
        return df.merge(lookup_df, on='unitSlug', how='left')

    def _add_thread_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add thread information to the dataframe."""
        thread_response = self.client.execute_query(self._build_thread_query())
        if not thread_response or 'data' not in thread_response:
            return df
        df_threads = self._process_thread_data(thread_response)

        return df.merge(df_threads, on='unitId', how='left')
    
    def _add_slide_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add slide information to the dataframe."""
        filename = self.config.SLIDE_DATA_CSV
        file_path = Path(filename)
        if not file_path.exists():
            return df
    
        df_slides = pd.read_csv(filename)
        df_slides = df_slides[['lessonId', 'slideContent']]
        
        return df.merge(df_slides, on='lessonId', how='left')

    @staticmethod
    def _build_lesson_query(subjects: List[str], filters: str) -> str:
        """Build the GraphQL query for lesson data."""
        return f"""
            query MyQuery {{
                published_mv_lesson_openapi_1_2_3(
                    where: {{
                        subjectSlug: {{ _in: [{subjects}] }},
                        {filters},
                        isLegacy: {{ _eq: false }}
                    }}
                ) {{
                    contentGuidance
                    examBoardTitle
                    exitQuiz
                    exitQuizId
                    keyLearningPoints
                    keyStageTitle
                    lessonCohort
                    lessonEquipmentAndResources
                    lessonId
                    lessonKeywords
                    lessonOrderInUnit
                    lessonTitle
                    misconceptionsAndCommonMistakes
                    programmeSlug
                    pupilLessonOutcome
                    starterQuiz
                    starterQuizId
                    subjectTitle
                    supervisionLevel
                    teacherTips
                    tierTitle
                    unitOrder
                    unitSlug
                    unitTitle
                    unitVariantId
                    yearTitle
                }}
            }}
        """


class StreamlitInterface:
    """Streamlit UI for CAT data import."""

    def __init__(self, importer: CATDataImporter):
        """Initialize the StreamlitInterface."""
        self.importer = importer
        self.initialize_session_state()
        # Re-initialize client if token exists but client doesn't
        if st.session_state.get('id_token') and not self.importer.client:
            self.importer.set_auth_token(st.session_state.id_token)

    @staticmethod
    def initialize_session_state() -> None:
        """Initialize Streamlit session state variables."""
        defaults = {
            'id_token': '',
            'subject_selection': [],
            'key_stage_selection': [],
            'year_selection': [],
            'save_success': False,
            'current_df': None,
            'filter_choice': "Key Stage"
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def render_ui(self) -> None:
        """Render the main UI components."""
        self.render_auth_section()
        self.render_filter_section()
        self.render_import_section()
        self.render_results_section()

    def render_auth_section(self) -> None:
        """Render authentication section."""
        id_token = st.text_input(
            "Enter your Hasura ID token",
            type="password",
            value=st.session_state.id_token
        )
        if st.session_state.id_token != id_token:
            st.session_state.id_token = id_token
            if id_token:
                self.importer.set_auth_token(id_token)
                st.success("Token set successfully!")
            st.session_state.current_df = None

    def render_filter_section(self) -> None:
        """Render data filter section."""
        subject_selection = st.multiselect(
            "Filter by Subject (select one or more)",
            self.importer.get_subject_list()
        )
        if st.session_state.subject_selection != subject_selection:
            st.session_state.subject_selection = subject_selection
            st.session_state.current_df = None

        filter_choice = st.radio(
            "Filter by Key Stage or Year:", ["Key Stage", "Year"]
        )
        if st.session_state.filter_choice != filter_choice:
            st.session_state.filter_choice = filter_choice
            st.session_state.current_df = None

        if filter_choice == "Key Stage":
            st.session_state.key_stage_selection = st.multiselect(
                "Filter by Key Stage (select one or more)",
                self.importer.get_key_stage_list()
            )
        else:
            st.session_state.year_selection = st.multiselect(
                "Filter by Year (select one or more)", 
                self.importer.get_year_list()
            )

    def render_import_section(self) -> None:
        """Render import button and logic."""
        if st.button("Import Data"):
            if not st.session_state.id_token:
                st.error("Please enter your Hasura ID token.")
                return

            if not st.session_state.subject_selection:
                st.error("Please select at least one subject.")
                return

            with st.spinner("Importing data..."):
                st.session_state.current_df = self.importer.import_data(
                    st.session_state.subject_selection,
                    st.session_state.key_stage_selection,
                    st.session_state.year_selection
                )

    def render_results_section(self) -> None:
        """Render results and save options."""
        if st.session_state.current_df is not None:
            if not st.session_state.current_df.empty:
                st.write("Data Preview:")
                st.dataframe(st.session_state.current_df.head())

                if st.button("Save to CSV"):
                    try:
                        filename = self.importer.config.CAT_DATA_CSV

                        with st.spinner("Saving the CSV file..."):
                            deduplicated_df = st.session_state.current_df.drop_duplicates()
                            deduplicated_df.to_csv(
                                filename,
                                index=False,
                                encoding='utf-8'
                            )

                        st.session_state.save_success = True
                        st.success(f"Data saved to {filename}")
                    except Exception as e:
                        st.error(f"Error saving data: {str(e)}")
            else:
                st.warning("No data retrieved. Please refine your selections.")
        else:
            st.warning("No data imported yet. Please import data first.")


def main():
    """Main entry point for the application."""
    config = Config.from_env()
    importer = CATDataImporter(config)
    interface = StreamlitInterface(importer)
    interface.render_ui()


main()