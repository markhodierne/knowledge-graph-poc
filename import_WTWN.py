"""
CAT Data Import Module. - Edited to import just the Why This Why Now data for each unit


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
import hashlib


st.set_page_config("Import WTWN Data", page_icon="😺")
st.title("Import WTWN Data to CSV")
    
    
@dataclass
class Config:
    """Application configuration settings."""

    HASURA_URL: str
    DATA_DIR: Path = Path('data')
    CAT_DATA_CSV: Path = DATA_DIR / 'cat_data_wtwn.csv'

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

    @staticmethod
    def generate_sequence_group_id(row: pd.Series) -> str:
        key = f"{row['unitId']}_{row['subjectTitle']}_{row['yearTitle']}_{row['examBoardTitle']}_{row['tierTitle']}"
        return hashlib.md5(key.encode()).hexdigest()


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
    
    def import_data(
        self
    ) -> Optional[pd.DataFrame]:
        """Import CAT data based on selected filters."""
        if not self.client:
            raise RuntimeError("Auth token not set")

        data = self._query_lesson_data()
        if not data:
            return None

        df = pd.DataFrame(data)
        if df.empty:
            return None
        return self._process_dataframe(df)

    def _query_lesson_data(
        self
    ) -> List[Dict]:
        """Query data from Hasura."""

        query = self._build_lesson_query()
        response = self.client.execute_query(query)
        return (response.get('data', {})
                .get('units', []))

        return df
        

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the raw dataframe into final format."""

        df = self.processor.unify_column_types(df)

        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()

        float_cols = df.select_dtypes(include='float').columns
        df[float_cols] = df[float_cols].round(2)
    
        df.drop_duplicates(inplace=True)

        return df


    @staticmethod
    def _build_lesson_query() -> str:
        """Build the GraphQL query for lesson data."""
        return """
            query MyQuery {
                units {
                    why_this_why_now
                    unit_id
                }
            }
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


    def render_import_section(self) -> None:
        """Render import button and logic."""
        if st.button("Import Data"):
            if not st.session_state.id_token:
                st.error("Please enter your Hasura ID token.")
                return

            with st.spinner("Importing data..."):
                st.session_state.current_df = self.importer.import_data()

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