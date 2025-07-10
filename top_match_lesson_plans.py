import os
import json

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

st.set_page_config(page_title="Matching Lesson Plans", page_icon="🔎")
st.title("Top Matching Lesson Plans")

if 'user_subject' not in st.session_state:
    st.session_state.user_subject = None
if 'user_key_stage' not in st.session_state:
    st.session_state.user_key_stage = None

st.markdown("#### I'm Aila, Oak's AI lesson assistant. What do you want to teach?")

# User inputs
subject_list = [
    "Biology", "Chemistry", "Combined Science", "Physics", "Science"
]
user_subject = st.selectbox(
    "Select Subject:",
    subject_list
)
if st.session_state.user_subject != user_subject:
    st.session_state.user_subject = user_subject

key_stage_list = [
    "Key Stage 1", "Key Stage 2", "Key Stage 3", "Key Stage 4"
]
user_key_stage = st.selectbox(
    "Select Key Stage:",
    key_stage_list
)
if st.session_state.user_key_stage != user_key_stage:
    st.session_state.user_key_stage = user_key_stage

user_title = st.text_input(
    label="Enter lesson title:",
    placeholder=""
).strip()

# Filtering mode
filter_mode = st.selectbox(
    "Select Filtering Mode:",
    ["Off", "Pre-filter", "Post-filter"]
)

# Allow user to select metadata fields for filtering
metadata_filters = {}
if filter_mode != "Off":
    if st.checkbox("Key Stage"):
        metadata_filters["keyStageTitle"] = user_key_stage
    if st.checkbox("Subject"):
        metadata_filters["subjectTitle"] = user_subject


def get_embedding(text):
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding


# Function to format display values
def format_subject_display(subject_value):
    """Convert stored subject format to display format"""
    subject_mapping = {
        "biology": "Biology",
        "chemistry": "Chemistry", 
        "combined-science": "Combined Science",
        "physics": "Physics",
        "science": "Science"
    }
    return subject_mapping.get(subject_value, subject_value.title())


def format_key_stage_display(key_stage_value):
    """Convert stored key stage format to display format"""
    key_stage_mapping = {
        "key-stage-1": "Key Stage 1",
        "key-stage-2": "Key Stage 2",
        "key-stage-3": "Key Stage 3", 
        "key-stage-4": "Key Stage 4"
    }
    return key_stage_mapping.get(key_stage_value, key_stage_value.replace("-", " ").title())


# Helper functions to convert between UI format and JSON format
def convert_subject_to_json_format(subject):
    """Convert UI subject format to JSON format stored in content"""
    mapping = {
        "Biology": "biology",
        "Chemistry": "chemistry", 
        "Combined Science": "combined-science",
        "Physics": "physics",
        "Science": "science"
    }
    return mapping.get(subject, subject.lower().replace(" ", "-"))


def convert_key_stage_to_json_format(key_stage):
    """Convert UI key stage format to JSON format stored in content"""
    return key_stage.lower().replace(" ", "-")


def convert_subject_from_json_format(subject):
    """Convert JSON subject format to display format"""
    mapping = {
        "biology": "Biology",
        "chemistry": "Chemistry", 
        "combined-science": "Combined Science",
        "physics": "Physics",
        "science": "Science"
    }
    return mapping.get(subject, subject.title())


def convert_key_stage_from_json_format(key_stage):
    """Convert JSON key stage format to display format"""
    return key_stage.replace("-", " ").title()


# Function to search lesson plans in Neo4j
def vector_search_lessons(query_text, filters=None, mode="Off", top_k=5):
    # Get embedding for the query
    query_embedding = get_embedding(query_text)

    if mode == "Pre-filter" and filters:
        # Pre-filtering: Use JSON extraction in Cypher WHERE clause
        # Since subjectTitle/keyStageTitle aren't directly accessible, 
        # we need to extract from JSON content
        
        # Increase top_k for pre-filtering as it's less efficient
        search_top_k = max(top_k * 3, 50)
        
        query = """
        CALL db.index.vector.queryNodes(
            'lesson_content_embedding_index',
            $search_top_k,
            $query_embedding
        ) YIELD node, score
        WHERE """
        
        # Build WHERE conditions for JSON content filtering
        where_conditions = []
        query_params = {
            "query_embedding": query_embedding,
            "search_top_k": search_top_k,
            "top_k": top_k
        }
        
        if "subjectTitle" in filters:
            subject_json = convert_subject_to_json_format(filters["subjectTitle"])
            where_conditions.append('apoc.convert.fromJsonMap(node.content).subject = $subject_filter')
            query_params["subject_filter"] = subject_json
            
        if "keyStageTitle" in filters:
            key_stage_json = convert_key_stage_to_json_format(filters["keyStageTitle"])
            where_conditions.append('apoc.convert.fromJsonMap(node.content).keyStage = $key_stage_filter')
            query_params["key_stage_filter"] = key_stage_json
        
        query += " AND ".join(where_conditions)
        query += """
        RETURN 
            node AS node,
            node.lessonPlanId AS lesson_plan_id,
            node.content AS content,
            score
        ORDER BY score DESC
        LIMIT toInteger($top_k)
        """
        
    else:
        # No pre-filtering or post-filtering mode
        query = """
        CALL db.index.vector.queryNodes(
            'lesson_content_embedding_index',
            $top_k,
            $query_embedding
        ) YIELD node, score
        RETURN 
            node AS node,
            node.lessonPlanId AS lesson_plan_id,
            node.content AS content,
            score
        ORDER BY score DESC
        """
        
        query_params = {
            "query_embedding": query_embedding,
            "top_k": top_k
        }

    try:
        # Run the Neo4j query
        results = neo4j_client.run_query(query, query_params)

        # Convert results to list of dictionaries
        lesson_plans = []
        for record in results:
            try:
                content = json.loads(record.get('content', '{}'))
                lesson_plan = {
                    "lesson_plan_id": record.get('lesson_plan_id', 'N/A'),
                    "title": content.get('title', 'N/A'),
                    "subject": content.get('subject', 'N/A'),
                    "key_stage": content.get('keyStage', 'N/A'),
                    "score": record.get('score', 0)
                }
                lesson_plans.append(lesson_plan)
            except json.JSONDecodeError:
                st.error(f"Could not parse content for lesson plan: {record.get('lesson_plan_id')}")

        # Apply post-filtering if selected
        if mode == "Post-filter" and filters:
            filtered_plans = []
            for plan in lesson_plans:
                include = True
                
                if "subjectTitle" in filters:
                    expected_subject = convert_subject_to_json_format(filters["subjectTitle"])
                    if plan.get('subject') != expected_subject:
                        include = False
                        
                if "keyStageTitle" in filters and include:
                    expected_key_stage = convert_key_stage_to_json_format(filters["keyStageTitle"])
                    if plan.get('key_stage') != expected_key_stage:
                        include = False
                
                if include:
                    filtered_plans.append(plan)
            
            lesson_plans = filtered_plans[:top_k]  # Limit results after filtering

        return lesson_plans

    except Exception as e:
        st.error(f"Error in vector search: {e}")
        import traceback
        st.error(traceback.format_exc())
        return []


# Handle search
def vector_search():
    if st.button("Search for Lesson Plans"):
        if not user_title.strip():
            st.warning("Please enter a lesson title.")
        else:
            with st.spinner("Searching for matching lesson plans..."):                
                # Fetch results from Neo4j
                lesson_plans = vector_search_lessons(user_title, metadata_filters, filter_mode)
            
            # Display results
            if lesson_plans:
                st.markdown("### Results:")
                for i, plan in enumerate(lesson_plans, start=1):
                    st.markdown(f"**{i}. Lesson Plan:** {plan['title']}")
                    st.write(f"**Lesson ID:** {plan['lesson_plan_id']}")
                    st.write(f"**Subject:** {convert_subject_from_json_format(plan['subject'])}")
                    st.write(f"**Key Stage:** {convert_key_stage_from_json_format(plan['key_stage'])}")
                    st.write(f"**Score:** {plan['score']:.4f}")
                    st.write("---")
            else:
                st.warning("No matching lesson plans found. Try refining your input.")


vector_search()
neo4j_client.close()
