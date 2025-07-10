"""
Knowledge Concept Extraction from Lesson Plans

This script:
1. Connects to a Neo4j database containing lesson plans
2. Extracts knowledge concepts using an LLM approach
3. Creates concept nodes and relationships in the graph
4. Provides summary of the extraction process
"""
import json
import os
import re
from datetime import datetime
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import streamlit as st
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# Load environment variables
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

client = OpenAI(api_key=OPENAI_API_KEY)
text_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    raise EnvironmentError("Neo4j credentials not set in the environment.")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
st.set_page_config("Build Prior Knowledge", page_icon="📚")


def extract_knowledge_concepts(lesson_content, lesson_title, subject_area, max_concepts=5):
    prompt = f"""
    Analyze this lesson content from the subject area of {subject_area}.
    The lesson title is: "{lesson_title}"
    
    Extract a JSON list of **at most {max_concepts}** knowledge concepts that:
    1. Represent key learning objectives or knowledge components
    2. Use canonical terminology in the field
    3. Include both conceptual understanding and skill-based competencies
    4. Indicate the specificity level (fundamental, intermediate, advanced)
    
    For each concept, provide:
    - concept_name: The canonical term for this concept
    - concept_type: "conceptual" or "procedural"
    - importance: 1-5 rating of centrality to this lesson (1=low, 5=high)
    - definition: Brief definition
    - prerequisites: List of likely prerequisite concepts, ordered from most to least essential for understanding
    
    Return only a valid JSON array of objects. Do not include any text before or after.
    
    Lesson content:
    {lesson_content}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an educational expert specializing in curriculum analysis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    try:
        content = response.choices[0].message.content.strip()
        
        # Log the raw response for debugging
        #st.subheader("LLM Raw Output")
        #st.code(content, language="json")
        
        # Try to extract JSON array from response
        if content.find('[') > -1 and content.find(']') > -1:
            json_content = content[content.find('['):content.rfind(']') + 1]
            concepts = json.loads(json_content)
        else:
            concepts = json.loads(content)
            
        # Truncate the result if needed
        concepts = concepts[:max_concepts]

        # Validate that we got a list of concept dicts
        if not isinstance(concepts, list):
            st.error("LLM did not return a list. Raw content:")
            st.code(content)
            return []

        return concepts
    
    except json.JSONDecodeError as e:
        st.error(f"LLM JSON parsing failed: {e}")
        st.code(content)
        return []
    
def get_all_lessons_with_oak_context(selected_subject):
    """This function returns lesson plan content needed for the prompt.
    The main content is extracted from the Oak content related to a lesson.
    """
    query = """
    MATCH (u:Unit)-[:HAS_VARIANT]->(v:Variant)-[:HAS_LESSON]->(l:Lesson)
    WHERE u.subjectTitle = $selected_subject
    WITH DISTINCT l, u
    OPTIONAL MATCH (l)-[:HAS_KEYWORD]->(kw:LessonKeyword)
    OPTIONAL MATCH (l)-[:HAS_LEARNING_POINT]->(klp:KeyLearningPoint)
    OPTIONAL MATCH (l)-[:HAS_PUPIL_OUTCOME]->(plo:PupilLessonOutcome)
    OPTIONAL MATCH (l)-[:HAS_SLIDE_CONTENT]->(sc:SlideContent)
    OPTIONAL MATCH (l)-[:HAS_EXIT_QUIZ]->(eq:ExitQuiz)

    WITH l, u,
        collect(DISTINCT {keyword: kw.keyword, description: kw.description}) AS keywords,
        collect(DISTINCT klp.keyLearningPoint) AS learning_points,
        collect(DISTINCT plo.pupilLessonOutcome) AS learning_outcomes,
        sc, eq
        

    RETURN 
        l.lessonTitle AS lesson_title,
        l.lessonId AS lesson_id,
        u.subjectTitle AS subject,
        u.unitTitle AS unit_title,
        sc.slideContent AS slide_content,
        eq.exitQuiz AS exit_quiz,
        keywords,
        learning_points,
        learning_outcomes
    """
    with driver.session() as session:
        return session.run(query, selected_subject=selected_subject).data()

def create_concept_nodes_and_relationships(lesson_id, concepts, max_prereqs):
    count = 0
    with driver.session() as session:
        for concept in concepts:
            if not isinstance(concept, dict) or "concept_name" not in concept:
                st.warning(f"Skipping malformed concept: {concept}")
                continue
    
            try:
                raw_name = concept["concept_name"]
                clean_name = re.sub(r"[\"'.,;:!?]+$", "", raw_name).strip()

                session.run("""
                    MERGE (c:Concept {name: $name})
                    SET c.type = $type, c.definition = $definition, c.last_updated = $timestamp
                """, name=clean_name, type=concept["concept_type"],
                    definition=concept["definition"], timestamp=datetime.now().isoformat())

                session.run("""
                    MATCH (lesson:Lesson {lessonId: $lesson_id})
                    MATCH (c:Concept {name: $concept_name})
                    MERGE (lesson)-[r:TEACHES]->(c)
                    SET r.importance = $importance
                """, lesson_id=lesson_id, concept_name=clean_name, importance=concept["importance"])

                for prereq in concept.get("prerequisites", [])[:max_prereqs]:
                    prereq_clean = re.sub(r"[\"'.,;:!?]+$", "", prereq).strip()
                    session.run("""
                        MERGE (p:Concept {name: $prereq})
                        WITH p
                        MATCH (c:Concept {name: $concept})
                        MERGE (c)-[r:REQUIRES]->(p)
                    """, concept=clean_name, prereq=prereq_clean)
                count += 1
            except Neo4jError as e:
                st.error(f"Neo4j error on concept '{concept['concept_name']}': {e}")
    return count

def show_concept_teaching_frequency(driver, limit=50):
    query = """
    MATCH (c:Concept)<-[:TEACHES]-(:Lesson)
    RETURN c.name AS concept, count(*) AS times_taught
    ORDER BY times_taught DESC
    LIMIT $limit
    """
    with driver.session() as session:
        results = session.run(query, limit=limit).data()
        df = pd.DataFrame(results)
        st.markdown(f"#### Top {limit} Concepts by Number of Lessons Teaching Them")
        st.dataframe(df)

def build_knowledge_graph_projection():
    query = """
    CALL gds.graph.project(
        'knowledge_concepts',
        ['Lesson', 'Concept'],
        {
            TEACHES: {
            type: 'TEACHES',
            orientation: 'NATURAL',
            properties: ['importance']
            },
            REQUIRES: {
            type: 'REQUIRES',
            orientation: 'NATURAL'
            }
        }
    )
    """
    with driver.session() as session:
        session.run(query)

def drop_and_rebuild_knowledge_graph(driver):
    with driver.session() as session:
        try:
            session.run("CALL gds.graph.drop('knowledge_concepts', false) YIELD graphName")
            st.info("Dropped old GDS graph projection.")
        except Exception as e:
            st.warning("Graph did not exist or was already clean.")

    build_knowledge_graph_projection()

def generate_fastrp_embeddings(driver, graph_name="knowledge_concepts", embedding_dim=128):
    query = f"""
    CALL gds.fastRP.write(
        '{graph_name}',
        {{
            embeddingDimension: {embedding_dim},
            nodeLabels: ['Concept'],
            relationshipTypes: ['TEACHES', 'REQUIRES'],
            writeProperty: 'conceptEmbedding'
        }}
    )
    YIELD nodePropertiesWritten, configuration
    """
    with driver.session() as session:
        try:
            result = session.run(query)
            data = result.single()
            count = data["nodePropertiesWritten"]
            config_used = data["configuration"]

            st.success(f"Graph embeddings created and stored in Neo4j.")

        except Exception as e:
            st.error(f"Failed to generate FastRP embeddings: {e}")

def run_pagerank_and_display_top_concepts(driver, graph_name="knowledge_concepts", limit=20):
    # Persist PageRank scores to Concept nodes
    query = f"""
    CALL gds.pageRank.write('{graph_name}', {{
        nodeLabels: ['Concept'],
        writeProperty: 'pageRank',
        maxIterations: 20,
        dampingFactor: 0.85
    }})
    YIELD nodePropertiesWritten
    """
    with driver.session() as session:
        result = session.run(query).single()
        count = result["nodePropertiesWritten"]
        st.success(f"✅ PageRank scores written to {count} Concept nodes")

    # Retrieve persisted scores for display
    with driver.session() as session:
        results = session.run("""
        MATCH (c:Concept)
        WHERE c.pageRank IS NOT NULL
        RETURN c.name AS concept, c.pageRank AS score
        ORDER BY score DESC
        """).data()
        df = pd.DataFrame(results)

    # Show top-ranked concepts
    st.markdown("#### 🔝 Top Concepts by PageRank")
    st.dataframe(df.head(limit))

    # Show potential low-importance outliers
    LOW_IMPORTANCE_THRESHOLD = 0.05
    outliers = df[df["score"] < LOW_IMPORTANCE_THRESHOLD].sort_values(by="score", ascending=True).head(20)
    if not outliers.empty:
        st.markdown(f"#### ⚠️ Potential Outlier Concepts (Low PageRank < {LOW_IMPORTANCE_THRESHOLD})")
        st.dataframe(outliers)
    else:
        st.success(f"No outlier concepts detected below PageRank < {LOW_IMPORTANCE_THRESHOLD}")

    return df

def create_text_embeddings():
    with driver.session() as session:
        concepts = session.run("""
            MATCH (c:Concept)
            RETURN c.name AS name
        """).data()

        for row in concepts:
            name = row['name']
            embedding = text_model.encode(name).tolist()
            session.run("""
                MATCH (c:Concept {name: $name})
                SET c.textEmbedding = $embedding
            """, name=name, embedding=embedding)
    st.success("Text embeddings created and stored in Neo4j.")

def create_hybrid_embeddings():
    with driver.session() as session:
        concepts = session.run("""
            MATCH (c:Concept)
            WHERE c.conceptEmbedding IS NOT NULL AND c.textEmbedding IS NOT NULL
            RETURN c.name AS name, c.conceptEmbedding AS graph, c.textEmbedding AS text
        """).data()

        for row in concepts:
            name = row['name']
            hybrid = row['text'] + row['graph']
            session.run("""
                MATCH (c:Concept {name: $name})
                SET c.hybridEmbedding = $embedding
            """, name=name, embedding=hybrid)
    st.success("Hybrid embeddings created and stored in Neo4j.")


def streamlit_ui():
    st.title("Knowledge Concept Graph Builder")
    
    subject_options = ["Biology", "Chemistry", "Combined science", "Physics", "Science"]
    selected_subject = st.selectbox("Select subject to extract concepts from:", subject_options)
    
    lesson_limit = st.number_input(
    label="Enter limit for the number of lessons to process (testing only)",
    min_value=1,       # Prevents zero or negative values
    step=10,             # Increment in steps of 1
    value=500,            # Default starting value
    format="%d"         # Force integer display
)
    # Slider to set max concepts per lesson
    max_concepts = st.number_input(
        "Maximum number of knowledge concepts per lesson",
        min_value=1,
        max_value=15,
        value=5,
        step=1,
        help="This limits the number of concepts extracted and stored per lesson"
    )
    # Slider to set max concepts per lesson
    max_prereqs = st.number_input(
        "Maximum number of prerequisite concepts per concept",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        help="This limits the number of prerequisite concepts extracted and stored per lesson"
    )
    

    if st.button("Extract Concepts from Lessons"):
        lessons = get_all_lessons_with_oak_context(selected_subject)
        
        # 👇 Limit to the first N lessons for testing
        if lesson_limit < len(lessons):
            lessons = lessons[:lesson_limit]
            st.info(f"(Test mode) Using first {len(lessons)} lessons for debugging")
        
        lessons = list({lesson["lesson_id"]: lesson for lesson in lessons}.values())
        st.info(f"Found {len(lessons)} lessons for subject: {selected_subject}")
        
        results = []
        total_concepts = 0

        with st.spinner("Extracting knowledge concepts from lessons..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, lesson in enumerate(lessons):
                progress_bar.progress((idx + 1) / len(lessons))
                status_text.text(f"Processing lesson {idx + 1} of {len(lessons)}")
                
                if "lesson_id" not in lesson:
                    st.warning(f"Skipping lesson due to missing lesson_id: {lesson.get('lesson_title', 'Unknown')}")
                    continue
        
                unit_title = lesson["unit_title"]
                lesson_id = lesson["lesson_id"]
                lesson_title = lesson.get("lesson_title", "Untitled")
                subject = lesson.get("subject", "Unknown")
                slides = lesson.get("slide_content", "")
                exit_quiz = lesson.get("exit_quiz", "")
                keywords = lesson.get("keywords", "")
                learning_points = lesson.get("learning_points", "")
                learning_outcomes = lesson.get("learning_outcomes", "")
                
                # Skip empty lessons with no content
                if not any([learning_points, learning_outcomes, keywords, slides]):
                    st.warning(f"Lesson {lesson_id} has missing content. Skipping.")
                    results.append({
                        "lesson_id": lesson_id,
                        "lesson_title": lesson_title,
                        "subject": subject,
                        "concepts_extracted": 0,
                        "concepts_total": 0,
                        "status": "skipped (missing content)"
                    })
                    continue
                
                # 👇 Construct lesson_content from all relevant pieces
                lesson_content = "\n\n".join(filter(None, [
                    f"Subject: {subject}",
                    f"Unit Title: {unit_title}",
                    f"Lesson Title: {lesson_title}",
                    f"Key Learning Points:\n" + "\n".join(learning_points),
                    f"Pupil Learning Outcomes:\n" + "\n".join(learning_outcomes),
                    f"Keywords:\n" + "\n".join([f"- {kw['keyword']}: {kw['description']}" for kw in keywords]),
                    f"Exit Quiz: {exit_quiz}",
                    f"Lesson Content: {slides}"
                ]))

                with driver.session() as session:
                    existing_concepts = session.run("""
                        MATCH (lesson:Lesson {lessonId: $lesson_id})-[:TEACHES]->(c:Concept)
                        RETURN count(c) AS count
                    """, lesson_id=lesson_id).single()["count"]

                if existing_concepts > 0:
                    results.append({
                        "lesson_id": lesson_id,
                        "lesson_title": lesson_title,
                        "subject": subject,
                        "concepts_extracted": 0,
                        "concepts_total": existing_concepts,
                        "status": "skipped"
                    })
                    continue

                try:
                    concepts = extract_knowledge_concepts(lesson_content, lesson_title, subject, max_concepts=max_concepts)
                    added = create_concept_nodes_and_relationships(lesson_id, concepts, max_prereqs=max_prereqs)
                    total_concepts += added

                    results.append({
                        "lesson_id": lesson_id,
                        "lesson_title": lesson_title,
                        "subject": subject,
                        "concepts_extracted": added,
                        "concepts_total": added,
                        "status": "success"
                    })
                    
                except Exception as e:
                    results.append({
                        "lesson_id": lesson_id,
                        "lesson_title": lesson_title,
                        "subject": subject,
                        "concepts_extracted": 0,
                        "concepts_total": 0,
                        "status": f"error: {e}"
                    })
                
        progress_bar.empty()
        status_text.text("✅ All lessons processed.")

        df = pd.DataFrame(results)
        st.dataframe(df)

        df.to_csv(f"knowledge_extraction_summary.csv", index=False)
        st.success("Summary saved to knowledge_extraction_summary.csv")

        success_df = df[df["status"] == "success"]
        subject_summary = success_df.groupby("subject").agg(
            concepts_extracted=("concepts_extracted", "sum"),
            lesson_count=("lesson_id", "count")
        )
        subject_summary["avg_concepts_per_lesson"] = (
            subject_summary["concepts_extracted"] / subject_summary["lesson_count"]
        )

        st.subheader("Concepts per Subject")
        st.dataframe(subject_summary.sort_values(by="concepts_extracted", ascending=False))
    
    if st.button("Show Teaching Frequency of Concepts"):
        try:
            show_concept_teaching_frequency(driver)
        except Exception as e:
            st.error(f"Failed to calculate teaching frequency: {e}")

    if st.button("Build GDS Graph Projection"):
        try:
            drop_and_rebuild_knowledge_graph(driver)
            st.success("Fresh GDS projection created.")
        except Exception as e:
            st.error(f"Failed to create GDS projection: {e}")

    if st.button("Run PageRank and Show Top Concepts"):
        try:
            run_pagerank_and_display_top_concepts(driver)
        except Exception as e:
            st.error(f"Failed to run PageRank: {e}")

    if st.button("Generate Concept Embeddings (FastRP)"):
        try:
            generate_fastrp_embeddings(driver)
        except Exception as e:
            st.error(f"Failed to generate embeddings: {e}")

    if st.button("Generate Text Embeddings from Concept Names"):
        try:
            create_text_embeddings()
        except Exception as e:
            st.error(f"Failed to create text embeddings: {e}")

    if st.button("Generate Hybrid Embeddings (Graph + Text)"):
        try:
            create_hybrid_embeddings()
        except Exception as e:
            st.error(f"Failed to create hybrid embeddings: {e}")

try:
    streamlit_ui()
finally:
    driver.close()