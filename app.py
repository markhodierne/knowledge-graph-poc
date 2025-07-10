import streamlit as st


pages = {
    "Home": [
        st.Page("welcome.py", title="Welcome", icon="👋"),
    ],
    "Load KG": [
        st.Page("import_cat_data.py", title="Import CAT Lesson Data to CSV", icon="✅"),
        st.Page("import_cat_unit_data.py", title="Import CAT Unit Data to CSV", icon="✅"),
        st.Page("add_cat_lesson_data_to_kg.py", title="Add CAT Lesson Data CSV to KG", icon="✅"),
        st.Page("add_cat_unit_data_to_kg.py", title="Add CAT Unit Data to KG", icon="✅"),
        st.Page("add_aila_data_to_kg.py", title="Add AILA Lesson Plans to KG", icon="✅"),
        st.Page("add_nc_data_to_kg.py", title="Add NC Data to KG", icon="✅")
    ],
    "Embeddings": [  
        st.Page("kg_embeddings_LP_OpenAI.py", title="Embeddings - Lesson Plans (OpenAI)", icon="✅"),
        st.Page("kg_embeddings_LP_FastRP.py", title="Embeddings - Lesson Plans (FastRP)", icon="✅"),
        st.Page("kg_embeddings_OpenAI.py", title="Embeddings - Other Nodes (OpenAI)", icon="✅")
    ],
    "Graph RAG": [  
        st.Page("top_match_lesson_plans.py", title="Top Matching Lesson Plans", icon="✅")
    ],
    "KG Queries": [
        st.Page("create_prior-knowledge.py", title="Build Prior Knowledge and Relationships", icon="✅"),
        st.Page("knowledge_similarity.py", title="Knowledge Concept Similarity", icon="✅"),
        st.Page("prior_knowledge_query.py", title="Prior Knowledge Query", icon="✅"),
        st.Page("top_match_topics.py", title="National Curriculum Mapping", icon="✅")
    ]
}

pg = st.navigation(pages)
pg.run()
