import streamlit as st

st.set_page_config(page_title="Using a KG", page_icon="👋")

st.write('# Knowledge Graph Proof of Concept')
st.markdown("""
    ***The proof of concept aims to demonstrate that incorporating knowledge graphs into RAG retrieval significantly enhances the precision, reasoning, and reliability of information retrieval:*** 

    - Enable the retrieval of specific, contextually relevant information rather than relying solely on unstructured document embeddings
    - Facilitate relational queries that are challenging to perform on plain text or embeddings
    - Support logical reasoning and complex queries, such as "Which lessons are prerequisites for understanding cellular respiration?"
    - Improve the retrieval of conceptually similar content, using the graph's semantic structure
    - Better alignment with user queries, such as retrieving content based on cross-curricular relationships or hierarchical dependencies
    - Improve trust by making the reasoning chain explicit (e.g., "This lesson is related to 'Classification' because it covers 'Photosynthesis,' which involves classifying plant types")
    - Retrieve accurate and domain-relevant information, filtering out irrelevant or ambiguous results that may arise in purely text-based search
    - Reduce risks of hallucinations as the LLM generates content grounded in trusted, structured data
    - Updates to the knowledge graph are dynamic - without requiring retraining or re-indexing
    - Support discovery of hidden relationships and patterns (e.g., lessons that reinforce the same skills or concepts across key stages)
""")
