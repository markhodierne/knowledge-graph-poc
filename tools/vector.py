import streamlit as st
from llm import llm, embeddings
from graph import graph

# Create the Neo4jVector
from langchain_community.vectorstores.neo4j_vector import Neo4jVector

neo4jvector = Neo4jVector.from_existing_index(  
    embeddings,                                         # (1)
    graph=graph,                                        # (2)
    index_name="Lesson",                                # (3)
    node_label="Lesson",                                # (4)
    text_node_property="lessonContent",                 # (5)
    embedding_node_property="lessonContentEmbedding",   # (6)
    retrieval_query="""
MATCH (node:Lesson)
WHERE node.lessonContent IS NOT NULL
RETURN
    node.lessonContent AS text,
    score,
    {
        title: node.title,
        keyStage: node.keyStage,
        subject: node.subject,
        curriculumLinks: node.curriculumLinks
    } AS metadata
ORDER BY score DESC
"""
)

# Create the retriever
retriever = neo4jvector.as_retriever()

# Create the prompt
from langchain_core.prompts import ChatPromptTemplate

instructions = (
    "Use the given context about the UK National Curriculum to answer the question."
    "Provide your answer as plain text."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)

# Create the chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

question_answer_chain = create_stuff_documents_chain(llm, prompt)
curriculum_retriever = create_retrieval_chain(
    retriever, 
    question_answer_chain
)

# Create a function to call the chain
def get_lesson_info(input):
    return curriculum_retriever.invoke({"input": input})