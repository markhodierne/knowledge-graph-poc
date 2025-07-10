import os
import numpy as np

from dotenv import load_dotenv
import streamlit as st
import tiktoken
import openai
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from trulens_eval import Tru
from trulens.apps.custom import instrument
from trulens.core import TruSession
from trulens.core import Feedback
from trulens.core import Select
from trulens.providers.openai import OpenAI
from trulens.apps.custom import TruCustomApp

CHROMA_DIR = "chroma_data/"


st.set_page_config(page_title="RAG Triad", page_icon="👋")
st.title("RAGAs Evaluations")

session = TruSession()
session.reset_database()

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY is not set in the environment.")

#openai.api_key = openai_api_key
provider = OpenAI(model_engine="gpt-4o")

# Initialise embedding function
embedding_function = OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-ada-002"
)


class RAG:
    def __init__(self, collection):
        self.collection = collection

    def _query_chroma(self, query_text: str, n_results=4):
        query_embedding = embedding_function([query_text])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

    @instrument
    def retrieve(self, query: str) -> list:
        """
        Retrieve relevant text from vector store.
        """
        def flatten(nested_list):
            """
            Recursive function that flattens any level of nested lists 
            into a single list.
            """
            flat_list = []
            for item in nested_list:
                if isinstance(item, list):
                    flat_list.extend(flatten(item))
                else:
                    flat_list.append(item)
            return flat_list

        results = self._query_chroma(query_text=query, n_results=4)
        all_docs = list(flatten(results["documents"]))
        context_str = "\n\n".join(all_docs)
        return context_str

    @instrument
    def generate_completion(self, query: str, context_str: str, token_limit=4096) -> str:
        """
        Generate answer from context.
        """
        def count_tokens(text, model="gpt-4o"):
            tokenizer = tiktoken.encoding_for_model(model)
            return len(tokenizer.encode(text))
        
        def truncate_context(context, max_tokens, model="gpt-4o"):
            tokenizer = tiktoken.encoding_for_model(model)
            tokens = tokenizer.encode(context)
            truncated_tokens = tokens[:max_tokens]
            return tokenizer.decode(truncated_tokens)
        
        if not context_str:
            return "Sorry, I couldn't find an answer to your question."
        
        # Define the prompt
        prompt = (
            f"You are an expert assistant. Use only the information provided below to answer the question. "
            f"If the information is insufficient to provide a complete answer, respond with: "
            f"'I don't know based on the provided context.'\n\n"
            f"---------------------\n"
            f"Context:\n{context_str}\n"
            f"---------------------\n"
            f"Question: {query}\n"
            f"Answer:"
        )

        # Count tokens for the context and prompt
        context_tokens = count_tokens(context_str, model="gpt-4o")
        prompt_tokens = count_tokens(prompt, model="gpt-4o")
        total_tokens = context_tokens + prompt_tokens

        # Log token information
        st.write(f"Context tokens: {context_tokens}")
        st.write(f"Prompt tokens: {prompt_tokens}")
        st.write(f"Total tokens: {total_tokens}")

        # Check if the total tokens exceed the limit
        if total_tokens > token_limit:
            st.warning(f"Token limit exceeded ({total_tokens} > {token_limit}). Truncating context...")
            # Reduce the context to fit within the token limit
            available_tokens = token_limit - prompt_tokens
            context_str = truncate_context(context_str, available_tokens, model="gpt-4o")
            st.write(f"Truncated context tokens: {count_tokens(context_str, model='gpt-4o')}")

        # Generate the completion
        try:
                completion = (
                    openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=0
                    ).choices[0].message.content
                )
                return completion if completion else "Did not find an answer."
        except Exception as e:
            st.error(f"Error generating completion: {str(e)}")
            return "An error occurred while generating the answer."

    @instrument
    def query(self, query: str) -> str:
        context_str = self.retrieve(query=query)
        completion = self.generate_completion(
            query=query, context_str=context_str
        )
        return completion


def load_existing_collection(path):
    # Initialize Chroma Client
    try:
        chroma_client = chromadb.PersistentClient(path=path)
    except Exception as e:
        st.error(f"Failed to initialize Chroma client: {e}")
        st.stop()

    # Load Collection
    try:
        lesson_plans_collection = chroma_client.get_collection("lesson_plans")
        if lesson_plans_collection.count() == 0:
            st.error("The collection is empty. Please populate it before querying.")
            st.stop()
        else:
            return lesson_plans_collection
    except Exception as e:
        st.error(f"Collection not found: {e}")
        st.stop()


# Define a groundedness feedback function
f_groundedness = (
    Feedback(
        provider.groundedness_measure_with_cot_reasons, name="Groundedness"
    )
    .on(Select.RecordCalls.retrieve.rets.collect())
    .on_output()
)
# Question/answer relevance between overall question and answer.
f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
    .on_input()
    .on_output()
)

# Context relevance between question and each context chunk.
f_context_relevance = (
    Feedback(
        provider.context_relevance_with_cot_reasons, name="Context Relevance"
    )
    .on_input()
    .on(Select.RecordCalls.retrieve.rets[:])
    .aggregate(np.mean)  # choose a different aggregation method if you wish
)

lesson_plans_collection = load_existing_collection(CHROMA_DIR)
rag = RAG(collection=lesson_plans_collection)

# Initialize Tru and TruCustomApp
tru = Tru()
tru.reset_database()
tru_rag = TruCustomApp(
    rag,
    app_name="RAG",
    app_version="base",
    #feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
    feedbacks=[f_groundedness, f_answer_relevance],
)

# Use the recorder context manager and collect all queries
with tru_rag as recording:
    # Query the RAG model
    answer1 = rag.query(
        "Are bacteria multi cellular or not?"
    )
    answer2 = rag.query(
        "What themes in Chemistry occur over several Key Stages?"
    )
    answer3 = rag.query(
        "What are examples of Shakespeare's historical plays?"
    )

# Process and display records with feedback
try:
    records = recording.records
    for i, record in enumerate(records):
        st.write(f"\n**Query {i+1} Details:**")
        st.write(f"- Query: {record.main_input}")
        st.write(f"- Response: {record.main_output}")
        
        # Wait for feedback results to be available
        try:
            # Get feedback results
            if hasattr(record, 'feedback_results'):
                feedback_futures = record.feedback_results
                if feedback_futures:
                    st.write("**Feedback Results:**")
                    # Wait for each feedback future to complete
                    record.wait_for_feedback_results()
                    # Access the completed feedback results
                    for future in feedback_futures:
                        if hasattr(future, 'result'):
                            result = future.result()
                            #st.write(result)
                            st.write(f"{result.name}: {result.result}")
            st.write("---")
            
        except Exception as e:
            st.write(f"Error processing feedback for record {i+1}: {str(e)}")
            
except Exception as e:
    st.error(f"Error processing records: {str(e)}")

st.write("\n\nProcessing complete.")