# -*- coding: utf-8 -*-


!pip install -q langchain langchain-google-genai pgvector psycopg2-binary sqlalchemy google-generativeai

# Commented out IPython magic to ensure Python compatibility.
# %pip install --quiet --upgrade langchain-text-splitters langchain-community langgraph

import getpass
import os

api_key = getpass.getpass("Enter your LangSmith API key: ")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = api_key

# Commented out IPython magic to ensure Python compatibility.
# %pip install -qU "langchain[google-genai]"

import os

if not os.environ.get("GOOGLE_API_KEY"):
    api_key = input("Enter your Google Gemini API key: ")
    os.environ["GOOGLE_API_KEY"] = api_key

from langchain.chat_models import init_chat_model

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# Commented out IPython magic to ensure Python compatibility.
# %pip install -qU langchain-google-genai

import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = input("Enter your Google Gemini API key: ")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

from langchain_community.vectorstores.pgvector import PGVector

os.environ["NEON_DB_CONNECTION_STRING"] = "postgresql://neondb_owner:npg_qLz2PnWRG3dI@ep-shy-hill-a8iblc2k-pooler.eastus2.azure.neon.tech/neondb?sslmode=require"
connection_string = os.environ["NEON_DB_CONNECTION_STRING"]

vector_store = PGVector(
    collection_name="documents",
    connection_string=connection_string,
    embedding_function=embeddings,
)

# Commented out IPython magic to ensure Python compatibility.
# %pip install -q pymupdf langchain-community

# Commented out IPython magic to ensure Python compatibility.
# %pip install pypdf

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdfs_with_vessels = [
    ("/content/2022-03-29-08-17-9d9733cf70d8fdddbd30982c04f26530.pdf", "MT Banglar Agragoti"),
    ("/content/voyage-charter-example.pdf", "KAKATUA"),
    ("/content/voyage_charter_contract_bowditch.pdf", "BOWDITCH"),
]

all_documents = []
for file_path, vessel_name in pdfs_with_vessels:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["vessel"] = vessel_name
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_documents.extend(splitter.split_documents(docs))

_ = vector_store.add_documents(documents=all_documents)

from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph


prompt = PromptTemplate.from_template(
    """
    You are a legal assistant extracting direct answers from shipping charter party clauses.

    Given the legal document context and the question, return a **short and specific 1–2 line answer** that:

    - Includes clause numbers if mentioned or applicable.
    - Omits all extra explanation or formatting.
    - Returns "N/A" or "No such clause found." if the information is not present.

    ### CONTEXT:
    {context}

    ### QUESTION:
    {question}

    ### ANSWER:
    """
)


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


from langchain.chat_models import init_chat_model
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
conversation_state = {"pending_question": None, "vessel_name": None}


def check_if_vessel_needed(query):
    instruction = f"""
    You are an assistant helping classify user queries.

    If the following query refers to a vessel, ship, or voyage but does **not** mention a specific vessel name like 'KAKATUA' or 'BOWDITCH', respond with **YES** — meaning vessel name is missing.

    Otherwise, respond with **NO**.

    Query: "{query}"
    """
    response = llm.invoke(instruction).content.strip().lower()
    return response.startswith("yes")

print("Welcome! Type 'exit' or 'quit' to end the session.\n")


while True:
    user_input = input("You: ").strip()


    if user_input.lower() in {"exit", "quit"}:
        print("A: Goodbye!")
        break

    if conversation_state["pending_question"] is None and check_if_vessel_needed(user_input):
        conversation_state["pending_question"] = user_input
        print("A: Can you tell me the vessel name?")


    elif conversation_state["pending_question"]:
        conversation_state["vessel_name"] = user_input.strip()

        full_query = f"{conversation_state['pending_question']} (Vessel: {conversation_state['vessel_name']})"
        result = graph.invoke({"question": full_query})
        print(f"A: {result['answer']}\n")


        conversation_state = {"pending_question": None, "vessel_name": None}

    else:

        result = graph.invoke({"question": user_input})
        print(f"A: {result['answer']}\n")
