from pathlib import Path
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# -----------------------------
# Embeddings & Vector DB config
# -----------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
VECTOR_DB_DIR = "db/company_vectors"


# -----------------------------
# State Schema
# -----------------------------
class CompanyState(BaseModel):
    messages: List[BaseMessage] = []
    company_name: str
    retrieved_docs: List[str] = []
    summary: Optional[str] = None


# -----------------------------
# Vectorstore Utilities
# -----------------------------

def get_or_create_vectorstore(company_name: str) -> Qdrant:
    db_path = Path(VECTOR_DB_DIR) / company_name

    # Create QdrantClient with persistent storage
    client = QdrantClient(path=str(db_path))

    # Check if collection exists
    collections = client.get_collections().collections
    collection_exists = any(col.name == company_name for col in collections)

    if not collection_exists:
        # Get embedding dimension by testing with dummy text
        sample_embedding = embedding_model.embed_query("dummy")
        dimension = len(sample_embedding)

        # Create collection if it doesn't exist
        client.create_collection(
            collection_name=company_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE  # or Distance.EUCLIDEAN, Distance.DOT
            )
        )

    return Qdrant(
        client=client,
        collection_name=company_name,
        embeddings=embedding_model
    )


def update_vectorstore_with_new_docs(company_name: str, new_texts: List[str]) -> List[Document]:
    vs = get_or_create_vectorstore(company_name)
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    all_chunks = []

    for text in new_texts:
        if not text or not isinstance(text, str):
            continue
        try:
            chunks = splitter.split_text(text)
        except IndexError:
            print(f"Skipping text due to split error: {text[:80]}")
            continue
        all_chunks.extend([Document(page_content=chunk) for chunk in chunks])

    # Get existing documents to avoid duplicates
    existing = vs.similarity_search(company_name, k=20)
    existing_contents = set(doc.page_content for doc in existing)

    # Filter out duplicates
    new_chunks = [doc for doc in all_chunks if doc.page_content not in existing_contents]

    if new_chunks:
        # Add documents to Qdrant (automatically persisted)
        vs.add_documents(new_chunks)
        # No need for save_local() - Qdrant auto-saves
        print(f"Added {len(new_chunks)} new document chunks to {company_name} vectorstore")
    else:
        print(f"No new chunks to add to {company_name} vectorstore")

    return new_chunks


def retrieve_from_vectorstore(company_name: str, query: str) -> List[Document]:
    vs = get_or_create_vectorstore(company_name)
    return vs.similarity_search(query, k=5)


# -----------------------------
# RAG Retriever Tool
# -----------------------------
@tool(
    description="Retrieve company information using web search and update a per-company vector database with new information. Deduplicates and returns relevant chunks from the vector store for the given query.")
def rag_company_retriever(query: str, company_name: str) -> List[str]:
    search = TavilySearch(max_results=5)
    fresh_results = search.run(query)
    fresh_docs = [r['content'] for r in fresh_results['results']]

    update_vectorstore_with_new_docs(company_name, fresh_docs)
    results = retrieve_from_vectorstore(company_name, query)

    return [doc.page_content for doc in results]


# -----------------------------
# Retrieval Node
# -----------------------------
def retrieval_node(state: CompanyState) -> CompanyState:
    query = state.messages[-1].content
    docs = rag_company_retriever.invoke({"query": query, 'company_name': 'Meta'})
    return CompanyState(
        messages=state.messages,
        company_name=state.company_name,
        retrieved_docs=docs,
        summary=None
    )


# -----------------------------
# Summarization Node
# -----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def summarize_company(state: CompanyState) -> CompanyState:
    query = state.messages[-1].content
    context = "\n".join(state.retrieved_docs) if state.retrieved_docs else "No company information available."

    prompt = f"""
Use the following information to summarize the company based on the user's query:

Query:
{query}

Company Context:
{context}

---
Provide:
1. Company Description
2. Main Products/Services
3. Industry & Market
4. Recent News or Events (if any)
5. Funding/Valuation (if available)
"""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return CompanyState(
            messages=state.messages,
            company_name=state.company_name,
            retrieved_docs=state.retrieved_docs,
            summary=response.content
        )
    except Exception as e:
        print(f"Error generating summary: {e}")
        return state


# -----------------------------
# LangGraph Assembly
# -----------------------------
def get_company_summary_graph():
    graph = StateGraph(state_schema=CompanyState)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("summarize", summarize_company)
    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


# -----------------------------
# Runtime Execution
# -----------------------------
if __name__ == '__main__':
    try:
        graph = get_company_summary_graph()

        query = "Tell me about Meta"
        company = "Meta"

        initial_state = CompanyState(
            company_name=company,
            messages=[HumanMessage(content=query)]
        )

        result = graph.invoke(initial_state)
        final_state = result if isinstance(result, CompanyState) else CompanyState(**result)

        print(f"Company: {final_state.company_name}")
        print(f"Retrieved {len(final_state.retrieved_docs)} documents from vector DB")
        print("\nSummary:\n", final_state.summary)

    except Exception as e:
        print(f"Error running the graph: {e}")
