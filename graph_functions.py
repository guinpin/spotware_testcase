import logging
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def generate_answer(data, llm, prompt):
    question = data["question"]
    docs = data.get("documents", [])
    history = data.get("history", [])

    context = "\n\n".join([doc.page_content for doc in docs])
    logger.info(f"Found context: {context}...")

    chain = prompt | llm

    result = chain.invoke({
        "context": context,
        "question": question,
        "stop": ["\nHuman:", "\nQuestion:"]
    })

    logger.info(f"LLM raw output: {result}")
    result = result.split("Answer:")[-1].split("Assistant:")[0].strip()

    history.append(f"Q: {question}\nA: {result}")
    logger.info(f"LLM raw output: {result}, history: {history}")
    return {"answer": result, "history": history}


def rerank_documents(reranker, query, docs, top_k=3):

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))
    ranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


def retrieve_docs(data, retriever, reranker):
    query = data["question"]
    raw_docs = retriever.invoke(query)
    logger.info(f"Retrieved {len(raw_docs)} documents")

    reranked_docs = rerank_documents(reranker, query, raw_docs, top_k=3)
    logger.info(f"Top-{3} reranked documents selected")

    return {"documents": reranked_docs}


def has_documents(data):
    return "generate" if data.get("documents") else "no_docs"


def no_docs_found(data):
    return {"answer": "I couldn't find any relevant information to answer your question."}


class RAGState(TypedDict):
    question: str
    documents: list
    answer: str
    history: list[str]


def build_rag_graph(llm, prompt, retriever, reranker=None):
    logger.info(f"Building llm graph...")
    builder = StateGraph(RAGState)

    builder.add_node("retrieve", RunnableLambda(lambda data: retrieve_docs(data, retriever, reranker)))
    builder.add_node("generate", RunnableLambda(lambda data: generate_answer(data, llm, prompt)))
    builder.add_node("no_docs", RunnableLambda(no_docs_found))

    builder.set_entry_point("retrieve")

    builder.add_conditional_edges("retrieve", has_documents, {
        "generate": "generate",
        "no_docs": "no_docs"
    })

    builder.add_edge("generate", END)
    builder.add_edge("no_docs", END)

    return builder.compile()
