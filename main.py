import os
import logging
from langchain import hub
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
from data_functions import load_or_create_vectordb, load_chat_model
from graph_functions import build_rag_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

DATASET_FILE = 'data/netflix_titles.csv'
chunk_size = 500
chunk_overlap = 100
rank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
emb_model = "sentence-transformers/all-MiniLM-L6-v2"
chat_model = "google/gemma-3-1b-it"
sm_token = os.getenv("SMITH_TOKEN")
hf_token = os.getenv("HF_TOKEN")
DB_PATH = "data/chroma_db"

prompt = hub.pull("rlm/rag-prompt", api_key=sm_token)

db_exists = os.path.exists(os.path.join(DB_PATH, "index"))

# Create or load Chroma DB
vectordb = load_or_create_vectordb(emb_model_name=emb_model, dataset_path=DATASET_FILE, chunk_size=chunk_size,
                                   chunk_overlap=chunk_overlap, db_path=DB_PATH)
retriever = vectordb.as_retriever()

# Load chat model
llm = load_chat_model(chat_model, hf_token, device="cpu")

logger.info("Loading CrossEncoder")
reranker = CrossEncoder(rank_model)

# Building graph
graph = build_rag_graph(llm, prompt, retriever, reranker=reranker)

app = FastAPI()


class QuestionInput(BaseModel):
    question: str


@app.post("/answer")
def answer(input: QuestionInput):
    logger.info(f"Received question: {input.question}")
    result = graph.invoke({"question": input.question})
    logger.info(f"Generated answer: {result['answer']}")
    return {"answer": result}
