import os
import logging
import pandas as pd
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_film_dataset(file_name: str) -> pd.DataFrame:
    film_df = pd.read_csv(file_name)
    film_df = film_df.dropna(subset=['title'])
    return film_df


def process_maininfo_to_document(row: pd.Series) -> Document:
    text = f'''Title: {row.get('title', '')}
             Directed by: {row.get('director', '')}
             Country: {row.get('country', '')}
             Date Added: {row.get('date_added', '')}
             Release Year: {row.get('release_year', '')}
             Rating: {row.get('rating', '')}
             Duration: {row.get('duration', '')}
             Listed In: {row.get('listed_in', '')}
             '''
    return Document(page_content=text)


def process_castinfo_to_document(row: pd.Series) -> Document:
    text = f'''Title: {row.get('title', '')}
             Directed by: {row.get('director', '')}
             Cast: {row.get('cast', '')}
             '''
    return Document(page_content=text)


def process_description_to_document(row: pd.Series) -> Document:
    text = f'''Title: {row.get('title', '')}
             Date Added: {row.get('date_added', '')}
             Description: {row.get('description', '')}
             '''
    return Document(page_content=text)


def db_exists(path: str) -> bool:
    sqlite_path = os.path.join(path, "chroma.sqlite3")
    return os.path.exists(sqlite_path) and len(os.listdir(path)) > 1


def load_or_create_vectordb(emb_model_name: str, dataset_path: str, chunk_size: int = 500, chunk_overlap: int = 100, db_path: str = "data/chroma_db") -> Chroma:
    embedding = HuggingFaceEmbeddings(model_name=emb_model_name)

    if db_exists(db_path):
        logger.info(f"Loading existing Chroma DB from {db_path}")
        return Chroma(persist_directory=db_path, embedding_function=embedding)

    logger.info("No existing DB found. Creating new one...")

    film_df = load_film_dataset(dataset_path)

    docs = [process_maininfo_to_document(row) for _, row in film_df.iterrows()]
    docs.extend([process_castinfo_to_document(row) for _, row in film_df.iterrows()])
    docs.extend([process_description_to_document(row) for _, row in film_df.iterrows()])

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splitted_docs = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(documents=splitted_docs, embedding=embedding, persist_directory=db_path)
    return vectordb


def load_chat_model(model_name: str, hf_token: str, device: str = "cpu", max_new_tokens: int = 512) -> HuggingFacePipeline:
    logger.info(f"Loading chat model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, trust_remote_code=True, token=hf_token)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens)

    llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={"stop": ["\nHuman:", "\nQuestion:", "\nAssistant:"]})
    return llm
