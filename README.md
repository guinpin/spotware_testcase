# Netflix RAG (Retrieval-Augmented Generation)
This FastAPI app uses LangGraph and LangChain to answer questions about Netflix content. It finds and uses information from documents to give better answers (RAG).

## Tech Stack

- Python 3.12+
- FastAPI
- LangChain + LangGraph
- Hugging Face Transformers
- ChromaDB
- Sentence Transformers + CrossEncoder

## Project structure
```bash
├── main.py               
├── data_functions.py     
├── graph_functions.py    
├── data/
│   └── netflix_titles.csv
│   └──chroma_db/            
├── .env
└── requirements.txt
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/your-name/netflix-rag.git
cd spotware_testcase
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a .env file:
```env
HF_TOKEN=your_huggingface_token
SMITH_TOKEN=your_langchain_hub_token
```

## Run the app
```bash
uvicorn main:app --reload
```
Access the API at:
📍 http://localhost:8000/docs

Example request
```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the TV series Stranger Things about?"}'
```
## How to use the app via Swagger UI
1. Open your browser and go to: http://localhost:8000/docs
2. Find the POST /answer endpoint 
3. Click "Try it out"
4. Enter a question in JSON format, e.g.:
```json
{
  "question": "What is the TV series Stranger Things about?"
}
```
5. Click "Execute"
6. Scroll down to see the generated answer

## Example response
```json
{
  "answer": "Stranger Things is a sci-fi series about a young boy who disappears, a government experiment, and supernatural forces."
}
```

## Notes on Quality and Limitations
* The answer quality can be better if we use larger rerankers and language models.
But this project uses small and free models that can run on a local computer.

* Current language model: google/gemma-3-1b-it
Current reranker: cross-encoder/ms-marco-MiniLM-L-6-v2

* The system keeps the chat history, but it is not used in the prompt.
This is because the current task does not need long conversations.

## Potential Improvements
* Add conversation-type memory to the prompt for multi-turn interactions
* Categorize questions and build more complex graph branches
* Reintroduce response streaming (currently disabled for simplicity)
* Add frontend interface for interaction (e.g., chat UI)
