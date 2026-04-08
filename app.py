from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from models import Query
from gpt_utils import stream_open_ai_response
from app_utils import chunk_text, detect_file_type, extract_docx_text, extract_pdf_text, semantic_chunking, encode_chunks
from fastapi.concurrency import run_in_threadpool
from qdrant_utils import search_qdrant, upsert_qdrant
from app_logging import logger

app = FastAPI()

V = "1.0.0"


@app.get("/")
async def read_root():
    return f"FastAPI App is running in version {V}"

@app.post("/streamResponse")
async def ask_questions(query: Query):
    context="no context just answer the question"
    question = query.question
    return StreamingResponse(stream_open_ai_response(question, context), media_type="text/plain")

@app.post("/vectoriseAndUpsertDoc")
async def vectoriseAndUpsertDoc(file: UploadFile = File(...)):
    logger.info(f"processing file and qdrant upload for file: {file.filename}")

    header = await file.read(2048)  
    file_type = detect_file_type(header)
    if not file_type:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    await file.seek(0)
    
    content = await file.read()
    
    if file_type == "pdf":
        text = await run_in_threadpool(extract_pdf_text, content)
    elif file_type == "docx":
        text = await run_in_threadpool(extract_docx_text, content)
    # chunks = await run_in_threadpool(semantic_chunking, text)
    chunks = await run_in_threadpool(chunk_text, text)
    embeddings = await run_in_threadpool(encode_chunks, chunks)
    await upsert_qdrant(chunks, embeddings, file.filename)
    return {"message": "File uploaded and processed successfully"}
    

@app.post("/search")
async def search(query: Query):
    question = query.question
    query_vector = await run_in_threadpool(encode_chunks, [question])
    query_result = await search_qdrant(query_vector[0])  
    # print(query_result) 
    return StreamingResponse(stream_open_ai_response(question, query_result), media_type="text/plain")