import asyncio
import time
import traceback

from fastapi import BackgroundTasks, FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from gpt_utils import stream_open_ai_response
from app_utils import chunk_text, detect_file_type, extract_docx_text, extract_pdf_text, result_reranker, semantic_chunking, encode_chunks
from fastapi.concurrency import run_in_threadpool
from models import SearchQuery
from qdrant_utils import search_qdrant, upsert_qdrant
from app_logging import logger

app = FastAPI()

V = "1.0.0"

async def process_file_pipeline(content: bytes, filename: str, collection_name: str):
    header = content[:2048]
    file_type = detect_file_type(header)

    if not file_type:
        return

    if file_type == "pdf":
        text = extract_pdf_text(content)
    elif file_type == "docx":
        text = extract_docx_text(content)
    else:
        logger.error(f"Unhandled file type: {file_type}")
        return

    if not text.strip():
        logger.warning(f"No text extracted from: {filename}")
        return

    chunks = chunk_text(text)
    if len(chunks) == 0:
        return
    embeddings = encode_chunks(chunks)
    # asyncio.run(upsert_qdrant(chunks, embeddings, filename, collection_name))
    await upsert_qdrant(chunks, embeddings, filename, collection_name)

    logger.info(f"Finished processing: {filename}")


@app.get("/")
async def read_root():
    return f"FastAPI App is running in version {V}"

@app.post("/streamResponse")
async def ask_questions(query: SearchQuery):
    context="no context just answer the question"
    question = query.question
    return StreamingResponse(stream_open_ai_response(question, context), media_type="text/plain")


@app.post("/vectoriseAndUpsertDoc")
async def vectoriseAndUpsertDoc(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection_name: str = Form(...)
):
    
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Invalid file")
        
        logger.info(f"Received file: {file.filename}")
        content = await file.read()

        background_tasks.add_task(process_file_pipeline, content, file.filename, collection_name)
        return {"message": "File received. Processing started in background."}
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Error processing file")

# @app.post("/vectoriseAndUpsertDoc")
# async def vectoriseAndUpsertDoc(file: UploadFile = File(...)):
#     logger.info(f"processing file and qdrant upload for file: {file.filename}")

#     header = await file.read(2048)  
#     file_type = detect_file_type(header)
#     if not file_type:
#         raise HTTPException(status_code=400, detail="Unsupported file type")
#     await file.seek(0)
    
#     content = await file.read()
    
#     if file_type == "pdf":
#         text = await run_in_threadpool(extract_pdf_text, content)
#     elif file_type == "docx":
#         text = await run_in_threadpool(extract_docx_text, content)
#     # chunks = await run_in_threadpool(semantic_chunking, text)
#     chunks = await run_in_threadpool(chunk_text, text)
#     embeddings = await run_in_threadpool(encode_chunks, chunks)
#     await upsert_qdrant(chunks, embeddings, file.filename)
#     return {"message": "File uploaded and processed successfully"}
    

@app.post("/search")
async def search(query: SearchQuery):
    query_start_time = time.time()  
    question = query.question
    hnsw = query.hnsw
    if hnsw:
        logger.info(f"hnsw search enabled") 
    query_vector = await run_in_threadpool(encode_chunks, [question])
    query_result = await search_qdrant(query_vector[0], hnsw)
    reranked_results = await result_reranker(question, query_result)
    # logger.info(f"Query result: {query_result}")
    retrival_time = time.time() - query_start_time
    logger.info(f"Retrival time: {retrival_time} seconds")
    return StreamingResponse(stream_open_ai_response(question, reranked_results), media_type="text/plain")