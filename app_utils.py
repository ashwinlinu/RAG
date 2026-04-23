import zipfile
import io
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer,CrossEncoder
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
from app_logging import logger
import pytesseract
from pdf2image import convert_from_bytes

from nltk.tokenize import sent_tokenize
import numpy as np

model = SentenceTransformer("BAAI/bge-base-en-v1.5")  


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# Better options:
# "all-mpnet-base-v2"  ← higher quality
# "BAAI/bge-base-en-v1.5" ← best for retrieval

def encode_chunks(chunks):
    return model.encode(chunks, batch_size=32, normalize_embeddings=True)


def semantic_chunking(text, threshold=0.7):
    sentences = sent_tokenize(text)

    embeddings = model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(embeddings[i-1], embeddings[i])

        if sim < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    chunks.append(" ".join(current_chunk))
    return chunks


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def extract_docx_text(content: bytes):
    doc = Document(io.BytesIO(content))
    return "\n".join([p.text for p in doc.paragraphs])

# def extract_pdf_text(content: bytes):
#     reader = PdfReader(io.BytesIO(content))
#     text = ""

#     for page in reader.pages:
#         text += page.extract_text() or ""

#     return text
# import pdfplumber

# def extract_pdf_text(content: bytes):
#     text = ""

#     with pdfplumber.open(io.BytesIO(content)) as pdf:
#         for i, page in enumerate(pdf.pages):
#             text += page.extract_text() or ""
#             logger.info(f"chunk: {i+1} extracted")
#     return text

def extract_pdf_text(content: bytes):
    images = convert_from_bytes(content)

    text = ""
    for i, img in enumerate(images):
        page_text = pytesseract.image_to_string(img)
        text += page_text + "\n"
        logger.info(f"pages: {i+1} extracted")

    return text

def is_docx(content: bytes):
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            return "word/document.xml" in z.namelist()
    except:
        return False

def detect_file_type(content: bytes):
    if content.startswith(b"%PDF"):
        return "pdf"
    if content.startswith(b"PK") and is_docx(content):
        return "docx"
    return "unknown"

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

async def result_reranker(query_vector, query_result):
    pairs = [(query_vector,doc) for doc in query_result]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(query_result, scores), key=lambda x: x[1], reverse=True)

    # Return top 5
    return [doc for doc, _ in ranked[:5]]