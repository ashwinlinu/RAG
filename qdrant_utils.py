from qdrant_client.models import Distance, VectorParams
import uuid
from qdrant_client import AsyncQdrantClient
import asyncio
from app_logging import logger
qdrant = AsyncQdrantClient(url="http://localhost:6333") # or use API key

async def create_collection():
    if not await qdrant.collection_exists(collection_name="documents"):
        await qdrant.create_collection(
            collection_name="documents",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        logger.info("Collection created")
    else:
        logger.info("Collection already exists")


async def upsert_qdrant(chunks, embeddings, filename):
    points = [
        {
            "id": str(uuid.uuid4()),
            "vector": vector.tolist(),
            "payload": {
                "text": chunk,
                "source": filename
            }
        }
        for chunk, vector in zip(chunks, embeddings)
    ]

    await qdrant.upsert(
        collection_name="documents",
        points=points
    )
    logger.info(f"File{filename} uploaded and {len(chunks)} chunks processed successfully")

async def search_qdrant(query_vector):
    results = await qdrant.query_points(
        collection_name="documents",
        query=query_vector.tolist(),
        limit=3
    )
    result_text = []
    for point in results.points:
        result_text.append(point.payload["text"])
    return result_text


if __name__ == "__main__":
    asyncio.run(create_collection())