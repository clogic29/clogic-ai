

import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import hashlib


model_to_docs = {
    'BAAI/bge-m3': {'collection_name': 'docs-baai', 'dimension': 1024},
    'paraphrase-multilingual-MiniLM-L12-v2': {'collection_name': 'docs-minilm', 'dimension': 384},
    # 'ko-sroberta-multitask': {'collection_name': 'docs-sroberta', 'dimension': 768},
    # 'multilingual-e5-base': {'collection_name': 'docs-e5', 'dimension': 1024},
}

class QdrantService:
    def __init__(self, collection_name: str = None, model_name: str = None):
        self.client = QdrantClient(host="localhost", port=6333)


    def create_collection(self, model_name: str=None):
        if not self.client.collection_exists(model_to_docs[model_name]['collection_name']):
            self.client.create_collection(collection_name=model_to_docs[model_name]['collection_name'],
                vectors_config=VectorParams(size=model_to_docs[model_name]['dimension'], distance=Distance.COSINE))

    def upsert(self, model_name: str=None, texts: list[str]=None):
        model = SentenceTransformer(model_name)
        self.create_collection(model_name)

        if texts is None:
            texts = []

        points = []
        for text in texts:
            uuid_value = str(uuid.UUID(hashlib.sha256(text.encode()).hexdigest()[:32]))
            points.append(
                PointStruct(id=uuid_value, vector=model.encode(text), payload={"text": text})
            )

        self.client.upsert(collection_name=model_to_docs[model_name]['collection_name'], points=points)

    def query(self, model_name: str=None, query: str=None, limit: int = 10):
        model = SentenceTransformer(model_name)
        self.create_collection(model_name)

        query_vector = model.encode(query)
        return self.client.query_points(collection_name=model_to_docs[model_name]['collection_name'], query=query_vector, limit=limit)


qdrant_service = QdrantService()