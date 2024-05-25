import os
import sys
import embedding

from embedding import (
    api_loader,
    embedding_api_docs
)

import retrieval_generation
from retrieval_generation import retrieval_text

# from embeddings_api_docs import test_run
# from retrieval_generation import test_run2

from config import EMB_MODEL

def ask_api(query):
    qdrant_client = QdrantClient(location=':memory:')
    vstore = EncodedApiDocVectorStore(collection_name='api_docs', qdrant_client=qdrant_client)
    response = generate_response(query, vstore)
    return response

def summary(query):
    response = generate_response(query)
    return response
    
if __name__ == "__main__":
    retrieval_text.test_run2()
    print("hello")