import os
import sys
import embedding

from embedding import (
    api_loader,
    embedding_api_docs,
    EncodedApiDocVectorStore
)

import retrieval_generation
from retrieval_generation import (
    retrieval_text
)

from qdrant_client import QdrantClient

# from embeddings_api_docs import test_run
# from retrieval_generation import test_run2

from config import EMB_MODEL

def ask_api(query, vstore):
    response = retrieval_text.generate_response(query, vstore)
    return response

def summary(query):
    response = retrieval_text.generate_response(query)
    return response
    
if __name__ == "__main__":
    retrieval_text.test_run2()
    print("hello")