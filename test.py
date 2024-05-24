# from openai import OpenAI

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

if __name__ == "__main__":
    retrieval_text.test_run2()
    print("hello")