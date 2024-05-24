from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# import torch.nn.functional as F
# from transformers import AutoModel, AutoTokenizer

from qdrant_client import QdrantClient
import pathlib
from .api_loader import YamlLoader, JsonLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

class EmcodedApiDocVectorStore:
    def __init__(self, collection_name, 
                 qdrant_client, 
                 model=None, 
                 emb_model_name="Alibaba-NLP/gte-large-en-v1.5"):
        
        self.collection_name = collection_name
        self.qdrant_local_path = "local_qdrant"
        self.qdrant_client = qdrant_client

        if model is not None:
            self.emd_model = model
        else:
            self.emb_model_name =HuggingFaceEmbeddings(
                model_name=emb_model_name,
                model_kwargs={
                    'trust_remote_code': True
                }
            )
        if not qdrant_client.collection_exists(self.collection_name):
            self.vector_store = None
        else:
            self.vector_store = Qdrant.from_existing_collection( 
#                 path=self.qdrant_local_path,
                location = ':memory:',
                collection_name=self.collection_name, 
                embeddings=self.emd_model,
            )
            
    def __load_loader(self,ext):
        support_ext = {
            '.yaml': YamlLoader,
            '.json': JsonLoader
        }
        if support_ext.get(ext, False):
            return support_ext.get(ext)
        else: 
            print("Loader cannot be loader, check your file type")

    def __load_apidoc_segments(self,api_data_path: str) -> list[Document]:
        document_loader = self.__load_loader(pathlib.Path(api_data_path).suffix)        
        raw_documents = document_loader(api_data_path).load()
        return raw_documents
    
    def get_retriever(self, top_k):
        if self.vector_store is not None:
            # using cosine similarity
            retriever = self.vector_store.as_retriever(search_type="similarity",
                                                 search_kwargs={"k": top_k})
            return retriever
        else:
            print("Something wrong with the vector store")

    def embeddings_apidocs(self, api_data_path: str, collection_name: str):
        try:            
            if self.qdrant_client.collection_exists(self.collection_name):
                print("Api document is indexed")
                return None
            
            documents = self.__load_apidoc_segments(api_data_path)

            if self.vector_store is not None:
                _ = self.vector_store.add_documents(documents)
            else:       
                self.vector_store = Qdrant.from_documents(
                    documents,
                    self.emd_model,
#                     path=self.qdrant_local_path,
                    location=':memory:',
                    collection_name=collection_name
                )
        except Exception as ex:
            print(ex)

    def query_relevants(self, query, top_k):
        # using cosine similarity
        retriever = self.get_retriever(top_k)
        if retriever is not None:
            relevants = retriever.invoke(query)
            return relevants

def test_run() -> EmcodedTranscriptpionVectorStore:
    api_data_path = ['data/sponsored_brands_v4.json', 'data/sponsored_brands_v3.yaml']
    test_query = "Which is the api for listing the add account?"
    model_name = "Alibaba-NLP/gte-large-en-v1.5"

    import sys
    sys.path.append('../')
    from config import EMB_MODEL

    collection_name = 'api_docs'

    try:
#         qdrant_client = QdrantClient(path='local_qdrant')
        qdrant_client = QdrantClient(location=':memory:')
        vstore = EmcodedApiDocVectorStore(model=EMB_MODEL, collection_name=collection_name, qdrant_client=qdrant_client)
        
        # test embeddings
        vstore.embeddings_apidocs(api_data_path, collection_name)
        
        # test query
        print('Running test for querying the indexed data')
        relevants = vstore.query_relevants(test_query, 1)
        print(relevants)

        return vstore
    except Exception as ex:
        print(ex)

if __name__ == "__main__":
    test_run()