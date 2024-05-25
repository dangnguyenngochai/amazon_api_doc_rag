import os
from embedding import (
    EncodedApiDocVectorStore,
    test_run as dummy_emb,
    )

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
# from langchain_google_genai import (
#     ChatGoogleGenerativeAI
# )
from langchain import hub
from langchain_cohere import ChatCohere

import langchain
langchain.debug=True

# os.environ["OPENAI_API_KEY"] = 'sk-proj-0VuYdXfFXk6evfzOzLJTT3BlbkFJcshpzLyIT1ij7tR8q11Q'
# os.environ["GOOGLE_API_KEY"] = 'AIzaSyB-PQLnrQm2Z5UGXW28R24TXG99MLPICFw'
os.environ['COHERE_API_KEY'] = 'OhqrR0Bude8zr30XWVjreKC4LNbNHPhivxw7n0Vw'

def prompt_generator(mode):
    if mode=='question_answer':
        sys_message = """
        SYSTEM:
        You are an assistant for question-answering tasks relating to Amazon Advertising who gives concise, providing no more information than what is directly related to the questions, and accurate answers on topics regarding the Amazon Advertising API. 
        Try to categorize user questions into the following 2 cases:

            Case 1: If the question concerns the API of Amazon, construct the CURL with the latest version of the Amazon Advertising APIs if there are multiple versions of the requested API, along with their reference links. 
            Case 2: If the question concerns a description of an Amazon Advertisement object, do step-by-step reasoning using information from the Amazon Advertising API website only and come up with an answer on the objects that fit the discription

        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
        CONTEXT:
        {context}
        """
        prompt = ChatPromptTemplate.from_messages([
            ('system', sys_message),
            ('human', "QUENSTION:\n{question}")
            ]
        )
        return prompt
    elif mode=='summary':
        sys_message = """
        SYSTEM:
        You are an expert on the Amazon Advertising API and provide concise and accurate answers on the subject. 
        You are tasked with providing definitions for Amazon Advertising API objects and concepts.
        Your answer will follow the following format:

        <Concept>:<Definition>
        
        If the mentioned objects or concepts do not exist in the documentation of the Amazon Advertising API, you have to say that they do not exist."""
        prompt = ChatPromptTemplate.from_messages([
            ('system', sys_message),
            ('human', "QUENSTION:\n{question}")
            ]
        )
        return prompt
        
        
def generate_response(query: str, vstore: EncodedApiDocVectorStore = None): 
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # fetch prompt template
    # prompt = hub.pull("rlm/rag-prompt")

    if vstore is not None:
        mode = 'question_answer'
        prompt = prompt_generator(mode)

        retriever = vstore.get_retriever(5)
        llm = ChatCohere(model="command-r")
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        response = rag_chain.invoke(query)
    else:
        mode = 'summary'
        prompt = prompt_generator(mode)
        
        llm = ChatCohere(model="command-r")
        summary_chain = (
            {"question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        response = summary_chain.invoke(query)
    return response

def test_run2():
    query = "Which is the api for listing the ads account?"
    dummy_vt = dummy_emb(run_query=False)
    response = generate_response(query, dummy_vt)
    print(response)
    
def test_run3():
    query = "Campaign"
    response = generate_response(query)
    print(response)
    
if __name__ == '__main__':
    test_run2()