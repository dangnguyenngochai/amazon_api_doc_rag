import sys
import os
import gradio as gr
import gradio_app
from gradio_app.theme import minigptlv_style, custom_css,text_css

from config import EMB_MODEL
from main import (
    ask_api,
    summary
)

from qdrant_client import QdrantClient

sys.path.append('/embedding')
sys.path.append('/retrieval_generation')

from embedding import EncodedApiDocVectorStore

DEMO_VSTORE = None
        
def run_demo_ask_api(query):
    try:
        response = ask_api(query, DEMO_VSTORE)
        return response
    except Exception as ex:
        print(ex)
        print('Keep going !!! Almost there')

def run_demo_summary(query):
    try:
        response = summary(query)
        return response
    except Exception as ex:
        print(ex)
        print('Keep going !!! Almost there')
        
def run_demo_summary2():
    try:
        response = ""
        
        with open('amazon_api_concepts.txt', 'r') as f:
            print('testing')
            data = f.readlines()
            print(data)
            for i in range(0, len(data), 10):
                concepts = data[i:i + 10]
                query = f"Please give me the definitions for the following\n{concepts}"%concepts
                result = summary(query)
                result = '\n'.join(result.split('\n')[1:-1])
                response = response + '\n' + result
                
        with open("output/amazon_api_concepts_def.csv", 'w+') as f:
            for item in response.split('\n'):
                item = item.strip('\\n|-')
                item = item.split(':')
                if len(item) == 2:
                    f.writelines(','.join(item))
                    f.writelines('\n')
                elif len(item) == 1 and item[0] == '':
                    pass
                else:
                    print('This should not happen')
                    print("Wrong response", item)
        return response
    except Exception as ex:
        print(ex)
        print('Keep going !!! Almost there')
        
        
title = """<h1 Ask API align="center"></h1>"""
description = """<h5>This is the demo for Amazon Advertising API Assistant</h5>"""
with gr.Blocks(title="Ask API Prototype 🎞️🍿",css=text_css ) as demo :

    gr.Markdown(title)
    gr.Markdown(description)
        
    with gr.Tab("Ask API"):
        with gr.Row():
            with gr.Column():
                question_local = gr.Textbox(label="Your Question", placeholder="Default: What is the API for listing ads account?")
                process_button_local = gr.Button("Answer the Question (QA)")
                
            with gr.Column():
                answer_local=gr.Text("Answer will be here",label="Ask-API's Answer")
        try:
            process_button_local.click(fn=run_demo_ask_api, inputs=[question_local], outputs=[answer_local])
        except Exception as ex:
            print(ex)
    with gr.Tab("Summary for me"):
        with gr.Row():
            with gr.Column():
                question = gr.Textbox(label="Your Question", placeholder="Default: Amazon - Portfolio, Product, Campaign, Ad Group, Product Ad, Keyword")
                process_button = gr.Button("Answer the Question (QA)")
                getall_button = gr.Button("Get me all the defintions")
            with gr.Column():
                answer=gr.Text("Answer will be here",label="Ask-API's Answer")
        try:
            process_button.click(fn=run_demo_summary, inputs=[question], outputs=[answer])
            getall_button.click(fn=run_demo_summary2, inputs=[], outputs=[answer])
        except Exception as ex:
            print(ex)
       
if __name__ == "__main__":
    print("Setting things up....")
    
    _ = EMB_MODEL #load embedding model into memory

    collection_name = 'api_docs'
    qdrant_client =QdrantClient(location=":memory:")
    vstore = EncodedApiDocVectorStore(collection_name=collection_name, qdrant_client=qdrant_client, model=EMB_MODEL)
    
    for file in os.listdir('test_data'):
        path = os.path.join('test_data', file)
        vstore.embeddings_apidocs(path, collection_name)
        
    DEMO_VSTORE = vstore
    print("Done")
    
    demo.queue().launch(share=True,show_error=True, server_port=2411)
