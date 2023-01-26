import os
from ast import literal_eval
from datasets import load_dataset
import numpy as np
import pandas as pd

import openai
import tiktoken
from transformers import GPT2TokenizerFast
import gradio as gr

# get API key from top-right dropdown on OpenAI website
openai.api_key = os.getenv("OPEN_AI_API_KEY")

EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETIONS_MODEL = "text-davinci-003"
MAX_SECTION_LEN = 2000
COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 500,
    "model": COMPLETIONS_MODEL,
}

hf_ds = "juancopi81/yannic_ada_embeddings"
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

HEADER = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "This is not covered in my videos." Try imitating the style of the provided context. \n\nContext:\n"""
RESPONSE_SOURCES = " For more information, check out my following videos: "

# query separator to help the model distinguish between separate pieces of text.
SEPARATOR = "\n* "
ENCODING = "cl100k_base"  # encoding for text-embedding-ada-002

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

f"Context separator contains {separator_len} tokens"

# UTILS
def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))

def load_embeddings(hf_ds: str) -> dict:
    """
    Read the document embeddings and their keys from a HuggingFace dataset.
    
    hf_ds is the name of the HF dataset with exactly these named columns: 
        "TITLE", "URL", "TRANSCRIPTION", "transcription_length", "text", "ada_embedding"
    """
    hf_ds = load_dataset(hf_ds, split="train")
    hf_ds.set_format("pandas")
    df = hf_ds[:]
    df.ada_embedding = df.ada_embedding.apply(literal_eval)
    df["idx"] = df.index
    return {
        (r.idx, r.TITLE, r.URL): r.ada_embedding for idx, r in df.iterrows()
    }

def create_dataframe(hf_ds: str):
    hf_ds = load_dataset(hf_ds, split="train")
    hf_ds.set_format("pandas")
    df = hf_ds[:]
    df["num_tokens"] = df["text"].map(count_tokens)
    df["idx"] = df.index
    df = df.set_index(["idx", "TITLE", "URL"])
    return df

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def vector_similarity(x: list, y: list) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict) -> list:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> tuple:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.num_tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.text.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    
    header = HEADER
    
    return (header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:",
            chosen_sections_indexes)

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict,
    show_prompt: bool = False
) -> str:
    prompt, sources = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )
    gpt_answer = response["choices"][0]["text"].strip(" \n")
    
    if gpt_answer != "This is not covered in my videos.":
        res_sources = RESPONSE_SOURCES
        for source in sources[:2]:
            src_lst = eval(source)
            title = "".join(src_lst[1])
            url = "".join(src_lst[2])
            if url not in res_sources:
                final_src = title + " " + url
                res_sources += " " + final_src
    else:
        res_sources = ""
        
    final_answer = gpt_answer + res_sources

    return final_answer

df = create_dataframe(hf_ds)
document_embeddings = load_embeddings(hf_ds)

def predict(question, history):
    history = history or []
    response = answer_query_with_context(question, df, document_embeddings)
    history.append((question, response))
    return history, history

block = gr.Blocks()

with block:
    gr.Markdown("""<h1><center>Chat with Yannic</center></h1>
                <p>Each question is independent. You should not base your new questions on the previous conversation</p>
    """)
    chatbot = gr.Chatbot()
    question = gr.Textbox(placeholder="Enter your question")
    state = gr.State()
    submit = gr.Button("SEND")
    submit.click(predict, inputs=[question, state], outputs=[chatbot, state])

block.launch(debug = True)