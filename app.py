import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pickle
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import login



token = "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"




faiss_index = faiss.read_index("catalog_index.faiss")
with open("catalog_documents.pkl", "rb") as f:
    documents = pickle.load(f)


embed_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')


model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=token)



def get_query_embedding(query):
    return embed_model.encode([query])[0].astype("float32")

def search_faiss(query, top_k=5):
    query_vec = get_query_embedding(query)
    D, I = faiss_index.search(np.array([query_vec]), top_k)
    return [documents[i] for i in I[0]]


def generate_response(query, top_docs):
    prompt = f"""You are an AI assistant. Based on the following retrieved documents, answer the user query.

User Query: {query}

Retrieved Documents:
{top_docs}

Answer:"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm_model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = llm_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()


st.set_page_config(page_title="Assessment Recommender", layout="centered")

st.title("🧠 AI Assessment Recommender for Hiring")
st.markdown("Suggest suitable assessments for roles like **entry-level banking, cashiers, etc.**")

query = st.text_input("Enter your query", placeholder="Suggest assessments for entry-level banking roles")

if st.button("🔍 Get Recommendations") and query:
    with st.spinner("Searching and generating response..."):
        top_docs = search_faiss(query, top_k=5)
        response = generate_response(query, "\n\n".join(top_docs))

        st.subheader("📌 Recommended Assessments")
        st.write(response)

        with st.expander("📄 Retrieved Supporting Documents"):
            for i, doc in enumerate(top_docs):
                st.markdown(f"**Document {i+1}:**\n{doc}", unsafe_allow_html=True)


