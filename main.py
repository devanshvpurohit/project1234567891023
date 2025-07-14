import streamlit as st
import os
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import tempfile

# Load BERT model and tokenizer
@st.cache_resource
def load_bert():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

tokenizer, model = load_bert()

# Function to extract text from file
def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = " ".join([para.text for para in doc.paragraphs])
    else:
        text = str(file.read(), 'utf-8')
    return text

# Function to compute BERT embedding
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()[0]

# Function to compute similarity using TF-IDF
def compute_tfidf_similarity(job_desc, resumes):
    vectorizer = TfidfVectorizer()
    docs = [job_desc] + resumes
    tfidf_matrix = vectorizer.fit_transform(docs)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return cosine_sim[0]

# Function to compute similarity using BERT
def compute_bert_similarity(job_desc, resumes):
    job_embed = get_bert_embedding(job_desc)
    scores = []
    for res in resumes:
        res_embed = get_bert_embedding(res)
        score = np.dot(job_embed, res_embed) / (np.linalg.norm(job_embed) * np.linalg.norm(res_embed))
        scores.append(score)
    return scores

# Streamlit UI
st.title("Automated Resume Shortlisting System")

st.markdown("### Upload Job Description or Paste Manually")
job_desc_file = st.file_uploader("Upload Job Description (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
job_desc_text_input = st.text_area("Or paste the job description here")

st.markdown("### Upload Resumes or Paste Manually")
resume_files = st.file_uploader("Upload Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)
manual_resume_texts = st.text_area("Or paste one or more resumes here (separated by '---')")

method = st.selectbox("Select Matching Method", ["TF-IDF", "BERT"])

if st.button("Shortlist Candidates"):
    if job_desc_file or job_desc_text_input:
        if job_desc_file:
            job_desc_text = extract_text(job_desc_file)
        else:
            job_desc_text = job_desc_text_input

        resumes_text = []
        if resume_files:
            resumes_text.extend([extract_text(file) for file in resume_files])

        if manual_resume_texts:
            manual_resumes = manual_resume_texts.split("---")
            resumes_text.extend([res.strip() for res in manual_resumes if res.strip()])

        if not resumes_text:
            st.warning("Please upload or paste at least one resume.")
        else:
            if method == "TF-IDF":
                scores = compute_tfidf_similarity(job_desc_text, resumes_text)
            else:
                scores = compute_bert_similarity(job_desc_text, resumes_text)

            st.subheader("Ranked Resumes")
            for i, (res_text, score) in enumerate(sorted(zip(resumes_text, scores), key=lambda x: x[1], reverse=True)):
                st.write(f"{i+1}. Score: {score:.4f}")
                st.text_area(f"Resume {i+1}", res_text[:1000] + ('...' if len(res_text) > 1000 else ''), height=150)
    else:
        st.warning("Please upload or paste the job description.")
