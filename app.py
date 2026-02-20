import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

# ----------------------------
# LOAD FILES
# ----------------------------

embeddings = np.load("embeddings.npy")

with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()


# ----------------------------
# QUERY EMBEDDING (DUMMY SAFE VERSION)
# ----------------------------

# Since we are not loading Word2Vec,
# we approximate query embedding by averaging
# document embeddings of documents containing query words

def get_query_embedding(query):
    query_words = query.lower().split()

    matched_indices = [
        i for i, doc in enumerate(documents)
        if any(word in doc.lower() for word in query_words)
    ]

    if matched_indices:
        return np.mean(embeddings[matched_indices], axis=0)

    # fallback if no match
    return np.zeros(embeddings.shape[1])


# ----------------------------
# DOCUMENT RETRIEVAL
# ----------------------------

def retrieve_top_k(query_embedding, k=5):
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings
    )[0]

    top_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_indices]


# ----------------------------
# SENTENCE RANKING
# ----------------------------

def get_top_sentences(doc_text, query, top_n=3):
    query_words = query.lower().split()
    sentences = sent_tokenize(doc_text)

    scored_sentences = []

    for sentence in sentences:
        score = sum(word in sentence.lower() for word in query_words)
        scored_sentences.append((sentence, score))

    scored_sentences.sort(key=lambda x: x[1], reverse=True)

    return scored_sentences[:top_n]


# ----------------------------
# STREAMLIT UI
# ----------------------------

st.title("Information Retrieval System")

query = st.text_input("Enter your query:")

if st.button("Search") and query:

    query_embedding = get_query_embedding(query)
    results = retrieve_top_k(query_embedding, k=5)

    st.write("## Top Relevant Documents")

    for doc_text, score in results:

        st.write(f"### Document Similarity: {score:.4f}")

        top_sentences = get_top_sentences(doc_text, query)

        for sentence, sent_score in top_sentences:
            if sent_score > 0:
                st.write(f"- {sentence}")

        st.write("---")
