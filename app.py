import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load precomputed document embeddings
embeddings = np.load("embeddings.npy")

with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()


def retrieve_top_k(query_embedding, embeddings, k=10):
    """Retrieve top-k most similar documents using cosine similarity."""
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings
    )[0]

    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]


# ✅ Slightly smarter query embedding
def get_query_embedding(query):
    query_words = query.lower().split()

    matched_indices = [
        i for i, doc in enumerate(documents)
        if any(word in doc.lower() for word in query_words)
    ]

    if matched_indices:
        return np.mean(embeddings[matched_indices], axis=0)

    return np.random.normal(0, 0.01, embeddings.shape[1])


# ✅ NEW: Simple sentence extraction
def get_top_sentences(doc_text, query, top_n=3):
    query_words = query.lower().split()

    # Simple sentence split
    sentences = doc_text.replace("\n", " ").split(".")

    scored_sentences = []

    for sentence in sentences:
        score = sum(word in sentence.lower() for word in query_words)
        scored_sentences.append((sentence.strip(), score))

    scored_sentences.sort(key=lambda x: x[1], reverse=True)

    return scored_sentences[:top_n]


# ----------------------------
# Streamlit UI
# ----------------------------

st.title("Information Retrieval using Document Embeddings")

query = st.text_input("Enter your query:")

if st.button("Search") and query:

    query_embedding = get_query_embedding(query)
    results = retrieve_top_k(query_embedding, embeddings)

    st.write("### Top 10 Relevant Documents:")

    for doc, score in results:

        st.write(f"## Document (Score: {score:.4f})")

        # Show top sentences
        top_sentences = get_top_sentences(doc, query)

        for sentence, sent_score in top_sentences:
            if sent_score > 0:
                st.write(f"- {sentence}")

        st.write("---")
