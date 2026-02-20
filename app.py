import streamlit as st
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Download needed NLTK data
nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# ----------------------------
# LOAD FILES
# ----------------------------

model = Word2Vec.load("word2vec.model")
embeddings = np.load("embeddings.npy")

with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()


# ----------------------------
# HELPER FUNCTIONS
# ----------------------------

def get_embedding(words):
    vectors = [model.wv[word] for word in words if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)


def get_query_embedding(query):
    words = [
        word.lower()
        for word in word_tokenize(query)
        if word.isalnum() and word.lower() not in stop_words
    ]
    return get_embedding(words)


def retrieve_top_k(query_embedding, k=5):
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings
    )[0]

    top_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_indices]


def get_top_sentences(doc_text, query_embedding, top_n=3):
    sentences = sent_tokenize(doc_text)
    scored_sentences = []

    for sentence in sentences:
        words = [
            word.lower()
            for word in word_tokenize(sentence)
            if word.isalnum() and word.lower() not in stop_words
        ]

        sent_embedding = get_embedding(words)

        score = cosine_similarity(
            query_embedding.reshape(1, -1),
            sent_embedding.reshape(1, -1)
        )[0][0]

        scored_sentences.append((sentence, score))

    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    return scored_sentences[:top_n]


# ----------------------------
# STREAMLIT UI
# ----------------------------

st.title("Information Retrieval with Semantic Sentence Ranking")

query = st.text_input("Enter your query:")

if st.button("Search") and query:

    query_embedding = get_query_embedding(query)

    results = retrieve_top_k(query_embedding, k=5)

    st.write("## Top Relevant Documents")

    for doc_text, score in results:

        st.write(f"### Document Similarity: {score:.4f}")

        top_sentences = get_top_sentences(doc_text, query_embedding)

        for sentence, sent_score in top_sentences:
            st.write(f"- ({sent_score:.4f}) {sentence}")

        st.write("---")
