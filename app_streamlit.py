# app.py
import streamlit as st
from my_rag import MovieRAGWithDeepSearch
import time

st.set_page_config(
    page_title="CinemaRAG ",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
# 🎬 CinemaRAG 
### Your personal movie expert with deep RAG search

Just describe a movie you loved or the vibe you're looking for — I'll find perfect recommendations  
""")

@st.cache_resource(show_spinner="First launch: loading 50k movies and connecting to Mistral… (~30 sec)")
def get_rag():
    return MovieRAGWithDeepSearch(
        csv_path="tmdb_5000_movies.csv",
        vector_database_path="data/movie_chroma_db",
        initial_k=30,
        final_k=12
    )

with st.sidebar:
    st.header("🚀 Ready-made query examples")
    examples = [
        "Movies like Interstellar but with more philosophy and no happy ending",
        "Atmospheric noir in a rainy city, like Blade Runner 2049",
        "Russian comedies from the 2000s like What Men Talk About",
        "Mecha anime but not about teenagers, with an adult protagonist",
        "Warmer and kinder Christmas movies than Home Alone",
        "Dark psychological thrillers like Se7en or Zodiac",
        "70-80s sci-fi with practical effects like Tarkovsky's",
        "Korean revenge thrillers better than Oldboy",
    ]

    for i, ex in enumerate(examples):
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.current_query = ex

    st.divider()
    st.caption("""
    ⚙️ Engine: Mistral-7B + Chroma + mistral-embed (cloud)  
    🗃 Database: ~50 000 movies from TMDB  
    NLP/LLM-25F
    """)


if "current_query" in st.session_state:
    default_text = st.session_state.current_query
else:
    default_text = ""

query = st.text_area(
    "Your request",
    value=default_text,
    height=120,
    placeholder="e.g. something like Knockin' on Heaven's Door but with a female lead and a happy ending…",
    label_visibility="collapsed"
)

col1, col2 = st.columns([1, 4])
with col1:
    search_btn = st.button("🔍 Find movies", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🗑 Clear", use_container_width=True)

if clear_btn:
    query = ""
    st.session_state.pop("current_query", None)
    st.rerun()

if search_btn or (query := query.strip()):
    if not query:
        st.warning("Please write something =)")
        st.stop()

    with st.spinner("Searching for the perfect movies for you…"):
        start = time.time()
        rag = get_rag()                                   # only once
        result = rag.recommend_formatted(query)
        duration = time.time() - start

    st.success(f"Done in {duration:.1f} seconds!")

    st.markdown(result)

    with st.expander("🔍 Show raw retrieved documents"):
        docs = rag.recommend(query)["source_documents"]
        for i, doc in enumerate(docs, 1):
            st.write(f"**{i}.** {doc['title']} (ID: {doc['tmdb_id']})")
            st.caption(doc['content'][:500] + "...")

else:
    st.info("👆 Write a query above and hit the button — I'll find cinematic gems for you!")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888;'>"
    "CinemaRAG"
    "</p>",
    unsafe_allow_html=True
)