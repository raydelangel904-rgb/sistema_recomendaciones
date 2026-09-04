import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from utils import load_and_clean_data, fuzzy_match_title


st.set_page_config(
    page_title="CineMatch | NLP Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #12304a;
        --blue: #1769aa;
        --teal: #078c87;
        --green: #3aa76d;
        --mist: #f4f8fb;
        --ink: #183247;
    }
    .stApp { background: var(--mist); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(244, 248, 251, 0.92); }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #dbe7ef; }
    .hero {
        background: linear-gradient(120deg, #12304a 0%, #1769aa 58%, #078c87 100%);
        color: white; padding: 2.1rem 2.4rem; border-radius: 12px;
        margin-bottom: 1.2rem; box-shadow: 0 10px 28px rgba(18, 48, 74, .16);
    }
    .hero h1 { margin: 0; font-size: 2.35rem; letter-spacing: 0; }
    .hero p { margin: .55rem 0 0; color: #dceef2; font-size: 1.05rem; }
    .metric-card {
        background: white; border: 1px solid #dbe7ef; border-top: 4px solid var(--teal);
        border-radius: 8px; padding: 1rem 1.1rem; min-height: 100px;
    }
    .metric-label { color: #557185; font-size: .82rem; text-transform: uppercase; letter-spacing: .04em; }
    .metric-value { color: var(--navy); font-size: 1.65rem; font-weight: 700; margin-top: .3rem; }
    .section-title { color: var(--navy); font-size: 1.25rem; font-weight: 700; margin: 1.2rem 0 .55rem; }
    .result-card { background: white; border-left: 5px solid var(--blue); border-radius: 8px; padding: .9rem 1rem; margin: .55rem 0; border-top: 1px solid #e1ebf0; border-right: 1px solid #e1ebf0; border-bottom: 1px solid #e1ebf0; }
    .result-card strong { color: var(--navy); font-size: 1.05rem; }
    .muted { color: #5f7483; font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Cargando y limpiando el catálogo...")
def get_data():
    return load_and_clean_data(include_keywords=True)


@st.cache_resource(show_spinner="Entrenando modelo TF-IDF...")
def train_model(overviews):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=20000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(overviews)
    return vectorizer, matrix


def metric_card(label, value):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def recommend(title, df, matrix, count):
    matches = fuzzy_match_title(title, df["title"], n=5, cutoff=0.35)
    if not matches:
        return None, pd.DataFrame()
    matched_title = matches[0]
    row_index = df.index[df["title"] == matched_title][0]
    scores = linear_kernel(matrix[row_index], matrix).ravel()
    candidates = pd.DataFrame({"index": df.index, "similarity": scores})
    candidates = candidates[candidates["index"] != row_index]
    candidates = candidates.sort_values("similarity", ascending=False).head(count)
    results = df.loc[candidates["index"], ["title", "year", "genres_str", "vote_average", "overview"]].copy()
    results["similarity"] = candidates["similarity"].values
    return matched_title, results.reset_index(drop=True)


df = get_data()
df = df[df["overview"].str.strip().ne("")].reset_index(drop=True)
vectorizer, tfidf_matrix = train_model(df["overview"])

st.markdown(
    '<div class="hero"><h1>🎬 CineMatch</h1><p>Explorador de películas con NLP, TF-IDF y similitud coseno</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Consulta")
    query = st.text_input("Película de referencia", "The Dark Knight")
    amount = st.slider("Número de resultados", 3, 15, 8)
    genres = sorted({genre for values in df["genres_list"] for genre in values})
    genre_filter = st.selectbox("Filtrar por género", ["Todos"] + genres)
    min_rating = st.slider("Rating mínimo", 0.0, 10.0, 0.0, 0.1)
    min_year = st.number_input("Año mínimo", min_value=1880, max_value=2026, value=1880, step=1)
    search = st.button("🔎 Buscar recomendaciones", type="primary", width="stretch")

st.markdown('<div class="section-title">Panorama del catálogo</div>', unsafe_allow_html=True)
metric_columns = st.columns(4)
with metric_columns[0]:
    metric_card("Películas analizadas", f"{len(df):,}")
with metric_columns[1]:
    metric_card("Términos TF-IDF", f"{len(vectorizer.vocabulary_):,}")
with metric_columns[2]:
    metric_card("Rating promedio", f"{df['vote_average'].mean():.2f}")
with metric_columns[3]:
    metric_card("Con keywords", f"{(df['keywords_list'].str.len() > 0).sum():,}")

st.markdown('<div class="section-title">Resultados de similitud</div>', unsafe_allow_html=True)
matched_title, recommendations = recommend(query, df, tfidf_matrix, amount)

if matched_title is None:
    st.warning(f"No se encontró una película parecida a “{query}”. Prueba con otro título.")
elif matched_title.lower() != query.lower():
    st.info(f"Se interpretó la consulta como **{matched_title}**.")

if not recommendations.empty:
    if genre_filter != "Todos":
        recommendations = recommendations[recommendations["genres_str"].str.contains(genre_filter, na=False)]
    recommendations = recommendations[
        (recommendations["vote_average"] >= min_rating)
        & (recommendations["year"].fillna(0) >= min_year)
    ]

if recommendations.empty and matched_title is not None:
    st.info("No hay resultados con esos filtros. Reduce el rating mínimo, el año o cambia el género.")
else:
    for rank, row in recommendations.iterrows():
        year = int(row["year"]) if pd.notna(row["year"]) else "s/f"
        overview = row["overview"][:210].rstrip() + "..."
        st.markdown(
            f'<div class="result-card"><strong>{rank + 1}. {row["title"]}</strong> '
            f'<span class="muted">({year}) · similitud {row["similarity"]:.3f} · rating {row["vote_average"]:.1f}</span>'
            f'<br><span class="muted">{row["genres_str"]}</span><br>{overview}</div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="section-title">Distribución de ratings</div>', unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(10, 2.8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.hist(df["vote_average"], bins=25, color="#1769aa", edgecolor="white", alpha=0.9)
ax.axvline(df["vote_average"].mean(), color="#3aa76d", linewidth=2, linestyle="--")
ax.set_xlabel("Rating promedio")
ax.set_ylabel("Películas")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
st.pyplot(fig, width="stretch")
plt.close(fig)

st.caption("Modelo: TF-IDF sobre sinopsis en inglés, con unigramas y bigramas. Los filtros se aplican después de calcular la similitud.")
