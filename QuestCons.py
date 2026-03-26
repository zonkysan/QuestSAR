import os
import json
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================================================
# CONFIG
# =========================================================
CSV_PATH = r"C:\Users\stefano.zagarella\Downloads\streamlit\domande.csv"

INDEX_DIR = r"C:\Users\stefano.zagarella\Downloads\streamlit\index"
EMBEDDINGS_PATH = os.path.join(INDEX_DIR, "semantic_index_embeddings.npy")
METADATA_PATH = os.path.join(INDEX_DIR, "semantic_index_metadata.pkl")
INFO_PATH = os.path.join(INDEX_DIR, "semantic_index_info.json")

MODEL_NAME = "all-MiniLM-L6-v2"

# =========================================================
# STREAMLIT
# =========================================================
st.set_page_config(page_title="Ricerca Semantica CSV", layout="wide")
st.title("🔎 Ricerca Semantica Real-Time")

# =========================================================
# MODELLO
# =========================================================
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

# =========================================================
# CSV
# =========================================================
def load_csv_from_fixed_path(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File CSV non trovato: {csv_path}")

    df_raw = pd.read_csv(csv_path, sep=None, engine="python")
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    target_cols = ["DOMANDA", "RISPOSTA A", "MATERIA"]
    existing_cols = [c for c in target_cols if c in df_raw.columns]
    df = df_raw[existing_cols].copy()

    if "DOMANDA" not in df.columns:
        raise ValueError("Il file CSV deve contenere la colonna DOMANDA.")

    df["DOMANDA"] = df["DOMANDA"].fillna("").astype(str)

    if "RISPOSTA A" in df.columns:
        df["RISPOSTA A"] = df["RISPOSTA A"].fillna("").astype(str)
    else:
        df["RISPOSTA A"] = ""

    if "MATERIA" in df.columns:
        df["MATERIA"] = df["MATERIA"].fillna("").astype(str)

    # Testo usato per embedding
    df["TEXT_FOR_EMBEDDING"] = (
        df["DOMANDA"].fillna("").astype(str) + " " +
        df["RISPOSTA A"].fillna("").astype(str)
    ).str.strip()

    return df

# =========================================================
# INFO CSV
# =========================================================
def get_csv_signature(csv_path: str) -> dict:
    stat = os.stat(csv_path)
    return {
        "path": os.path.abspath(csv_path),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "model_name": MODEL_NAME
    }

def load_index_info() -> dict | None:
    if not os.path.exists(INFO_PATH):
        return None
    with open(INFO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_index_info(info: dict):
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

# =========================================================
# BUILD / LOAD INDEX
# =========================================================
def build_and_save_index(df: pd.DataFrame, csv_path: str):
    model = load_model()
    texts = df["TEXT_FOR_EMBEDDING"].tolist()

    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    metadata_df = df[["DOMANDA", "RISPOSTA A"] + ([ "MATERIA" ] if "MATERIA" in df.columns else [])].copy()

    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata_df, f)

    save_index_info(get_csv_signature(csv_path))

def is_index_valid(csv_path: str) -> bool:
    if not (
        os.path.exists(EMBEDDINGS_PATH)
        and os.path.exists(METADATA_PATH)
        and os.path.exists(INFO_PATH)
    ):
        return False

    saved_info = load_index_info()
    current_info = get_csv_signature(csv_path)

    return saved_info == current_info

@st.cache_resource
def load_or_build_index(csv_path: str):
    df = load_csv_from_fixed_path(csv_path)

    if not is_index_valid(csv_path):
        build_and_save_index(df, csv_path)

    embeddings = np.load(EMBEDDINGS_PATH)

    with open(METADATA_PATH, "rb") as f:
        metadata_df = pickle.load(f)

    return metadata_df, embeddings

# =========================================================
# SEARCH
# =========================================================
def semantic_search(query: str, metadata_df: pd.DataFrame, embeddings: np.ndarray, top_k: int = 300):
    model = load_model()

    query_embedding = model.encode(
        [query],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    # cosine similarity, dato che i vettori sono normalizzati
    scores = embeddings @ query_embedding

    top_k = min(top_k, len(metadata_df))
    top_indices = np.argsort(scores)[::-1][:top_k]

    result = metadata_df.iloc[top_indices].copy()
    result["SCORE"] = scores[top_indices]

    return result

# =========================================================
# UI
# =========================================================
try:
    st.caption(f"CSV fisso: `{CSV_PATH}`")

    with st.spinner("Caricamento indice semantico..."):
        metadata_df, embeddings = load_or_build_index(CSV_PATH)

    min_chars = 2
    batch_size = 100

    if "visible_rows" not in st.session_state:
        st.session_state.visible_rows = batch_size
    if "last_search" not in st.session_state:
        st.session_state.last_search = ""

    search_query = st.text_input(
        "Cerca per contesto nella DOMANDA:",
        placeholder=f"Inserisci almeno {min_chars} caratteri..."
    ).strip()

    threshold = st.slider(
        "Soglia minima similarità",
        min_value=0.10,
        max_value=0.95,
        value=0.25,
        step=0.01
    )

    if search_query != st.session_state.last_search:
        st.session_state.visible_rows = batch_size
        st.session_state.last_search = search_query

    if not search_query or len(search_query) < min_chars:
        st.info(f"Inserisci almeno {min_chars} caratteri per avviare la ricerca.")
        st.stop()

    df_filtered = semantic_search(
        query=search_query,
        metadata_df=metadata_df,
        embeddings=embeddings,
        top_k=500
    )

    df_filtered = df_filtered[df_filtered["SCORE"] >= threshold].copy()
    df_filtered["SCORE"] = df_filtered["SCORE"].round(4)

    display_cols = [c for c in ["DOMANDA", "RISPOSTA A", "MATERIA"] if c in df_filtered.columns]
    df_filtered = df_filtered[display_cols]

    total_results = len(df_filtered)
    st.write(f"Risultati trovati: {total_results}")

    if total_results == 0:
        st.warning("Nessun risultato trovato.")
    else:
        df_to_show = df_filtered.head(st.session_state.visible_rows).copy()

        st.dataframe(
            df_to_show,
            use_container_width=True,
            hide_index=True,
            height=650,
            column_config={
                "DOMANDA": st.column_config.TextColumn(
                    "DOMANDA",
                    width="large",
                    max_chars=None,  # 🔥 abilita multilinea (wrap)
                ),
                "RISPOSTA A": st.column_config.TextColumn(
                    "RISPOSTA A",
                    width="large",
                    max_chars=None,  # 🔥 abilita multilinea (wrap)
                ),
                "MATERIA": st.column_config.TextColumn(
                    "MATERIA",
                    width="small",
                ),
            },
        )

        if st.session_state.visible_rows < total_results:
            if st.button("Carica altri risultati"):
                st.session_state.visible_rows += batch_size
                st.rerun()

    with st.expander("Dettagli indice"):
        info = load_index_info()
        st.json(info)

        st.write(f"Numero righe indicizzate: {len(metadata_df)}")
        st.write(f"Dimensione embeddings: {embeddings.shape}")

    if st.button("Rigenera indice manualmente"):
        with st.spinner("Rigenerazione indice in corso..."):
            df_tmp = load_csv_from_fixed_path(CSV_PATH)
            build_and_save_index(df_tmp, CSV_PATH)
            st.cache_resource.clear()
            st.rerun()

except Exception as e:
    st.error(f"Errore: {e}")