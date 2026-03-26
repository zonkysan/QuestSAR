import io
import json
import pickle
import requests
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================================================
# CONFIG GITHUB
# =========================================================
GITHUB_OWNER = "zonkysan"
GITHUB_REPO = "QuestSAR"
GITHUB_BRANCH = "main"

# file CSV nel repo
CSV_FILE_PATH = "/domande.csv"

# cartella nel repo dove hai caricato i file di indicizzazione
INDEX_BASE_PATH = "index"
EMBEDDINGS_FILE = f"{INDEX_BASE_PATH}/semantic_index_embeddings.npy"
METADATA_FILE = f"{INDEX_BASE_PATH}/semantic_index_metadata.pkl"
INFO_FILE = f"{INDEX_BASE_PATH}/semantic_index_info.json"

MODEL_NAME = "all-MiniLM-L6-v2"

# =========================================================
# STREAMLIT
# =========================================================
st.set_page_config(page_title="Cons quest", layout="wide")
st.title("🔎 Ricerca domanda:")

# =========================================================
# URL HELPERS
# =========================================================
def github_raw_url(path: str) -> str:
    return f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{path}"

# =========================================================
# DOWNLOAD HELPERS
# =========================================================
@st.cache_data(show_spinner=False)
def download_text_from_github(path: str) -> str:
    url = github_raw_url(path)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text

@st.cache_data(show_spinner=False)
def download_bytes_from_github(path: str) -> bytes:
    url = github_raw_url(path)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content

def github_file_exists(path: str) -> bool:
    url = github_raw_url(path)
    response = requests.head(url, timeout=15)
    return response.status_code == 200

# =========================================================
# CSV
# =========================================================
@st.cache_data(show_spinner=False)
def load_csv_from_github(csv_path: str) -> pd.DataFrame:
    csv_bytes = download_bytes_from_github(csv_path)

    df_raw = pd.read_csv(io.BytesIO(csv_bytes), sep=None, engine="python")
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

    # Indicizzazione solo su DOMANDA
    df["TEXT_FOR_EMBEDDING"] = df["DOMANDA"].fillna("").astype(str).str.strip()

    return df

# =========================================================
# FIRMA REMOTA DEL CSV
# =========================================================
@st.cache_data(show_spinner=False)
def get_remote_csv_signature(csv_path: str) -> dict:
    csv_bytes = download_bytes_from_github(csv_path)

    return {
        "owner": GITHUB_OWNER,
        "repo": GITHUB_REPO,
        "branch": GITHUB_BRANCH,
        "csv_path": csv_path,
        "csv_size": len(csv_bytes),
        "model_name": MODEL_NAME,
    }

# =========================================================
# MODELLO
# =========================================================
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

# =========================================================
# BUILD INDICE IN MEMORIA
# =========================================================
def build_index_in_memory(df: pd.DataFrame):
    model = load_model()
    texts = df["TEXT_FOR_EMBEDDING"].tolist()

    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    metadata_df = df[[c for c in ["DOMANDA", "RISPOSTA A", "MATERIA"] if c in df.columns]].copy()
    return metadata_df, embeddings

# =========================================================
# LOAD INDICE DA GITHUB
# =========================================================
@st.cache_data(show_spinner=False)
def load_remote_index_info() -> dict | None:
    if not github_file_exists(INFO_FILE):
        return None
    content = download_text_from_github(INFO_FILE)
    return json.loads(content)

@st.cache_data(show_spinner=False)
def load_remote_metadata_df() -> pd.DataFrame:
    metadata_bytes = download_bytes_from_github(METADATA_FILE)
    return pickle.loads(metadata_bytes)

@st.cache_data(show_spinner=False)
def load_remote_embeddings() -> np.ndarray:
    emb_bytes = download_bytes_from_github(EMBEDDINGS_FILE)
    return np.load(io.BytesIO(emb_bytes))

def is_remote_index_valid(csv_path: str) -> bool:
    if not (
        github_file_exists(INFO_FILE)
        and github_file_exists(METADATA_FILE)
        and github_file_exists(EMBEDDINGS_FILE)
    ):
        return False

    saved_info = load_remote_index_info()
    current_info = get_remote_csv_signature(csv_path)

    return saved_info == current_info

@st.cache_resource
def load_or_build_index(csv_path: str):
    df = load_csv_from_github(csv_path)

    if is_remote_index_valid(csv_path):
        metadata_df = load_remote_metadata_df()
        embeddings = load_remote_embeddings()
    else:
        metadata_df, embeddings = build_index_in_memory(df)

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
    with st.spinner("Caricamento indice semantico..."):
        metadata_df, embeddings = load_or_build_index(CSV_FILE_PATH)

    min_chars = 2
    batch_size = 100

    if "visible_rows" not in st.session_state:
        st.session_state.visible_rows = batch_size
    if "last_search" not in st.session_state:
        st.session_state.last_search = ""

    threshold_options = [0.48, 0.50, 0.52, 0.54]

    col1, col2 = st.columns([5, 1])

    with col1:
        search_query = st.text_input(
            "Cerca per contesto nella DOMANDA:",
            placeholder=f"Inserisci almeno {min_chars} caratteri..."
        ).strip()

    with col2:
        st.markdown("**Soglia**")
        threshold = st.selectbox(
            label="Soglia similarità",
            options=threshold_options,
            index=2,
            format_func=lambda x: f"{x:.2f}",
            label_visibility="collapsed"
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
                    max_chars=None
                ),
                "RISPOSTA A": st.column_config.TextColumn(
                    "RISPOSTA A",
                    width="large",
                    max_chars=None
                ),
                "MATERIA": st.column_config.TextColumn(
                    "MATERIA",
                    width="small"
                ),
            },
        )

        if st.session_state.visible_rows < total_results:
            if st.button("Carica altri risultati"):
                st.session_state.visible_rows += batch_size
                st.rerun()

    with st.expander("Dettagli indice"):
        st.write(f"Righe indicizzate: {len(metadata_df)}")
        st.write(f"Shape embeddings: {embeddings.shape}")

except Exception as e:
    st.error(f"Errore: {e}")
