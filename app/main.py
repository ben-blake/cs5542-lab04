import sys
import os

# Add the parent directory to sys.path so we can import 'rag'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import time
import json
import requests
from datetime import datetime
from rag.pipeline import (
    evaluate_retrieval, # We still need this for evaluation logic
    DATA_DIR
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multimodal RAG Lab 4",
    page_icon="🤖",
    layout="wide"
)

LOG_FILE = "logs/query_metrics.csv"
os.makedirs("logs", exist_ok=True)

# -----------------------------------------------------------------------------
# Predefined Queries
# -----------------------------------------------------------------------------
PREDEFINED_QUERIES = [
    # Evidence IDs are FILENAMES that contain the answer.
    # Users should verify these match their actual file names in project_data_mm/
    {
        "id": "Q1",
        "question": "Based on the risk matrix shown in the figures and the accompanying text, which combination of likelihood and impact corresponds to the highest risk level?",
        "gold_evidence_ids": ["impact_likelihood_matrix.png", "risk_management.png", "doc1.pdf"]
    },
    {
        "id": "Q2",
        "question": "Using both the Zero Trust architecture diagram and the document text, what core principle is emphasized for access decisions?",
        "gold_evidence_ids": ["zero_trust.png", "doc4.pdf"]
    },
    {
        "id": "Q3",
        "question": "What specific encryption algorithm (for example, AES-256 or RSA-2048) is mandated by the organization’s policy?",
        "gold_evidence_ids": ["doc5.pdf"]
    },
    {
        "id": "Q4",
        "question": "What are the key components of the Zero Trust Architecture as depicted in the diagram?",
        "gold_evidence_ids": ["zero_trust.png"]
    },
    {
        "id": "Q5",
        "question": "What is the specific budget allocation for the cybersecurity initiative?",
        "gold_evidence_ids": [] # Intentionally empty to test missing evidence behavior
    }
]

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def log_metrics(record):
    """Appends a record to the CSV log file."""
    df = pd.DataFrame([record])
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)


# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------
def main():
    st.title("🛡️ Multimodal RAG System (Lab 4)")
    st.markdown("""
    This application demonstrates a **Multimodal Retrieval-Augmented Generation** pipeline.
    It retrieves evidence from PDFs and Images to answer user queries.
    """)

    # --- Sidebar Controls ---
    st.sidebar.header("⚙️ Configuration")
    
    # Backend Configuration (Always FastAPI)
    api_url = "http://127.0.0.1:8000"
    
    st.sidebar.markdown("### Backend Status")
    if st.sidebar.button("Check API Connection"):
        try:
            r = requests.get(f"{api_url}/health", timeout=2)
            if r.status_code == 200:
                st.sidebar.success("API is Online ✅")
            else:
                st.sidebar.error(f"API Error: {r.status_code}")
        except Exception as e:
            st.sidebar.error(f"Connection Failed: {e}")
            st.sidebar.caption("Ensure `uvicorn api.server:app` is running.")

    # Retrieval Settings


    # Retrieval Settings
    method = st.sidebar.selectbox(
        "Retrieval Method",
        ["sparse", "bm25", "dense", "hybrid", "hybrid_rerank"],
        index=3
    )
    
    chunking = st.sidebar.radio("Chunking Strategy", ["page", "fixed"], index=0)
    
    st.sidebar.markdown("### Parameters")
    top_k_text = st.sidebar.slider("Top K Text", 1, 20, 5)
    top_k_images = st.sidebar.slider("Top K Images", 0, 10, 3)
    top_k_evidence = st.sidebar.slider("Total Evidence (Context)", 1, 20, 8)
    
    # --- Query Section ---
    st.header("1. Query")
    
    query_mode = st.radio("Select Query Source:", ["Predefined", "Custom"], horizontal=True)
    
    selected_query_obj = None
    user_query = ""
    
    if query_mode == "Predefined":
        q_options = [f"{q['id']}: {q['question']}" for q in PREDEFINED_QUERIES]
        selected_q_str = st.selectbox("Choose a query:", q_options)
        selected_query_obj = next(q for q in PREDEFINED_QUERIES if f"{q['id']}: {q['question']}" == selected_q_str)
        user_query = selected_query_obj["question"]
    else:
        user_query = st.text_input("Enter your question:")

    if st.button("🚀 Run RAG Pipeline", type="primary"):
        if not user_query:
            st.warning("Please enter a query.")
            return

        start_time = time.time()
        
        result_context = ""
        result_evidence = []
        result_image_paths = []
        generated_answer = ""
        result_evidence_ids = []

        try:
            # FastAPI Call (Always)
            with st.spinner("Calling API..."):
                payload = {
                    "query": user_query,
                    "method": method,
                    "chunking": chunking,
                    "top_k_text": top_k_text,
                    "top_k_images": top_k_images,
                    "top_k_evidence": top_k_evidence
                }
                resp = requests.post(f"{api_url}/query", json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                
                generated_answer = data["answer"]
                result_context = data["context"]
                result_evidence = data["evidence"]
                result_image_paths = data["image_paths"]
                # Extract IDs from evidence list
                result_evidence_ids = [e["id"] for e in result_evidence]

        except Exception as e:
            st.error(f"Error executing query: {e}")
            st.info("Make sure the API server is running: `uvicorn api.server:app --reload`")
            return
        
        latency = time.time() - start_time
        
        # 3. Evaluation
        gold_ids = selected_query_obj.get("gold_evidence_ids", []) if selected_query_obj else []
        precision, recall = evaluate_retrieval(result_evidence, gold_ids)
        
        # Faithfulness check (heuristic)
        faithfulness = 1.0
        if "Not enough evidence" in generated_answer:
            if not gold_ids:
                 faithfulness = 1.0 # Correct behavior for missing evidence test
            else:
                 faithfulness = 0.0 # Failed to answer
        
        # 4. Display Results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📝 Generated Answer")
            st.markdown(generated_answer)
            
            st.subheader("🔍 Retrieved Evidence")
            with st.expander("Show Context", expanded=True):
                for line in result_context.splitlines():
                    if "[IMAGE" in line:
                         st.markdown(f"🖼️ **{line}**")
                    else:
                         st.markdown(f"📄 {line}")

        with col2:
            st.subheader("📊 Metrics")
            st.caption(f"Retrieval: **{method}**")
            st.metric("Latency", f"{latency:.3f}s")
            
            if gold_ids:
                st.metric("Precision@5", f"{precision:.2f}")
                st.metric("Recall@10", f"{recall:.2f}")
            elif selected_query_obj:
                 st.info("No gold evidence defined (Missing Evidence Test).")
            else:
                 st.info("Select a predefined query to see automated metrics.")

            st.metric("Faithfulness Pass", "Yes" if faithfulness == 1.0 else "No")

            if result_image_paths:
                st.subheader("🖼️ Retrieved Images")
                for img_path in result_image_paths:
                    try:
                        st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not load image: {img_path}")

        # 5. Logging (columns per CS5542 Lab 4 Notebook)
        # missing_evidence_behavior: Pass if (no gold + missing-msg) or (gold + grounded); Fail otherwise
        has_gold = bool(gold_ids)
        said_missing = "Not enough evidence" in generated_answer
        if not has_gold:
            meb = "Pass" if said_missing else "Fail"
        else:
            meb = "Pass" if not said_missing else "Fail"

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "query_id": selected_query_obj["id"] if selected_query_obj else "custom",
            "retrieval_mode": method,
            "top_k": top_k_evidence,
            "latency_ms": round(latency * 1000, 2),
            "Precision@5": round(precision, 3),
            "Recall@10": round(recall, 3),
            "evidence_ids_returned": ";".join(result_evidence_ids),
            "faithfulness_pass": "Yes" if faithfulness == 1.0 else "No",
            "missing_evidence_behavior": meb,
        }
        log_metrics(log_data)
        st.toast("Query logged successfully!", icon="💾")

    # --- Logs View ---
    with st.expander("📈 View Evaluation Logs"):
        if os.path.exists(LOG_FILE):
            st.dataframe(pd.read_csv(LOG_FILE).sort_values("timestamp", ascending=False))
        else:
            st.info("No logs found yet.")

if __name__ == "__main__":
    main()
