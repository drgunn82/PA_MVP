import streamlit as st
from chromadb import PersistentClient
from openai import OpenAI
from dotenv import load_dotenv
import os

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()  # Load .env file from the project root
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found. Please create a .env file with your API key.")
    st.stop()

# --- CONFIGURATION ---
DB_PATH = r"C:\Users\Vincent Lin\OneDrive\Desktop\MVP_PA_Policies\data\db\chroma_UHC"
COLLECTION_NAME = "pa_policies"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"

# --- DEBUG INFO ---
st.write("🔍 Current working directory:", os.getcwd())
st.write("🔍 ChromaDB path:", DB_PATH)
st.write("🔍 Path exists:", os.path.exists(DB_PATH))

# --- INITIALIZE CLIENTS ---
try:
    client = PersistentClient(path=DB_PATH)
    collections = client.list_collections()
    st.write("📚 Collections found:", collections)
    collection = client.get_collection(COLLECTION_NAME)
    st.success(f"Connected to ChromaDB collection: {COLLECTION_NAME}")
except Exception as e:
    st.error(f"Error connecting to ChromaDB: {e}")
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- STREAMLIT UI ---
st.set_page_config(page_title="PA Policy Scanner (UHC)", layout="wide")
st.title("🧠 PA Policy Scanner – UnitedHealthcare")
st.write("Ask any question about UHC prior authorization policies. The app retrieves policy text and summarizes it using GPT-4o.")

query = st.text_input("Enter your question:", placeholder="e.g. What are the PA criteria for late stage lung cancer drugs?")
n_results = st.slider("Number of results to retrieve", 1, 10, 3)

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("🔎 Retrieving and analyzing policies..."):

            # 1️⃣ Create query embedding
            embedding = openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=query
            ).data[0].embedding

            # 2️⃣ Query Chroma collection
            results = collection.query(
                query_embeddings=[embedding],
                n_results=n_results
            )

            docs = [doc for sub in results["documents"] for doc in sub]
            context = "\n\n".join(docs)

            # 3️⃣ Construct the LLM prompt
            prompt = f"""
You are a managed care policy analyst specializing in prior authorization.
Use only the UHC policy context below to answer factually and concisely.

Context:
{context}

Question:
{query}

Answer clearly and professionally. If the context is incomplete, state that directly.
            """

            # 4️⃣ Generate response
            response = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            answer = response.choices[0].message.content.strip()

            # 5️⃣ Display results
            st.markdown("### 🧾 **Answer**")
            st.markdown(answer)

            # 6️⃣ Show retrieved policy snippets for transparency
            st.markdown("---")
            st.markdown("### 🔍 **Retrieved Policy Snippets**")
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                st.markdown(f"**Source:** {meta.get('source', 'Unknown Source')}")
                st.write(doc[:600] + "...")
                st.markdown("---")
