# =====================================
# BUILD CIGNA POLICY COLLECTION (2-STAGE, PERSISTENT)
# =====================================

import os
import pdfplumber
import chromadb
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

# --- 1. Load API Key from .env ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 2. Paths & Configs ---
PDF_FOLDER = r"C:\Users\Vincent Lin\OneDrive\Desktop\MVP_PA_Policies\data\raw\pdf_CIGNA_PA_policies"
TEXT_FOLDER = r"C:\Users\Vincent Lin\OneDrive\Desktop\MVP_PA_Policies\data\processed\text_CIGNA"
CHROMA_PATH = r"C:\Users\Vincent Lin\OneDrive\Desktop\MVP_PA_Policies\db\chroma_CIGNA"
os.makedirs(TEXT_FOLDER, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

# --- 3. Initialize Persistent ChromaDB ---
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection_name = "cigna_policies"

try:
    collection = chroma_client.get_collection(collection_name)
    print(f"✅ Loaded existing collection '{collection_name}'.")
except Exception:
    collection = chroma_client.create_collection(name=collection_name)
    print(f"🆕 Created new collection '{collection_name}'.")

# --- 4. Tokenizer setup ---
enc = tiktoken.get_encoding("cl100k_base")

def split_text_into_chunks(text, max_tokens=2000):
    """Split text into safe chunks for embedding."""
    tokens = enc.encode(text)
    return [enc.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]

# --- 5. Stage 1: Convert PDFs → Text Files ---
def convert_pdfs_to_text():
    for pdf_file in os.listdir(PDF_FOLDER):
        if not pdf_file.lower().endswith(".pdf"):
            continue

        text_file_path = os.path.join(TEXT_FOLDER, pdf_file.replace(".pdf", ".txt"))
        if os.path.exists(text_file_path):
            print(f"✅ Text already exists for {pdf_file}")
            continue

        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            with open(text_file_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"📝 Extracted text from {pdf_file}")
        except Exception as e:
            print(f"⚠️ Failed to process {pdf_file}: {e}")

# --- 6. Stage 2: Embed Text Files into ChromaDB ---
def embed_text_files():
    for txt_file in os.listdir(TEXT_FOLDER):
        if not txt_file.endswith(".txt"):
            continue

        text_path = os.path.join(TEXT_FOLDER, txt_file)
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            print(f"⚠️ Empty text file skipped: {txt_file}")
            continue

        num_tokens = len(enc.encode(text))
        print(f"{txt_file}: {num_tokens} tokens")

        # Split into safe chunks
        chunks = split_text_into_chunks(text, max_tokens=2000)

        for idx, chunk in enumerate(chunks):
            emb = client.embeddings.create(
                model="text-embedding-3-large",
                input=chunk
            ).data[0].embedding

            collection.add(
                documents=[chunk],
                embeddings=[emb],
                metadatas=[{"payer": "CIGNA", "file_name": txt_file, "chunk": idx}],
                ids=[f"{txt_file}_chunk_{idx}"]
            )

        print(f"✅ Added {len(chunks)} chunks from {txt_file}")

# --- 7. Run Both Stages ---
if __name__ == "__main__":
    print("=== Stage 1: PDF → Text ===")
    convert_pdfs_to_text()

    print("\n=== Stage 2: Embedding into Persistent ChromaDB ===")
    embed_text_files()

    print(f"\n✅ Completed embedding all Cigna text files into '{collection_name}' collection.")
    print(f"📦 ChromaDB path: {CHROMA_PATH}")
    print(f"📊 Total records: {collection.count()}")
