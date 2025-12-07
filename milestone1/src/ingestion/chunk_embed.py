import os
import hashlib
import logging
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

# ---------- config ----------
load_dotenv()
PDF_FOLDER = "data"
INDEX_NAME = "legal-index"
BATCH_SIZE = 100

# ---------- logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- Pinecone ----------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

# ---------- Embeddings ----------
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cuda"}  # change to "cpu" if you want CPU
)

# ---------- helpers ----------
def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def infer_act_from_filename(fname: str) -> str:
    f = fname.lower()
    if "ipc" in f or "penal" in f: return "Indian Penal Code"
    if "crpc" in f or "criminal" in f: return "Code of Criminal Procedure"
    if "evidence" in f: return "Indian Evidence Act"
    if "pocso" in f: return "POCSO Act"
    if "contract" in f: return "Indian Contract Act"
    if "domestic" in f: return "Domestic Violence Act"
    if "motor" in f: return "Motor Vehicles Act"
    if "negotiable" in f or "ni act" in f: return "Negotiable Instruments Act"
    if "juvenile" in f: return "Juvenile Justice Act"
    if "ndps" in f: return "NDPS Act"
    if "it" in f or "information technology" in f: return "Information Technology Act"
    if "constitution" in f: return "Constitution of India"
    return "Unknown Act"

def sanitize_metadata(m: dict) -> dict:
    md = {}
    for k, v in m.items():
        if k == "page_number":
            if v is None:
                md[k] = -1
            else:
                try:
                    md[k] = int(v)
                except Exception:
                    md[k] = -1
            continue
        if k == "chunk_index":
            try:
                md[k] = int(v)
            except Exception:
                md[k] = -1
            continue
        if k == "file_hash":
            md[k] = str(v) if v is not None else ""
            continue
        if k in ("source_pdf", "source_act"):
            md[k] = str(v) if v is not None else ""
            continue
        if k == "text":
            txt = str(v) if v is not None else ""
            md[k] = txt[:2000]
            continue
        if isinstance(v, (str, int, float, bool)):
            md[k] = v
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            md[k] = v
        else:
            md[k] = str(v) if v is not None else ""
    return md

# ---------- index a single PDF ----------
def index_pdf(file_path: str):
    logging.info("Indexing PDF: %s", file_path)
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    file_name = os.path.basename(file_path)
    file_hash = file_md5(file_path)
    act_name = infer_act_from_filename(file_name)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = splitter.split_documents(pages)
    logging.info(" - %s generated chunks", len(chunks))

    batch = []
    for i, chunk in enumerate(chunks):
        raw_page = chunk.metadata.get("page_number") or chunk.metadata.get("page") or None

        metadata = {
            "text": chunk.page_content,
            "source_pdf": file_name,
            "source_act": act_name,
            "page_number": raw_page,
            "file_hash": file_hash,
            "chunk_index": i
        }

        safe_md = sanitize_metadata(metadata)
        vector = embeddings.embed_documents([chunk.page_content])[0]

        vec_obj = {
            "id": f"{file_name}-chunk-{i}",
            "values": vector,
            "metadata": safe_md
        }
        batch.append(vec_obj)

        if len(batch) >= BATCH_SIZE:
            try:
                index.upsert(vectors=batch)
                logging.info("Upserted %d vectors", len(batch))
            except Exception as e:
                logging.exception("Failed to upsert batch: %s", e)
            batch = []

    if batch:
        try:
            index.upsert(vectors=batch)
            logging.info("Upserted final %d vectors", len(batch))
        except Exception as e:
            logging.exception("Failed to upsert final batch: %s", e)

    logging.info("Finished indexing %s", file_name)

# ---------- delete vectors for a removed/updated PDF ----------
def delete_pdf_vectors(file_name: str):
    logging.info("Deleting vectors for %s", file_name)
    try:
        index.delete(filter={"source_pdf": {"$eq": file_name}})
        logging.info("Deleted vectors for %s", file_name)
    except Exception as e:
        logging.exception("Error deleting vectors for %s: %s", file_name, e)

# ---------- index all PDFs ----------
def index_all_pdfs():
    logging.info("Scanning folder: %s", PDF_FOLDER)
    for fname in os.listdir(PDF_FOLDER):
        if not fname.lower().endswith(".pdf"):
            continue
        fp = os.path.join(PDF_FOLDER, fname)
        delete_pdf_vectors(fname)
        index_pdf(fp)
    logging.info("All PDFs processed.")

# ---------- run ----------
if __name__ == "__main__":
    index_all_pdfs()