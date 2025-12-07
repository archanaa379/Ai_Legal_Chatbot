import os
from langchain_community.document_loaders import PyPDFLoader

# 📂 Folder where all your PDF files are stored
PDF_FOLDER = "archive/supreme_court_judgements/1950"

# 📄 Output file path
OUTPUT_FILE = "merged_output1.txt"

def load_and_merge_pdfs(pdf_folder):
    texts = []

    # Loop through all PDF files (sorted for consistency)
    for filename in sorted(os.listdir(pdf_folder)):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_folder, filename)
            print(f"Loading: {file_path}")

            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                # Combine all pages
                for d in docs:
                    texts.append(d.page_content)
            except Exception as e:
                print(f"⚠️ Skipping {file_path} due to error: {e}")

    return "\n\n".join(texts)

# ▶ Run merging
merged_text = load_and_merge_pdfs(PDF_FOLDER)

# 💾 Save to a single file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(merged_text)

print(f"\n✅ Merged PDF text saved to: {OUTPUT_FILE}")