import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load PDF
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDF")
    return documents

# Step 2: Split text into chunks
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

# Step 3: Create FAISS vector store using HuggingFace embeddings
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("Vector store created")
    return vectorstore

# Step 4: Retrieve similar chunks
def retrieve(vectorstore, query, k=3):
    results = vectorstore.similarity_search(query, k=k)
    print(f"\nQuery: {query}\n")
    for i, doc in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(doc.page_content.encode('utf-8', errors='ignore').decode('utf-8'))
        print()

# Main
if __name__ == "__main__":
    pdf_path = "D:\\NLP\\vijaygpt\\FSD_dharsh_report.pdf"

    documents = load_pdf(pdf_path)
    chunks = split_documents(documents)
    vectorstore = create_vectorstore(chunks)

    query = "What is this document about?"
    retrieve(vectorstore, query)
