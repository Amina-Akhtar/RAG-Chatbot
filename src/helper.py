from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

def upload_document(file):
    with open("PDF/" + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_document(doc_path):
    loader = PDFPlumberLoader(doc_path)
    document = loader.load()
    return document

def split_document(doc_name):
    split_document = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    text = split_document.split_documents(doc_name)
    #texts = [t.page_content for t in text]
    return text

def load_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings