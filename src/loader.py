from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def upload_document(file):
    with open("PDF/" + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_document(doc_path):
    reader= PdfReader(doc_path)
    document= ""
    for i in range(len(reader.pages)):
        page= reader.pages[i]
        text=page.extract_text()
        if text:
            document += text+"\n"
    return document

def split_document(document):
    split = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    # convert pdf text into list of Document
    docs = [Document(page_content=document)]
    text = split.split_documents(docs)
    #texts = [t.page_content for t in text]
    return text