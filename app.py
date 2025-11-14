import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from src.helper import *

prompt="You are an expert assistant to answer the questions based on the pdf content. " \
"Use only the following extracted information  to answer the questions. " \
"If you do not know the answer say: I could not find any relevant information in the document. " \
"Keep the answers concise, use 3 sentences maximum." \
"{context}"

load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')
os.environ["GROQ_API_KEY"]=GROQ_API_KEY

index_name="custom-chatbot"
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings=load_embeddings()

#run only one time to create index
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(       
       name=index_name,
       dimension=384, 
       metric="cosine", 
       spec=ServerlessSpec(
         cloud="aws", 
         region="us-east-1") 
    ) 
    print(f"Index '{index_name}' created successfully")
else:
    print(f"Index '{index_name}' already exists")

template=ChatPromptTemplate.from_messages(
    [
        ("system", prompt),
        ("human", "{input}"),
    ]
)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key='answer' ,
    k=3
)

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
    st.session_state.embedding_upload=False
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

st.title("RAG Chatbot")
st.write("Upload a PDF file to start chat")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], label_visibility="collapsed")

if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    upload_document(uploaded_file)
    document = load_document("PDF/" + uploaded_file.name)
    split_document=split_document(document)
    st.session_state.embedding_upload=False    
    if not st.session_state.embedding_upload:
        #store embeddings
        index_data = PineconeVectorStore.from_documents(
        documents=split_document,
        index_name=index_name,
        embedding=embeddings )
        st.session_state.embedding_upload=True
    
elif st.session_state.uploaded_file is not None and uploaded_file is None:
    file_path = os.path.join("PDF", st.session_state.uploaded_file.name)
    if os.path.exists(file_path):
        os.remove(file_path)
    index = pc.Index(index_name)
    index.delete(delete_all=True)
    st.session_state.uploaded_file = None
    st.session_state.conversation_history = []
    st.session_state.embeddings_upload=False
    st.rerun()

load_doc = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings)

retriever = load_doc.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5,
    max_tokens=None
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, 
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

st.divider()

for message in st.session_state.conversation_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Type your message..."):
    st.session_state.conversation_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if st.session_state.uploaded_file is None:
        response= "Please upload a PDF file to start chat."
        st.info(response)
    else:
        with st.spinner("Thinking"):
         response=qa_chain.invoke({"question":user_input})
         response=response["answer"]
         with st.chat_message("assistant"):
          st.markdown(response)

    st.session_state.conversation_history.append({"role": "assistant", "content": response})
