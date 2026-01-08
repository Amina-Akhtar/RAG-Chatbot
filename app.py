import streamlit as st
from src.embeddings import *
from src.loader import load_document, upload_document, split_document
from src.langgraph_agent import ChatAgent  

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
    st.session_state.embedding_upload=False
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'agent' not in st.session_state:
        st.session_state.agent = ChatAgent()
st.title("Secure-cRAG")
st.write("Upload a PDF file to start chat")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], label_visibility="collapsed")

if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    upload_document(uploaded_file)
    document = load_document("PDF/" + uploaded_file.name)
    chunks=split_document(document)
    st.session_state.embedding_upload=False   
    if not st.session_state.embedding_upload:
        #store embeddings
        store_embeddings(chunks)
        st.session_state.embedding_upload=True
    
elif st.session_state.uploaded_file is not None and uploaded_file is None:
    file_path = os.path.join("PDF", st.session_state.uploaded_file.name)
    if os.path.exists(file_path):
        os.remove(file_path)
    delete_index()
    st.session_state.uploaded_file = None
    st.session_state.conversation_history = []
    st.session_state.embeddings_upload=False
    st.rerun()

st.divider()

for message in st.session_state.conversation_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            with st.expander("Debug Information"):
                st.markdown("Prompt Classification:")
                st.write(message["debug"]["label"])
                st.markdown("Web Search:")
                st.write(message["debug"]["web_search"])
                st.markdown("Retrieved Context:")
                for doc in message["debug"]["documents"]:
                    if isinstance(doc, str):
                        st.write(doc)
                    else:
                        st.write(doc.page_content)
        st.markdown(message["content"])                

if user_input := st.chat_input("Type your message..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.conversation_history.append({
        "role": "user",
        "content": user_input
    })
    if st.session_state.uploaded_file is None:
        response= "Please upload a PDF file to start chat."
        st.info(response)
    else:
        with st.spinner("Thinking"):
            result = st.session_state.agent.run(user_input)
            label = result["prompt_label"]
            web_search = result["web_search"]
            documents = result.get("documents", [])
            if label == 'Benign':
               response = result["answer"]
            else:
               response = "The question has been identified as potentially malicious and cannot be processed. Consider rephrasing it or try a different question."
            with st.expander("Debug Information"):
                st.markdown("Prompt Classification:")
                st.write(label)
                st.markdown("Web Search:")
                st.write(web_search)
                st.markdown("Retrieved Context:")
                for doc in documents:
                    if isinstance(doc, str):
                        st.write(doc)
                    else:
                        st.write(doc.page_content)
            with st.chat_message("assistant"):
               st.markdown(response)
            st.session_state.conversation_history.append({
            "role": "assistant",
            "content": response,
            "debug": {
              "label": label,
              "web_search": web_search,
              "documents": documents}
            })

# python -m streamlit run app.py    