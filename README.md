# RAG-Chatbot

A Corrective Retrieval-Augmented Generation-cRAG chatbot that allows users to chat with the content of uploaded pdf. It provides context-aware responses using LangChain memory. It also performs web search to answer user queries when retrieved context is insufficient.

For agent workflow, see [LangGraph Agent](src/agent.png)

### Hugging Face Model 

For finetuned model files, see [Prompt Classifier](https://huggingface.co/AminaAkhtar/Llama-3.2-1B-prompt-classifier)

## Technologies Used: 

LangChain, LangGraph, Pinecone, Streamlit

## How to run: 

```pip install -r requirements.txt```

Create .env file with PINECONE_API_KEY, GROQ_API_KEY, TAVILY_API_KEY

```streamlit run app.py```
