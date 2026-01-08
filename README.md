# Secure-cRAG

A Corrective Retrieval-Augmented Generation-cRAG chatbot that allows users to chat with the content of uploaded pdf. It delivers context-aware answers using LangGraph agent workflow by performsing web search when document context is insufficient, and proactively blocks malicious prompts to ensure secure interactions.

For agent workflow, see [LangGraph Agent](src/agent.png)

For finetuned model files, see [Hugging Face Prompt Classifier](https://huggingface.co/AminaAkhtar/Llama-3.2-1B-prompt-classifier)

## Technologies Used: 

LangChain, LangGraph, Pinecone, Streamlit

## How to run: 

```pip install -r requirements.txt```

Create .env file with PINECONE_API_KEY, GROQ_API_KEY, TAVILY_API_KEY

```streamlit run app.py```
