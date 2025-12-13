from typing import List,Literal
from typing_extensions import TypedDict, Dict
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel, Field
from langchain_tavily import TavilySearch
import os
from dotenv import load_dotenv
from src.embeddings import retrieve_embeddings
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"]=GROQ_API_KEY

class Evaluator(BaseModel):
    """Classify retrieved documents based on how relevant it is to the question."""
    binary_score: str = Field(description="Documents are relevant to the question, Only respond with 'Yes' or 'No'.")
   
class GraphState(TypedDict):
     """ Represents the state of the graph
         Attributes:
            question: question
            answer: LLM answer
            web_search: whether to add web search
            documents: list of relevant documents
            """
     question: str
     answer:str
     web_search: str
     documents:List[str]

class ChatAgent:
    def __init__(self):
        self.llm=ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.5,
            max_tokens=None
            )
        self.retriever=retrieve_embeddings()
        self.web_search_tool = TavilySearch(k=3)
        self.llm_evaluator= self.llm.with_structured_output(Evaluator)
        self.checkpointer=InMemorySaver()
        self.app=self.generate_graph()
        agent = self.app.get_graph().draw_mermaid_png()
        with open("src/agent.png", "wb") as f:
             f.write(agent)

    def generate_graph(self)->GraphState:
        workflow=StateGraph(GraphState)
        # nodes
        workflow.add_node("retrieve",self.retrieve)
        workflow.add_node("evaluate",self.evaluate_documents)
        workflow.add_node("generate",self.generate)
        workflow.add_node("transform_query",self.transform_query)
        workflow.add_node("web_search",self.web_search)
        # edges
        workflow.add_edge(START,"retrieve")
        workflow.add_edge("retrieve","evaluate")
        workflow.add_conditional_edges("evaluate",self.decide_to_generate,{
            "transform_query":"transform_query",
            "generate":"generate",
            },
        )
        workflow.add_edge("transform_query","web_search")
        workflow.add_edge("web_search","generate")
        workflow.add_edge("generate",END)
        app=workflow.compile(checkpointer=self.checkpointer)
        return app
  
    def retrieve(self,state:GraphState)->GraphState:
        """Retrieve relevant chunks from PineCone vector database"""
        question = state["question"]
        documents = self.retriever.invoke(question)
        state["documents"]=documents
        return state

    def generate(self,state:GraphState)->GraphState:
        """Generate answer based on question, retrieved documents and chat history"""
        question = state["question"]
        documents = state["documents"]
        context = "\n".join([d.page_content for d in documents])
        generate_prompt=f"""You are an expert assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        Question:{question}
        Context:{context}
        Keep the answer concise."""
        response = self.llm.invoke(generate_prompt)
        state["answer"]=response.content
        return state

    def evaluate_documents(self,state:GraphState)->GraphState:
       """Determines whether the retrieved documents are relevant to the question."""
       question = state["question"]
       documents = state["documents"]
       filtered_docs = []
       web_search = "No"
       for d in documents:
         evaluate_prompt=f"""You are a document retrieval evaluator that's responsible for checking the relevancy of a retrieved document to the user's question.  
           If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. 
           Question: {question}
           Document: {d.page_content}
           Output should be 'yes' or 'no' ONLY.
           """
         response = self.llm_evaluator.invoke(evaluate_prompt)
         if response.binary_score.lower() == "yes": # document relevant
            filtered_docs.append(d)
       if len(documents) > 0 and (len(filtered_docs) / len(documents) <= 0.6):
          web_search = "Yes"
       state["web_search"]=web_search
       state["documents"]=filtered_docs
       return state

    def transform_query(self,state:GraphState)->GraphState:
        """Transform the query for web search."""
        question = state["question"]
        transform_prompt=f"""You are a question re-writer that converts an input question to a better version that is optimized 
        for web search. Return the tansformed question only.
        Question:{question}"""
        web_question = self.llm.invoke(transform_prompt)
        state["question"]=web_question.content
        return state

    def web_search(self,state:GraphState)->GraphState:
        """Performs web search based on the rephrased question."""
        question = state["question"]
        documents = state["documents"]
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join(docs)
        web_results = Document(page_content=web_results)
        documents.append(web_results)
        state["documents"]=documents
        return state

    def decide_to_generate(self,state:GraphState)->str:
        """Determines whether to generate an answer, or re-generate a question. """
        web_search = state["web_search"]
        if web_search == "Yes":
            return "transform_query"
        else:
            return "generate"

    def run(self,userinput:str)->GraphState:
        initial_state: GraphState={
          "question": userinput,
          "documents": [],
          "answer": "",
          "web_search": "No",
          }
        thread = {"configurable": {"thread_id": "1"}}
        result= self.app.invoke(initial_state,config=thread)
        return result["answer"]
        # return result
        
# python -m src.langgraph_agent
# ensure there are embeddings in vector store
# agent=ChatAgent()
# response = agent.run("What is the architecture of efficient net b0")
# print(response)