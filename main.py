import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from src.run_analysis import InvoicePolicyComparator
from src.logger import logging as log
from src.exception import CustomException
from uuid import uuid4
from src.utils import get_data_to_embed
from src.vector_store.db import VectorStore
from src.rag_agent import graph
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, List, Optional
from pydantic import BaseModel
from src.config import Config
from langchain import hub
from langgraph.graph import START, END, StateGraph, MessagesState
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List
from src.logger import logging as log


app = FastAPI()

invoice_compare = InvoicePolicyComparator()
llm = init_chat_model("llama3-8b-8192", model_provider="groq")
graph_builder = StateGraph(MessagesState)
prompt = hub.pull("rlm/rag-prompt")
config = Config()
vector_store = VectorStore()


@app.post("/process_claim/")
async def process_claim(invoice_file: UploadFile = File(), policy_file: UploadFile = File())->bool:
    """FastAPI endpoint to process claim analysis.
    
    Params:
        invoice_file: uploaded invoice zip file.
        policy_file: uploaded policy pdf file.
    
    Returns:
        dictionary of results.
    """

    try:
        # Save files temporarily
        with tempfile.TemporaryDirectory() as temp_directory:
            log.info("BACKEND::Inside Temp directory")
            zip_path = os.path.join(temp_directory, invoice_file.filename)
            policy_path = os.path.join(temp_directory, policy_file.filename)
            log.info(f"zip path: {zip_path}, \npdf path: {policy_path}")

            with open(zip_path, "wb") as f:
                f.write(await invoice_file.read())

            with open(policy_path, "wb") as f:
                f.write(await policy_file.read())

            # Process claim
            log.info(f"About to analyse {invoice_file.filename} and {policy_file.filename}")
            decisions, invoice_texts = invoice_compare.process_zip_and_analyse(
                                zip_file_path=zip_path, 
                                policy_path=policy_path
                            )
            log.info(f"Analysed {invoice_file.filename} and {policy_file.filename}")
            
            documents = get_data_to_embed(decisions=decisions, invoice_texts=invoice_texts)
            log.info("Documnets prepared for Embedding with metadata")
            
            vector_store.add_documents(documents=documents)
            log.info("Storing Dicuments to Vector Store and Returning TRUE")

        return True
    
    except CustomException as e:
        log.error(f"{str(e)}")
        return False
    


class ChatRequest(BaseModel):
    query: str
    metadata_filter: Optional[Dict] = None

class ChatResponse(BaseModel):
    response: str
    status: str
    details: Optional[str] = None

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}


graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()



@app.post("/chat/", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    log.info("Inside CHatbot")
    try:
        # Initialize metadata_filter if None
        # metadata_filter = request.metadata_filter or {}
        
        # Create input messages
        input_messages = [HumanMessage(content=request.query)]
        log.info(f"Input Message:: {input_messages}")

        # Invoke the graph
        result = graph.invoke({
            "messages": input_messages,
            # "metadata_filter": metadata_filter
        })

        # Extract the AI response
        ai_messages = [msg for msg in result['messages'] if msg.type == 'ai']
        if not ai_messages:
            return ChatResponse(
                status="error",
                response="No response generated by the chatbot",
                details=str(result)
            )

        final_response = ai_messages[-1].content
        log.info(f"Final Response:: {final_response}")
        
        return ChatResponse(
            status="success",
            response=final_response,
            # metadata=metadata_filter
        )

    except Exception as e:
        return ChatResponse(
            status="error",
            response="Chatbot failed to process your query",
            details=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
