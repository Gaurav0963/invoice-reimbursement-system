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
from langchain_core.messages import HumanMessage


vector_store = VectorStore()
invoice_compare = InvoicePolicyComparator()
app = FastAPI()


@app.post("/process_claim/")
async def process_claim(invoice_file: UploadFile = File(), policy_file: UploadFile = File())->dict:
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
    

@app.post("/chat/")
async def chat_with_bot(query: str, metadata_filter: dict = {}):
    """
    Endpoint to interact with the RAG-based invoice chatbot.
    
    Params:
        query: Natural language query
        metadata_filter: Optional metadata filtering e.g., {"employee_name": "Gaurav", "status": "Rejected"}
    
    Returns:
        RAG-generated answer
    """
    try:
        input_messages = [HumanMessage(content=query)]

        result = graph.invoke({"messages": input_messages, "metadata_filter": metadata_filter})

        final_response = [msg for msg in result['messages'] if msg.type == 'ai'][-1].content
        return {"response": final_response}

    except Exception as e:
        log.error(f"RAG Chat Error: {str(e)}")
        raise CustomException("Chatbot failed to respond.", e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
