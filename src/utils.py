from langchain_core.documents import Document
from typing import List
from src.logger import logging as log
import re


def get_data_to_embed(decisions: List[dict], invoice_texts: List[str]) -> List[Document]:
    """
    Converts analysis decisions and invoice texts into LangChain Documents for vector storage
    
    Params:
        decisions: List of decision dictionaries from analysis
        invoice_texts: List of corresponding invoice text contents
            
    Returns:
        List of LangChain Documents ready for vector storage
    """
    documents = []
    
    if len(decisions) != len(invoice_texts):
        log.error(f"Mismatched input lengths: {len(decisions)} decisions vs {len(invoice_texts)} invoice texts")
        raise ValueError("Decisions and invoice texts must be of equal length")
    
    for decision, invoice_text in zip(decisions, invoice_texts):
        try:
            # Extract core fields with defaults
            status = decision.get("reimbursement_status", "unknown").lower()
            reason = decision.get("reason", "No reason provided")
            name = decision.get("customer_name", "Unknown")
            if name != "Unknown":
                employee_name = get_correct_name(name)
            
            # Prepare document content
            text_to_embed = f"Invoice Content: {invoice_text}, Status: {status}, Reason: {reason}"
            
            # Prepare metadata
            metadata = {
                "invoice_id": decision.get("invoice_ID", "unknown"),
                "status": status,
                "reason": reason,
                "employee_name": employee_name,
                "date": decision.get("date", "Unknown")
            }
            
            # Creating Langchain Document
            documents.append(Document(
                page_content=text_to_embed,
                metadata=metadata
            ))
            
        except Exception as e:
            log.error(f"Error processing decision {decision}: {str(e)}")
            continue
            
    return documents


def clean_invoice(text):
    """Fix broken words and normalize spacing in extracted text."""
    # Fix common broken patterns (e.g., "A njane y a K" -> "Anjaneya K")
    text = re.sub(r'(\b[A-Za-z])\s+([a-z]\b)', r'\1\2', text)  # Fix name fragments
    text = re.sub(r'(\b[A-Za-z]{2})\s+([a-z]+\b)', r'\1\2', text)  # Fix words like "Inv oice"
    
    # Fix specific known issues
    text = re.sub(r'T ax', 'Tax', text)
    text = re.sub(r'Inv oice', 'Invoice', text)
    text = re.sub(r'Cust omer', 'Customer', text)
    text = re.sub(r'Addr ess', 'Address', text)
    text = re.sub(r'Ser vice', 'Service', text)
    text = re.sub(r'Categor y', 'Category', text)
    text = re.sub(r'Driv er', 'Driver', text)
    text = re.sub(r'T rip', 'Trip', text)
    text = re.sub(r'La y out', 'Layout', text)
    text = re.sub(r'Char ges', 'Charges', text)
    text = re.sub(r'Conv enience', 'Convenience', text)
    text = re.sub(r'Descri ption', 'Description', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_correct_name(broken_name: str)->str:
    '''
    This function keeps adding letters to first name until the next capital letter is encontered.
    similarly for second/last name. This function assumes a person has only First ann/or Last Name.
    
    Param: broken_name: Takes a broken string of name(e.g., A n ja yna e K -> Anjaynae K)
    
    Returns: string'''
    tokens = broken_name.split()
    if not tokens:
        return ""
    
    merged_name = []
    i = 0
    n = len(tokens)
    
    while i < n:
        if tokens[i][0].isupper():  # Capitalized fragment found
            current = tokens[i]
            i += 1
            # Merge until next capital or end
            while i < n and not tokens[i][0].isupper():
                current += tokens[i]
                i += 1
            merged_name.append(current)
        else:
            i += 1  # Skipping non-capitalized (unlikely for names)
    possible_name = " ".join(merged_name)
    name = possible_name.split()
    if len(name)>2:
        return ' '.join(name[:2])
    return possible_name