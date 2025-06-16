import os
import PyPDF2
import re
import sys
import json
import zipfile
import tempfile
import PyPDF2.errors
from groq import Groq
from io import BytesIO
from src.config import Config
from src.prompt import LLM_prompt_template
from src.logger import logging as log
from src.exception import CustomException
from typing import List, Union
from src.utils import clean_invoice


config = Config()


class InvoicePolicyComparator:
    def __init__(self):
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.model = config.LLM_MODEL
        self.temperature = config.LLM_TEMPERATURE

    @staticmethod
    def extract_text_from_pdf(pdf_path: str):
        """Extract text from a PDF file using an in-memory approach."""
        text = ''
        with open(pdf_path, 'rb') as file:
            pdf_bytes = file.read()
            with BytesIO(pdf_bytes) as pdf_stream:
                reader = PyPDF2.PdfReader(pdf_stream)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
        return text.strip()

    @staticmethod
    def clean_text(text):
        """Remove excessive whitespace and newlines."""
        return re.sub(r'\s+', ' ', text).strip()
    

    def analyse_invoice_against_policy(self, invoice_text_data: str, policy_text_data: str)-> json:
        """Compare invoice with policy and get reimbursement decision."""
        
        prompt = LLM_prompt_template(invoice_text=invoice_text_data, policy_text=policy_text_data)
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                response_format={"type": "json_object"},
                temperature=self.temperature
            ) 
            content = response.choices[0].message.content
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    return {"error": "No valid JSON found", "raw_response": content}

            required_fields = ["customer_name", "reimbursement_status", "reason", "date", "invoice_ID", "policy_references"]
            for field in required_fields:
                if field not in result:
                    result[field] = "Unknown"  # providing default values to avoid failure

            if result["reimbursement_status"] not in ["accept", "partially accept", "reject"]:
                result["reimbursement_status"] = "Unknown"

            return result
        
        except Exception as e:
            log.error(f"{str(e)}")
            return {"error": str(e), "raw_response": content if 'content' in locals() else None}


    def process_zip_and_analyse(self, zip_file_path: str, policy_path: str)->Union[List[dict], List[str]]:
        """Extracts PDFs from ZIP, processes each file, and compares against policy document.
        Returns:
            tuple: (list of comparison results, list of extracted invoice texts)
        """
        policy_text = self.clean_text(self.extract_text_from_pdf(policy_path))
        results = []
        decisions = []
        
        temp_dir = None  # Initializing to ensure it exists for finally block
        try:
            temp_dir = tempfile.TemporaryDirectory()
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                log.info(f"Extracting zip: {zip_file_path} to {temp_dir}")
                zip_ref.extractall(temp_dir.name)

            for root, _, files in os.walk(temp_dir.name):
                for filename in files:
                    if filename.lower().endswith(".pdf"):
                        invoice_path = os.path.join(root, filename)
                        try:
                            # invoice_text = self.clean_text(self.extract_text_from_pdf(invoice_path))
                            invoice_text = clean_invoice(self.extract_text_from_pdf(invoice_path))
                            
                            results.append(invoice_text)                           
                            
                            decision = self.analyse_invoice_against_policy(
                                invoice_text_data=invoice_text, 
                                policy_text_data=policy_text
                                ) # too many requests to groq API client (!!!Need to Handle that!!!). Log shows ~10 req in 2 sec
                            
                            decisions.append(decision)
                        
                        except PyPDF2.errors.PdfReadError:
                            log.error(f"Could not read PDF: {invoice_path}")
                        
                        except Exception as path_err:
                            log.error(f"Failed to process {invoice_path}: {path_err}")
        
        except Exception as zip_process_error:
            log.error(f"Error during ZIP processing: {zip_process_error}")
            raise CustomException(zip_process_error, sys)
        
        finally:
            if temp_dir:
                try:
                    temp_dir.cleanup()
                except Exception as cleanup_error:
                    log.error(f"Error cleaning up temporary directory: {cleanup_error}")
        
        if len(decisions)>0 and len(results)>0:
            log.info("Successful analysis report prepared.")   
        else:
            log.info(f"Unsuccessful Extraction::Total descisions: {len(decisions)}, Total results: {len(results)}")    
        
        return decisions, results
