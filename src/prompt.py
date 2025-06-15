def LLM_prompt_template(invoice_text: str, policy_text: str) -> dict:
    """Compare invoice with policy and get reimbursement decision with robust JSON handling."""
    
    # Prepare the prompt with strict JSON formatting
    prompt = f'''You're Insurance claims analyst. Analyze this invoice against the policy and provide a response in EXACTLY this JSON format:
```json
{{
    "customer_name": "Customer Name here",
    "reimbursement_status": "accept | partially accept | reject",
    "reason": "Detailed explanation with Specific policy clauses and Approved Amount here",
    "date": "Invoice Date Here",
    "invoice_ID": "Invoice ID Here",
    "invoice_text": "specify invoice text content here"
}}

Policy Document:
{policy_text}

Invoice Details:
{invoice_text}

Important Rules:
1. All fields must be present
2. Values must be in double quotes
3. "reimbursement_status" must be one of: accept, partially accept, reject
4. Do not include any text outside the JSON brackets
5. First Name, Second Name and Thrid Name will all strat with a capital Letter.
6. Take care of extra spaces within the first Name. DO not break first name into parts (second name will start with a capital letter) for example **A njane y a K** is **Anjaneya K**
7. Ensure the 'reason' field in the JSON does not contain unescaped quotes or special characters.

Take care of some broken words:
1. **Cust omer Name** is **Customer Name**
2. **Inv oice Date** is **Date**
3. **Inv oice ID** is **Invoice ID**

'''
    return prompt