import streamlit as st
import requests
from src.config import Config
from src.logger import logging as log
from src.exception import CustomException

config = Config()
API_URL = config.FAST_API_URL

st.set_page_config(page_title="Invoice Policy Comparator", layout="centered")
st.title("Invoice Policy Comparator & Chatbot")

st.markdown("---")
st.subheader("Upload Invoice & Policy")

# File uploaders
invoice_zip = st.file_uploader("Upload Invoice ZIP", type="zip")
policy_pdf = st.file_uploader("Upload Policy PDF", type="pdf")

if st.button('Analyse'):
    if invoice_zip and policy_pdf:
        try:
            with st.spinner("Processing uploaded documents..."):
                files = {
                    "invoice_file": invoice_zip,
                    "policy_file": policy_pdf
                }

                log.info(f"In app.py:: Sending files to {API_URL}/process_claim/")
                response = requests.post(url=f"{API_URL}/process_claim/", files=files)

                if response.status_code == 200:
                    st.success("Documents processed and stored in vector DB.")
                else:
                    log.error(f"Upload error: {response.text}")
                    st.error(f"Error processing documents: {response.text}")
        except CustomException as e:
            log.error(f"Upload error: {str(e)}")
            st.error(f"An error occurred: {e}")

st.markdown("---")
st.subheader("Chat with Invoice Bot")

user_query = st.text_input("Ask something like 'Show rejected invoices for Gaurav in March'")

with st.expander("Optional Filters"):
    employee_name = st.text_input("Employee Name")
    status = st.selectbox("Reimbursement Status", ["", "Accepted", "Rejected", "Pending"])
    date = st.text_input("Invoice Date (YYYY-MM-DD)")

if st.button("Ask"):
    if user_query.strip() == "":
        st.warning("Please enter a question.")
    else:
        try:
            metadata_filter = {}
            if employee_name:
                metadata_filter["employee_name"] = employee_name
            if status:
                metadata_filter["status"] = status
            if date:
                metadata_filter["date"] = date

            response = requests.post(
                url=f"{API_URL}/chat/",
                params={"query": user_query},
                json={"metadata_filter": metadata_filter}
            )

            if response.status_code == 200:
                st.markdown("###Bot's Answer")
                st.markdown(response.json()["response"])
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            log.error(f"Chat error: {str(e)}")
            st.error("An error occurred during the chat interaction.")
