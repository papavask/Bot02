# rag_chatbot_streamlit_cloud.py
import streamlit as st
import chromadb
import requests
import time
import os

class SimpleRAGChatbot:
    def __init__(self, db_path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection("pdf_documents")
        self.hf_token = os.getenv("HF_API_TOKEN")  # Set in Streamlit secrets
    
    def search_documents(self, query, n_results=3):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results['documents'][0] if results['documents'] else [], []
    
    def hf_llm_response(self, query, context_chunks):
        """Use Hugging Face Inference API"""
        if not context_chunks:
            return "I couldn't find relevant information in the documents."
        
        context = "\n".join([f"--- Document {i+1} ---\n{chunk}" for i, chunk in enumerate(context_chunks)])
        
        prompt = f"""Based on the following documents, answer the question. If the documents don't contain relevant information, say "I don't have enough information from the documents to answer this."

DOCUMENTS:
{context}

QUESTION: {query}

ANSWER:"""
        
        try:
            response = requests.post(
                "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
                headers={"Authorization": f"Bearer {self.hf_token}"},
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 200,
                        "temperature": 0.7,
                        "do_sample": True
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result[0]['generated_text'].replace(prompt, "").strip()
            else:
                return f"Error from LLM API: {response.status_code}"
                
        except Exception as e:
            return f"Error connecting to LLM service: {str(e)}"

def response_generator(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.03)

def main():
    st.title("📚 Document Chatbot - Streamlit Cloud")
    
    # Initialize
    if 'chatbot' not in st.session_state:
        try:
            st.session_state.chatbot = SimpleRAGChatbot()
        except Exception as e:
            st.error(f"Database error: {e}")
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                documents, _ = st.session_state.chatbot.search_documents(prompt)
                response_text = st.session_state.chatbot.hf_llm_response(prompt, documents)
            
            response = st.write_stream(response_generator(response_text))
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
