import streamlit as st 
from PyPDF2 import PdfReader  
import os   
import tempfile  # Create temporary files
import json  
from datetime import datetime  

# langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.llms import Ollama 
from langchain_community.vectorstores import FAISS  
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_community.embeddings import OllamaEmbeddings  
from langchain.prompts import ChatPromptTemplate

 


def read_data(files, loader_type):
    documents = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name

        try:
            if loader_type == "PDF":
                pdf_reader = PdfReader(tmp_file_path)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    documents.append(Document(page_content=text, metadata={"source": file.name, "page_number": page_num + 1}))
            elif loader_type == "Text":
                loader = TextLoader(tmp_file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file.name
                documents.extend(docs)
            elif loader_type == "CSV":
                loader = CSVLoader(tmp_file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file.name
                documents.extend(docs)
        finally:
            os.remove(tmp_file_path)
    return documents

# Split text into chunks
def get_chunks(texts, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for text in texts:
        split_texts = text_splitter.split_text(text.page_content)
        for split_text in split_texts:
            chunks.append(Document(page_content=split_text, metadata=text.metadata))
    return chunks

# Store chunks in a vector store
def vector_store(text_chunks, embedding_model_name, vector_store_path):
    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model=embedding_model_name)
    vector_store = FAISS.from_texts(texts=[doc.page_content for doc in text_chunks], embedding=embeddings, metadatas=[doc.metadata for doc in text_chunks])
    vector_store.save_local(vector_store_path)

# Load the vector store 
def load_vector_store(embedding_model_name, vector_store_path):
    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model=embedding_model_name)
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store

# Save history of chat
def save_conversation(conversation, vector_store_path):
    conversation_path = os.path.join(vector_store_path, "conversation_history.json")
    with open(conversation_path, "w") as f:
        json.dump(conversation, f, indent=4)

# Load history 
def load_conversation(vector_store_path):
    conversation_path = os.path.join(vector_store_path, "conversation_history.json")
    if os.path.exists(conversation_path):
        with open(conversation_path, "r") as f:
            conversation = json.load(f)
    else:
        conversation = []
    return conversation


def document_to_dict(doc):
    return {
        "metadata": doc.metadata
    }


def get_conversational_chain(retriever, ques, llm_model, system_prompt):
    llm = Ollama(model=llm_model, verbose=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True  
    )
    response = qa_chain.invoke({"query": ques})
    return response

# user input handling and display content
def user_input(user_question, embedding_model_name, vector_store_path, num_docs, llm_model, system_prompt):
    vector_store = load_vector_store(embedding_model_name, vector_store_path)
    retriever = vector_store.as_retriever(search_kwargs={"k": num_docs})
    response = get_conversational_chain(retriever, user_question, llm_model, system_prompt)
    
    conversation = load_conversation(vector_store_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if 'result' in response:
        result = response['result']
        source_documents = response['source_documents'] if 'source_documents' in response else []
        conversation.append({
            "question": user_question, 
            "answer": result, 
            "timestamp": timestamp, 
            "llm_model": llm_model,
            "source_documents": [document_to_dict(doc) for doc in source_documents]
        })
        st.write("Reply: ", result)
        st.write(f"**LLM Model:** {llm_model}")
        
        st.write("### Source Documents")
        for doc in source_documents:
            metadata = doc.metadata
            st.write(f"**Source:** {metadata.get('source', 'Unknown')}, **Page Number:** {metadata.get('page_number', 'N/A')}, **Additional Info:** {metadata}")
        st.markdown("<hr style='border:1px solid gray;'>", unsafe_allow_html=True)
    else:
        conversation.append({"question": user_question, "answer": response, "timestamp": timestamp, "llm_model": llm_model})
        st.write("Reply: ", response)
    
    save_conversation(conversation, vector_store_path)
    
    st.write("### Conversation History")
    for entry in sorted(conversation, key=lambda x: x['timestamp'], reverse=True):
        st.write(f"**Q ({entry['timestamp']}):** {entry['question']}")
        st.write(f"**A:** {entry['answer']}")
        st.write(f"**LLM Model:** {entry['llm_model']}")
        if 'source_documents' in entry:
            for doc in entry['source_documents']:
                st.write(f"**Source:** {doc['metadata'].get('source', 'Unknown')}, **Page Number:** {doc['metadata'].get('page_number', 'N/A')}, **Additional Info:** {doc['metadata']}")  # Display source filename, page number, and additional metadata
        st.markdown("<hr style='border:1px solid gray;'>", unsafe_allow_html=True)

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.header("Chat with PDFs using Llama3 ")
     

    st.sidebar.markdown(
        """
        <div style="text-align: center;">
            <img src="https://th.bing.com/th/id/OIP.tZ0EH_Yi857WlxDiKDr6nAHaE7?w=255&h=180&c=7&r=0&o=5&dpr=1.3&pid=1.7" style="width: 300px; height: auto;">
        </div>
        """,
        unsafe_allow_html=True
    )

    # Add system prompt 
    system_prompt = st.sidebar.text_area("System Prompt", value= "You are a friendly and helpful chatbot designed to assist users with their questions and needs. Your primary goal is to provide accurate information, offer solutions, and engage users in a pleasant manner. Always be polite, clear, and concise. If you encounter a question or request you cannot handle, guide the user to appropriate resources or suggest they contact a human representative if necessary.")

    user_question = st.text_input("Ask a Question to the bot about your files.")
   

    embedding_model_name = "mistral:instruct"
      

    llm_model = "mistral:instruct" 

    vector_store_path = st.sidebar.text_input("Conversations are stored in :", "../data/vectorstore/my_store")


    data_type = st.sidebar.selectbox(
        "Select Data Type",
        ["PDF", "Text", "CSV"]
    )


    chunk_text = st.sidebar.checkbox("Chunk Text", value=True)
    chunk_size = 1000 
    chunk_overlap = 200 
    num_docs = 3 

    if user_question:
        user_input(user_question, embedding_model_name, vector_store_path, num_docs, llm_model, system_prompt)

    with st.sidebar:
        st.title("Documents:")
        data_files = st.file_uploader("Upload Files Here", accept_multiple_files=True)
        if st.button("Submit Here"):
            with st.spinner("Processing..."):
                raw_documents = read_data(data_files, data_type)
                if chunk_text:
                    text_chunks = get_chunks(raw_documents, chunk_size, chunk_overlap)
                else:
                    text_chunks = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in raw_documents]
                vector_store(text_chunks, embedding_model_name, vector_store_path)
                st.success("Done")
    
    # Footer with three columns
    st.markdown("<hr>", unsafe_allow_html=True)
    

if __name__ == "__main__":
    main()