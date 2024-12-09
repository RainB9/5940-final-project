import openai
# from os import environ
import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from PyPDF2 import PdfReader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.vectorstores import FAISS

# Streamlit setup
st.title("Genius Bot")
st.caption("Powered by Yuanze Bao")

# Set up OpenAI proxy client
client = openai.OpenAI(
    api_key="sk-fU_9e80K6l4Erj8Ls_KlHQ",  
    base_url="api.ai.it.cornell.edu"  
)

# Initialize Streamlit session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# File uploader
uploaded_files = st.file_uploader("Upload a file", type=("txt", "pdf"), accept_multiple_files=True)

question = st.chat_input("Ask something")

# Clear system cache if needed
# chromadb.api.client.SharedSystemClient.clear_system_cache()

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    if uploaded_files:
        combined_text = ""
        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith(".pdf"):
                pdf_reader = PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    combined_text += page.extract_text()
            elif uploaded_file.name.endswith(".txt"):
                combined_text += uploaded_file.read().decode("utf-8")

        # Vectorstore and context setup
        documents = [Document(page_content=combined_text)]
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

        # Prompt setup
        template = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

        Question: {question} 

        Context: {context} 

        Answer:
        """
        prompt = PromptTemplate.from_template(template)

        # Build the RAG chain
        rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | client.chat.completions.create  # Use chat completion API
            | StrOutputParser()
        )

        # Retrieve and process documents
        relevant_docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Generate response
        messages = {
            "model": "gpt-3.5-turbo",  # Specify the model
            "messages": [{"role": "user", "content": f"Question: {question}\n\nContext: {context}"}],
        }
        response = client.chat.completions.create(**messages)

        st.chat_message("assistant").write(response['choices'][0]['message']['content'])
        st.session_state.messages.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
        
    else:
        # Direct response without context
        messages = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": question}],
        }
        response = client.chat.completions.create(**messages)
        st.chat_message("assistant").write(response['choices'][0]['message']['content'])
        st.session_state.messages.append({"role": "assistant", "content": response['choices'][0]['message']['content']})

        # response = client([HumanMessage(content=question)])
        # st.chat_message("assistant").write(response.content)
        # st.session_state.messages.append({"role": "assistant", "content": response.content})