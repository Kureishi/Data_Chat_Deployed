import os
from tempfile import NamedTemporaryFile
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# load environment variables (GOOGLE_API_KEY)
from dotenv import load_dotenv
load_dotenv()

# get extracted text from specified webpage as Documents
def extract_web_text(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

# get extracted text from specified PDFs as Documents
def extract_pdf_text(pdf):
    bytes_data = pdf.read()
    with NamedTemporaryFile(delete=False) as tmp:
        tmp.write(bytes_data)
        docs = PyPDFLoader(tmp.name, extract_images=True).load()
    os.remove(tmp.name)
    return docs

# split docs into chunks
def get_text_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    return chunks

# embed web chunks into vectorstore
def get_web_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vector_store

# embed PDF chunks into vectorstore
def get_pdf_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    return vector_store

# get context for question (with relevant chat history) from data source
def get_context():
    llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0, convert_system_message_to_human=True)   # can change to whichever desired LLM
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(              # use above prompt to generate a query based on user's input and history
        [
            ("human", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()    # get pipeline to retrieve context as a chain
    return contextualize_q_chain

# get response using context from chat_history and current query for data source
def get_response(vector_store):
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\
    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages(               # get response using user's input and history
        [
            ("human", qa_system_prompt),                        # change type from 'system' to 'human' since initialize 'chat_history' with AIMessage
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    # how to present retrieved docs as the context
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    retriever = vector_store.as_retriever()
    llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0, convert_system_message_to_human=True)   # same model as in get_context()

    def contextualized_question(input: dict):   # define the context of question to use (whether to take into account chat_history)
        if input.get("chat_history"):
            return get_context()
        else:
            return input["question"]
        
    rag_chain = (                               # define the RAG pipeline to generate response
        RunnablePassthrough.assign(
            context=contextualized_question | retriever | format_docs
        )
        | qa_prompt
        | llm
    )
    return rag_chain

# define how to display chat messages
def display_messages():
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

# ----------------- UI ----------------------------
st.set_page_config(page_title="Data Chatter", page_icon="🤖")
st.title("Query Your Data")

# define how options for radio button appears
def custom_format(option):
    return f"{option}"

# instantiate chat_history if not done already (per session)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello, I am a chatbot used to query your data. How may I assist you?")]

# determine type of data source (either webpage or PDF)
with st.sidebar:
    st.header("Format of Knowledge Base")
    selected_option = st.radio("Type of Source:", ["Webpage", "PDF"], format_func=custom_format)

# if webpage -> extract contents into Documents -> embed into vectorstore -> get response to query using vectorstore (and chat_history)
if selected_option == "Webpage":
    with st.sidebar:
        webpage_url = st.text_input("Webpage URL")
    if webpage_url is not None and webpage_url != "":
        with st.spinner('Analyzing Data...'):
            if "web_vector_store" not in st.session_state:
                docs = extract_web_text(url=webpage_url)
                chunks = get_text_chunks(docs=docs)
                st.session_state.web_vector_store = get_web_vector_store(chunks=chunks)
        web_user_input = st.chat_input("Query Webpage...")
        if web_user_input is not None and web_user_input != "":
            with st.spinner('Generating Response...'):
                web_chain = get_response(vector_store=st.session_state.web_vector_store)
                web_response = web_chain.invoke({
                    "question": web_user_input, 
                    "chat_history": st.session_state.chat_history
                })
            st.session_state.chat_history.extend([
            HumanMessage(content=web_user_input), 
            web_response
            ])
        display_messages()

# if PDF -> extract contents into Documents -> embed into vectorstore -> get response to query using vectorstore (and chat_history)
elif selected_option == "PDF":
    with st.sidebar:
        pdf = st.file_uploader("Upload a single local PDF", type=['pdf'])
    if pdf is not None:
        with st.spinner('Analyzing Data...'):
            if "pdf_vector_store" not in st.session_state:
                docs = extract_pdf_text(pdf=pdf)
                chunks = get_text_chunks(docs=docs)
                st.session_state.pdf_vector_store = get_pdf_vector_store(chunks=chunks)
        pdf_user_input = st.chat_input("Query PDF...")
        if pdf_user_input is not None and pdf_user_input != "":
            with st.spinner('Generating Response...'):
                pdf_chain = get_response(vector_store=st.session_state.pdf_vector_store)
                pdf_response = pdf_chain.invoke({
                    "question": pdf_user_input, 
                    "chat_history": st.session_state.chat_history
                })
            st.session_state.chat_history.extend([
            HumanMessage(content=pdf_user_input), 
            pdf_response
            ])
        display_messages()

