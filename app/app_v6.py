import os
from tempfile import NamedTemporaryFile
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# load environment variables (GOOGLE_API_KEY)
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# get extracted text from specified webpage as Documents
def extract_web_text(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs


# get extracted text from specified PDF as Documents
def extract_pdf_text(pdf):
    bytes_data = pdf.read()
    with NamedTemporaryFile(delete=False) as tmp:
        tmp.write(bytes_data)
        docs = PyPDFLoader(tmp.name).load()
    os.remove(tmp.name)
    return docs


# get extracted text from specified CSV as Documents
def extract_csv_text(csv):
    bytes_data = csv.read()
    with NamedTemporaryFile(delete=False) as tmp:
        tmp.write(bytes_data)
        docs = CSVLoader(file_path=tmp.name).load()
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
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    return vector_store


# embed PDF or CSV chunks into vectorstore
def get_file_vector_store(chunks):
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
            return input["question"]            # referenced as dictionary
        
    rag_chain = (                               # define the RAG pipeline to generate response
        RunnablePassthrough.assign(             # get the relevant docs to the question and chat_history
            context=contextualized_question | retriever | format_docs
        )
        | qa_prompt                             # pass to docs to prompt -> tell LLM what to achieve -> get response
        | llm
    )
    return rag_chain


# load db using its URI
def load_database(uri):
    database = SQLDatabase.from_uri(uri)        # connect to local MySQL instance using URI
    return database


# get SQL query from table and user question
def get_sql_chain(database):
    template = """Based on the table schema below as well as the relevant context, write an SQL query that would answer the user's question.
    Only return the SQL query that can instantly be ran against a MySQL Database:
    {schema}

    Context: {context}
    Question: {question}
    SQL Query
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0, convert_system_message_to_human=True)

    def get_schema(_):
        return database.get_table_info()

    sql_chain = (
        RunnablePassthrough.assign(schema=get_schema)   # assign value to a variable (value to replace in prompt). return value of get_schema -> schema
        | prompt                                        # pass value for 'schema' to 'prompt' from above
        | llm                                           
        | StrOutputParser()                             # return just contents (without 'AIMessage(...))
    )
    return sql_chain


# get natural language response of DB using SQL query and user question
def get_sql_full_pipeline(database, sql_chain):
    template = """ Based on the table schema below, question, SQL query and SQL response, write a response using natural language:
    {schema}

    Question: {question}
    SQL Query: {query}
    SQL Response: {response}
    """

    prompt = ChatPromptTemplate.from_template(template)     # final template to generate output
    llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0, convert_system_message_to_human=True)

    def get_schema(_):
        return database.get_table_info()

    def run_query(query):
        stripped_query = query.replace("```sql\n", "").replace("\n```", "").strip()
        return database.run(stripped_query)
    
    full_chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(             # query: returned by SQL chain
            schema=get_schema,                                          # schema: returned from get_schema(_)
            response=lambda vars: run_query(vars['query'])              # reference generated SQL query as variables['query']
        )
        | prompt                                                        # pass response to prompt (prompt now contains all required variables)
        | llm                                                           # pass prompt with variables to LLM to generate response in natural language
    )
    return full_chain


# define how to display chat messages
def display_messages():
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)


# save the chat history to a text file if button pressed
def save_button():
    text_file = open('chat_history.txt', mode='w')
    for i in range(len(st.session_state.chat_history)):
        if i%2 == 0:
            text_file.write('AIMessage(' + str(st.session_state.chat_history[i]) + ')\n')
        else:
            text_file.write('HumanMessage(' + str(st.session_state.chat_history[i]) + ')\n')
    text_file.close()

    with open('chat_history.txt') as f:
        st.download_button(
            label='Save Chat History to Text File',
            data=f,
            file_name='chat_history.txt',
            mime='text/plain',
        )
    os.remove('chat_history.txt')


# save the SQL Queries to a text file if button pressed
def download_sql():
    sql_file = open('sql_queries.txt', mode='w')
    for i in range(len(st.session_state.sql_queries)):
        sql_file.write(f"{st.session_state.sql_queries[i]}\n\n\n")
    sql_file.close()

    with open('sql_queries.txt') as f:
        st.download_button(
            label='Save SQL Queries to Text File',
            data=f,
            file_name='sql_queries.txt',
            mime='text/plain',
        )
    os.remove('sql_queries.txt')



# ----------------- UI ----------------------------
st.set_page_config(page_title="Data Chatter", page_icon="🤖")
st.title("Query Your Data")

# define how options for radio button appears
def custom_format(option):
    return f"{option}"

# instantiate chat_history if not done already (per session)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello, I am a chatbot used to query your data. How may I assist you?")]

# store SQL Queries
if 'sql_queries' not in st.session_state:
    st.session_state.sql_queries = ["Below are SQL Queries ran to Generate Responses: \n\n\n"]

# determine type of data source (either webpage or PDF)
with st.sidebar:
    st.header("Format of Knowledge Base")
    selected_option = st.radio("Type of Source:", ["Webpage", "PDF", "CSV", "MySQL DB"], format_func=custom_format)

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
        save_button()

# if PDF -> extract contents into Documents -> embed into vectorstore -> get response to query using vectorstore (and chat_history)
elif selected_option == "PDF":
    with st.sidebar:
        pdf = st.file_uploader("Upload a single local PDF", type=['pdf'])
    if pdf is not None:
        with st.spinner('Analyzing Data...'):
            if "pdf_vector_store" not in st.session_state:
                docs = extract_pdf_text(pdf=pdf)
                chunks = get_text_chunks(docs=docs)
                st.session_state.pdf_vector_store = get_file_vector_store(chunks=chunks)
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
        save_button()

# if CSV -> extract contents into Documents -> embed into vectorstore -> get response to query using vectorstore (and chat_history)
elif selected_option == "CSV":
    with st.sidebar:
        st.warning("Source CSV Files are currently Experimental! Take caution when viewing results!", icon="🚨")
        csv = st.file_uploader("Upload a single local CSV File", type=['csv'])
    if csv is not None:
        with st.spinner('Analyzing Data...'):
            if "csv_vector_store" not in st.session_state:
                docs = extract_csv_text(csv=csv)
                chunks = get_text_chunks(docs=docs)
                st.session_state.csv_vector_store = get_file_vector_store(chunks=chunks)
        csv_user_input = st.chat_input("Query CSV...")
        if csv_user_input is not None and csv_user_input != "":
            with st.spinner('Generating Response...'):
                csv_chain = get_response(vector_store=st.session_state.csv_vector_store)
                csv_response = csv_chain.invoke({
                    "question": csv_user_input, 
                    "chat_history": st.session_state.chat_history
                })
            st.session_state.chat_history.extend([
            HumanMessage(content=csv_user_input), 
            csv_response
            ])
        display_messages()
        save_button()

# if the option is MySQL Database (must be present in local instance)
elif selected_option == "MySQL DB":
    # store SQL Queries
    if 'sql_queries' not in st.session_state:
        st.session_state.sql_queries = ["Below are SQL Queries ran to Generate Responses: \n\n\n"]

    with st.sidebar:
        st.warning('MySQL Database has to already be present in a MySQL server', icon="⚠️")
        st.warning("If running from the cloud, MySQL data source is not supported (not 'localhost')", icon="⚠️")
        un = st.text_input("MySQL Username", autocomplete='root', placeholder="root")
        pw = st.text_input("MySQL Password", type='password', placeholder='pass')
        pn = st.text_input("MySQL Port Number", autocomplete='3306', placeholder="3306")
        db = st.text_input("Database Name", placeholder='MySQL DB Name')
    if (un is not None and un != "") and (pw is not None and pw != "") and (pn is not None and pn != "") and (db is not None and db != ""):
        db_uri = f"mysql+mysqlconnector://{un}:{pw}@localhost:{pn}/{db}"
        st.write(f"Querying Database {db}")
        with st.spinner('Injesting Database'):
            data = load_database(uri=db_uri)
            sql_query = get_sql_chain(database=data)
        sql_user_input = st.chat_input("Query MySQL Database...")
        if sql_user_input is not None and sql_user_input != "":
            query = sql_query.invoke({
                'context': st.session_state.chat_history,
                'question': sql_user_input
            })
            with st.spinner('Generating Response...'):
                nat_response = get_sql_full_pipeline(database=data, sql_chain=sql_query)
            nat_lang = nat_response.invoke({
                'context': st.session_state.chat_history,
                'question': sql_user_input              # question: pass from user
            })
            st.session_state.chat_history.extend([
                HumanMessage(content=sql_user_input), 
                nat_lang
            ])
            st.session_state.sql_queries.extend([
                query.replace("```sql\n", "").replace("\n```", "").strip()
            ])
        display_messages()
        save_button()
        download_sql()
    
    else:
        with st.sidebar:
            st.info('Fill in all fields in sidebar and press Enter', icon="ℹ️")