import streamlit as st
import pandas as pd
import boto3
from io import StringIO
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Ollama
from langchain.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Function to load dataset from S3
@st.cache_resource
def load_dataset_from_s3(bucket_name, file_key):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
    return df

# Load your dataset from S3
bucket_name = 'interns-aws-experimentation-bucket'
file_key = 'gen-ai/Mental_Health_FAQ.csv'
df = load_dataset_from_s3(bucket_name, file_key)

# Prepare the data
@st.cache_resource
def prepare_vectorstore():
    texts = df['Answers'].tolist()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.create_documents(texts)
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

vectorstore = prepare_vectorstore()

# Initialize the chat models
@st.cache_resource
def init_chat_models():
    mh_qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOllama(model="llama2"),
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    general_llm = ChatOllama(model="llama2")
    return mh_qa_chain, general_llm

mh_qa_chain, general_llm = init_chat_models()

# Function to classify if a question is mental health-related
def classify_question(question):
    mental_health_keywords = ['mental health', 'depression', 'anxiety', 'therapy', 'counseling', 'stress', 'psychiatry', 'psychology']
    is_mental_health = any(keyword in question.lower() for keyword in mental_health_keywords)
    return is_mental_health

# Streamlit app
st.title("General and Mental Health Chatbot")

# Initialize session state for conversation histories
if 'general_history' not in st.session_state:
    st.session_state.general_history = [
        SystemMessage(content="You are a helpful assistant with broad knowledge on various topics. Please respond to general questions on any subject.")
    ]
if 'mental_health_history' not in st.session_state:
    st.session_state.mental_health_history = [
        SystemMessage(content="You are a helpful assistant with specialized knowledge about mental health. Please respond to questions related to mental health.")
    ]

# Function to handle user input
def handle_input():
    user_input = st.session_state.user_input
    if user_input:
        is_mental_health = classify_question(user_input)
        combined_response = ""

        if is_mental_health:
            # Use mental health QA chain and history
            result = mh_qa_chain({
                "question": user_input, 
                "chat_history": [(m.content, "") for m in st.session_state.mental_health_history if isinstance(m, (HumanMessage, AIMessage))]
            })
            mh_response = result['answer']
            combined_response += mh_response
            # Add to mental health history
            st.session_state.mental_health_history.append(HumanMessage(content=user_input))
            st.session_state.mental_health_history.append(AIMessage(content=mh_response))

        else:
            # Use general LLM and history
            general_prompt = f"Please answer the following question to the best of your ability: {user_input}"
            response = general_llm.invoke(st.session_state.general_history + [HumanMessage(content=general_prompt)])
            general_response = response.content
            combined_response += general_response
            # Add to general history
            st.session_state.general_history.append(HumanMessage(content=user_input))
            st.session_state.general_history.append(AIMessage(content=general_response))

        # Clear the input
        st.session_state.user_input = ""

# Display conversation history
st.write("Conversation History:")
for message in st.session_state.general_history + st.session_state.mental_health_history:
    if isinstance(message, HumanMessage):
        st.write(f"You: {message.content}")
    elif isinstance(message, AIMessage):
        st.write(f"Assistant: {message.content}")

# Input form and on_change handler
st.text_input("Ask any question!", key="user_input", on_change=handle_input)
