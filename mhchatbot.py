import streamlit as st
import boto3
import pickle
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Ollama
from langchain.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from botocore.exceptions import ClientError

bucket_name = 'interns-aws-experimentation-bucket'
index_prefix = 'gen-ai/faiss_index_'

# Function to load the vector store from S3
@st.cache_resource
def load_vectorstore_from_s3():
    s3 = boto3.client('s3')
    
    # List all existing index files
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=index_prefix)
    existing_indexes = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.pkl')]
    
    if existing_indexes:
        # Sort the indexes by name and get the most recent
        most_recent_index = sorted(existing_indexes)[-1]
        try:
            # Try to load the most recent index from S3
            response = s3.get_object(Bucket=bucket_name, Key=most_recent_index)
            index_bytes = response['Body'].read()
            vectorstore = pickle.loads(index_bytes)
            print(f"Loaded existing vector store from S3: {most_recent_index}")
            return vectorstore
        except ClientError as e:
            st.write(f"Error loading index: {e}")
            return None
    else:
        st.write("No vector store found in S3.")
        return None

vectorstore = load_vectorstore_from_s3()

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
    mental_health_keywords = ['mental health', 'depression', 'anxiety', 'therapy', 'counseling', 'stress', 'psychiatry', 'psychology', 'psychosis', 'mental disorder', 'post-traumatic stress disorder', 'ptsd', 'trauma']
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
