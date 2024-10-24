# chatbot.py
# Import necessary modules
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import streamlit as st

# Initialize the Streamlit framework
st.title('Langchain Chatbot With LLAMA2 model')  # Set the title of the Streamlit app

# Initialize session state to store conversation history
if 'history' not in st.session_state:
    st.session_state.history = [
        ("system", "You are a helpful assistant. Please respond to the questions")
    ]

# Define a prompt template for the chatbot
def build_prompt(history):
    return ChatPromptTemplate.from_messages(history)

# Initialize the Ollama model
llm = Ollama(model="llama2")

# Function to handle input submission and update conversation
def handle_input():
    input_text = st.session_state.input_text
    if input_text:
        st.session_state.history.append(("user", f"Question: {input_text}"))
       
        prompt = build_prompt(st.session_state.history)
        chain = prompt | llm
        response = chain.invoke({"question": input_text})
       
        st.session_state.history.append(("assistant", response))
       
        # Clear the input field after processing the question
        st.session_state.input_text = ""

# Display the conversation history
conversation = ""
for role, text in st.session_state.history:
    if role == "user":
        conversation += f"**You:** {text.split('Question: ')[-1]}\n"
    elif role == "assistant":
        conversation += f"**Assistant:** {text}\n"

st.text_area("Conversation", value=conversation, height=400, max_chars=None, key="conversation_history", disabled=True)

# Create a text input field in the Streamlit app for new questions with a callback
st.text_input("Ask your question!", key="input_text", on_change=handle_input)