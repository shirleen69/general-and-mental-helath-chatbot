import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCasualLM,pipeline
import torch

#load model and create the pipeline
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCasualLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=100)
llm = HuggingFacePipeline(pipeline=pipe)

#set up the streamlit framework
st.title('Example langchain with LLAMA2 model')
input_text = st.text_input('What is your question?')


#create a ChatPrompts Template
template = ChatPromptTemplate.from_messages([
    ("User: ", "{input}")
])

#Generate response when input is provided
if input_text:
    #Create the chain
    chain = template | llm
    
    #Run the chain
    response = chain.invoke({"input": input_text})
    
    #Display the response
    st.write(response)
