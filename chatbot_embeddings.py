import pandas as pd
import boto3
from io import StringIO
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import pickle

bucket_name = 'interns-aws-experimentation-bucket'
file_key = 'gen-ai/Mental_Health_FAQ.csv'
index_key = 'gen-ai/faiss_index.pkl'  # Single consistent key for the vector store

# Function to load dataset from S3
def load_dataset_from_s3(bucket_name, file_key):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
    return df

# Function to create and update the vector store and save it to S3
def save_embeddings_to_s3():
    s3 = boto3.client('s3')
    
    # Load the dataset
    df = load_dataset_from_s3(bucket_name, file_key)

    # Create or update the vector store
    print("Creating or updating vector store...")
    texts = df['Answers'].tolist()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.create_documents(texts)
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save the vector store to S3
    index_bytes = pickle.dumps(vectorstore)
    s3.put_object(Bucket=bucket_name, Key=index_key, Body=index_bytes)
    print(f"Saved vector store to S3: {index_key}")

# Execute the function to create and save embeddings
save_embeddings_to_s3()
