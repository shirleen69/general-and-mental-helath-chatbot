import boto3
import json
import os
from sagemaker.huggingface import HuggingFaceModel

# Set up AWS credentials and region
os.environ['AWS_DEFAULT_REGION'] = 'eu-west-1'  # Replace with your preferred region

# Initialize SageMaker and STS clients
sagemaker_client = boto3.client('sagemaker')
sts_client = boto3.client('sts')

# Get the AWS account ID and IAM role ARN
account_id = sts_client.get_caller_identity()['Account']
 # Replace with your actual SageMaker execution role name
role_arn = f'arn:aws:iam::####:role/Sagemaker-Endpoint-Creation-Role'

# Define the model parameters
hub = {
    'HF_MODEL_ID': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'HF_TASK': 'text-generation'
}

# Create HuggingFaceModel instance
huggingface_model = HuggingFaceModel(
    transformers_version="4.28.1",
    pytorch_version="2.0.0",
    py_version="py310",
    env=hub,
    role=role_arn
)

# Define the instance type for deployment
instance_type = "ml.g4dn.xlarge"  # This instance type has a GPU

print("Starting model deployment...")
predictor = huggingface_model.deploy(initial_instance_count=1, instance_type=instance_type)
print("Model deployed successfully!")

# Chatbot function
def chatbot(input_text):
    try:
        response = predictor.predict({
            "inputs": f"Human: {input_text}\nAssistant:",
            "parameters": {
                "max_new_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True,
                "repetition_penalty": 1.1
            }
        })
        return response[0]['generated_text'].split("Assistant:")[-1].strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Main loop
print("Chatbot: Hello! I'm a simple TinyLlama-based chatbot. Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    response = chatbot(user_input)
    print(f"Chatbot: {response}")

# Clean up: delete the endpoint
print("Cleaning up...")
predictor.delete_endpoint()
print("Endpoint deleted. Goodbye!")