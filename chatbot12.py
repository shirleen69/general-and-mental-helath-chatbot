import boto3
import os
from sagemaker.huggingface import HuggingFaceModel

# Set up AWS credentials and region
os.environ['AWS_DEFAULT_REGION'] = 'eu-west-1'  # Replace with your preferred region

# Initialize SageMaker and STS clients
sagemaker_client = boto3.client('sagemaker')
sts_client = boto3.client('sts')

# Get the AWS account ID and IAM role ARN
account_id = sts_client.get_caller_identity()['Account']
role_arn = f'arn:aws:iam::{account_id}:role/Sagemaker-Endpoint-Creation-Role'

# Define the model parameters
hub = {
    'HF_MODEL_ID': 'meta-llama/Llama-2-7b-chat-hf',  # Ensure this model ID is correct
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
instance_type = "ml.g4dn.xlarge"  # Adjust as needed for model size

# Deploy the model
print("Starting model deployment...")
try:
    predictor = huggingface_model.deploy(initial_instance_count=1, instance_type=instance_type)
    print("Model deployment successful")
except Exception as e:
    print(f"Error during model deployment: {str(e)}")
    raise

# Chatbot function
def chatbot(input_text):
    try:
        response = predictor.predict({
            "inputs": input_text,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
        })
        return response[0]['generated_text'].strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Main loop
print("Chatbot: Hello! I'm a simple LLaMA-based chatbot. Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    response = chatbot(user_input)
    print(f"Chatbot: {response}")

# Clean up: delete the endpoint
print("Cleaning up...")
try:
    predictor.delete_endpoint()
    print("Endpoint deleted. Goodbye!")
except Exception as e:
    print(f"Error during endpoint deletion: {str(e)}")
