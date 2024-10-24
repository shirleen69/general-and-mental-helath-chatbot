import json

# Load the first aid dataset
with open('intents.json') as f:
    intents = json.load(f)

# Inspect the data structure
print(intents)
