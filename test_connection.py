import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("FIREWORKS_API_KEY")

if not api_key:
    print("❌ API key not found! Please check your .env file")
else:
    print(f"✅ API key loaded: fw_{'*' * 20}")
    
    # Test connection
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.fireworks.ai/inference/v1"
    )
    
    try:
        # Simple test query
        response = client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3p1-8b-instruct",
            messages=[{"role": "user", "content": "Say 'Hello, Fireworks!'"}],
            max_tokens=10
        )
        print(f"✅ API connection successful!")
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ API connection failed: {e}")