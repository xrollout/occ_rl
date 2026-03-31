#!/usr/bin/env python3
"""Test LLM API connection"""

import os
import sys
from openai import OpenAI
import dotenv

# Load env
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    dotenv.load_dotenv(env_path, override=True)

api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN")
base_url = os.environ.get("ANTHROPIC_BASE_URL")
model = os.environ.get("ANTHROPIC_MODEL", "doubao-seed-1-8-251228")

print(f"Testing API connection...")
print(f"Base URL: {base_url}")
print(f"Model: {model}")
print(f"API key: {api_key[:8]}...")

try:
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello world and output vx: 0.3, vy: 0.0, omega: 0.0"},
        ],
        temperature=0.0,
        max_tokens=64,
    )
    print("\nSUCCESS!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
