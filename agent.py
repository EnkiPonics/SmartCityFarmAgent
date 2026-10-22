import os
import http.client
import json
from dotenv import load_dotenv
from pathlib import Path
# Load environment variables from configs/.env file
env_file = Path(__file__).parent.parent / "configs" / ".env"
print(f"Loading environment variables from: {env_file}")
if env_file.exists():
    load_dotenv(env_file)

introduction_llm = """For the OSI Hackathon 2026, the user has written a comprehensive EnkiPonics sensor data analysis and LSTM modeling
with data from SmartCityFarm e.V. (SCF) Berlin <3."""

def agent_chat(
    history,
    model: str = "nvidia/nemotron-3-ultra-550b-a55b:free",
    max_tokens=475,
    text="Your initial text here",
):
    # Add the assistant's initial message to the beginning of the history
    history.insert(0, {"role": "assistant", "content": introduction_llm})
    # Add the user's initial message to the history
    history.insert(1, {"role": "user", "content": text})


    # --- NATIVE OPENROUTER NETWORK CALL ---

    # 1. Establish network connection
    conn = http.client.HTTPSConnection("openrouter.ai")

    # 2. Structure payload with your randomized variables and chat history
    payload_data = {
        "model": model,
        "messages": history,
        "max_tokens": max_tokens,
    }

    # 3. Secure headers using your OpenRouter token
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENROUTER_KEY')}",
        "Content-Type": "application/json",
    }

    try:
        # 4. Fire the payload natively via HTTP POST
        conn.request(
            "POST", "/api/v1/chat/completions", json.dumps(payload_data), headers
        )

        # 5. Extract and parse network data
        response = conn.getresponse()
        raw_data = response.read().decode("utf-8")

        if response.status == 200:
            parsed_json = json.loads(raw_data)
            # Extract both the text response AND the actual model used by OpenRouter
            content = parsed_json["choices"][0]["message"]["content"]
            actual_model_used = parsed_json.get("model", model)

            print(f"\nOpenRouter Model Used: {actual_model_used}\n")  # Debugging output

            # Return them together as a tuple
            return content
        else:
            raise RuntimeError(f"HTTP Error {response.status}: {raw_data}")
    except Exception as e:
        raise RuntimeError(f"Network Connection Failed: {str(e)}")
    finally:
        conn.close()

print(agent_chat([], text="Hello! How are you?"))