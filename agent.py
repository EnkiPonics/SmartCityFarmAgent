import os
import http.client
import json
import base64
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional
import pandas as pd
from PIL import Image
import io

# Load environment variables from configs/.env file
env_file = Path(__file__).parent.parent / "configs" / ".env"
print(f"Loading environment variables from: {env_file}")
if env_file.exists():
    load_dotenv(env_file)

introduction_llm = """For the OSI Hackathon 2026, the user has written a comprehensive EnkiPonics sensor data analysis and LSTM modeling
with data from SmartCityFarm e.V. (SCF) Berlin <3."""

# ============ DATA LOADING FUNCTIONS ============

def load_csv_summary(csv_path: Path, max_rows: int = 5) -> str:
    """Load CSV and return a summary with basic statistics."""
    try:
        df = pd.read_csv(csv_path)
        summary = f"\n📊 File: {csv_path.name}\n"
        summary += f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        summary += f"   Columns: {', '.join(df.columns.tolist())}\n"
        
        # Add basic stats for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary += f"   Statistics:\n"
            for col in numeric_cols[:5]:  # Limit to first 5 numeric cols
                summary += f"      {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}\n"
        
        return summary
    except Exception as e:
        return f"\n❌ Error reading {csv_path.name}: {str(e)}\n"

def load_image_as_base64(image_path: Path) -> dict:
    """Load image and convert to base64 for LLM vision capabilities."""
    try:
        with open(image_path, 'rb') as img_file:
            image_data = base64.standard_b64encode(img_file.read()).decode('utf-8')
        return {
            "type": "image",
            "name": image_path.name,
            "base64": image_data,
            "media_type": "image/png"
        }
    except Exception as e:
        return {"type": "error", "name": image_path.name, "error": str(e)}

def load_json_data(json_path: Path) -> str:
    """Load JSON classification data."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return f"\n📋 File: {json_path.name}\n   Content: {json.dumps(data, indent=2)[:500]}...\n"
    except Exception as e:
        return f"\n❌ Error reading {json_path.name}: {str(e)}\n"

def load_txt_data(txt_path: Path) -> str:
    """Load TXT data (YOLO results, predictions, etc.)."""
    try:
        with open(txt_path, 'r') as f:
            content = f.read()
        # If it looks like JSON, format it nicely
        try:
            parsed = json.loads(content)
            formatted = json.dumps(parsed, indent=2)[:800]
            return f"\n📄 File: {txt_path.name}\n   Content: {formatted}\n"
        except:
            # Regular text - truncate to 500 chars
            return f"\n📄 File: {txt_path.name}\n   Content: {content[:500]}\n"
    except Exception as e:
        return f"\n❌ Error reading {txt_path.name}: {str(e)}\n"

def load_pipeline_data(pipeline_dir: Optional[Path] = None) -> dict:
    """Load all data from pipelines directory."""
    if pipeline_dir is None:
        pipeline_dir = Path(__file__).parent / "pipelines" / "data"
    
    data_context = {
        "csv_summaries": "",
        "images": [],
        "json_data": "",
        "raw_structure": ""
    }
    
    if not pipeline_dir.exists():
        return data_context
    
    # Load CSVs from data_input and lstm_output
    data_context["csv_summaries"] += "\n=== SENSOR DATA & LSTM OUTPUTS ===\n"
    for csv_file in sorted(pipeline_dir.glob("*_input/*.csv")) + sorted(pipeline_dir.glob("lstm_output/*.csv")):
        data_context["csv_summaries"] += load_csv_summary(csv_file)
    
    # Load images (PNG files)
    data_context["raw_structure"] += "\n=== AVAILABLE VISUALIZATIONS ===\n"
    for image_file in sorted(pipeline_dir.glob("*_output/*.png")):
        data_context["raw_structure"] += f"   📈 {image_file.name}\n"
        # Optionally load images as base64 for vision-capable models
        img_data = load_image_as_base64(image_file)
        if img_data["type"] == "image":
            data_context["images"].append(img_data)
    
    # Load classification results (JSON and TXT)
    data_context["json_data"] += "\n=== CLASSIFICATION & DETECTION RESULTS ===\n"
    for json_file in sorted(pipeline_dir.glob("*_output/*.json")):
        data_context["json_data"] += load_json_data(json_file)
    
    # Load TXT outputs (YOLO predictions, etc.)
    for txt_file in sorted(pipeline_dir.glob("*_output/*.txt")):
        data_context["json_data"] += load_txt_data(txt_file)
    
    return data_context

def build_context_message(data_context: dict) -> str:
    """Build a comprehensive context message from loaded data."""
    context = "\n" + "="*60 + "\n"
    context += "📥 ANALYZED PIPELINE DATA & OUTPUTS\n"
    context += "="*60 + "\n"
    context += data_context["csv_summaries"]
    context += data_context["json_data"]
    context += data_context["raw_structure"]
    context += "="*60 + "\n"
    context += "Based on the above data, provide a comprehensive recommendation for system improvements.\n"
    return context

def agent_chat(
    history,
    model: str = "nvidia/nemotron-3-ultra-550b-a55b:free",
    max_tokens=50000,
    text="Your initial text here",
    include_pipeline_data: bool = True,
):
    """Chat with agent, optionally including pipeline data context."""
    # Load pipeline data if requested
    if include_pipeline_data:
        data_context = load_pipeline_data()
        context_message = build_context_message(data_context)
    else:
        context_message = ""

    # Add the assistant's initial message to the beginning of the history
    history.insert(0, {"role": "assistant", "content": introduction_llm})
    
    # Add data context if available
    if context_message:
        history.insert(1, {"role": "user", "content": context_message})
    
    # Add the user's initial message to the history
    history.insert(2 if context_message else 1, {"role": "user", "content": text})

    # --- NATIVE OPENROUTER NETWORK CALL ---

    # 1. Establish network connection
    conn = http.client.HTTPSConnection("openrouter.ai")

    # 2. Structure payload with chat history
    payload_data = {
        "model": model,
        "messages": history,
        "max_tokens": max_tokens,
    }

    # 3. Secure headers using your OpenRouter token
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_KEY')}",
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
            content = parsed_json["choices"][0]["message"]["content"]
            actual_model_used = parsed_json.get("model", model)

            print(f"\nOpenRouter Model Used: {actual_model_used}\n")

            return content
        else:
            raise RuntimeError(f"HTTP Error {response.status}: {raw_data}")
    except Exception as e:
        raise RuntimeError(f"Network Connection Failed: {str(e)}")
    finally:
        conn.close()

# Example usage
if __name__ == "__main__":
    print("🚀 Loading pipeline data...")
    result = agent_chat(
        [], 
        text="Analyze the aquaponics system data and provide recommendations for optimization."
    )
    print("\n📋 Agent Recommendation:\n")
    print(result)
