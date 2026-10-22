import sys
import os
from pathlib import Path

# Add parent directory to path to allow imports from sibling packages
sys.path.insert(0, str(Path(__file__).parent.parent))

from yolo.src.workflows import TomatoHealthDetectionWorkflow
from dotenv import load_dotenv

# Load environment variables from configs/.env file
env_file = Path(__file__).parent.parent / "configs" / ".env"
print(f"Loading environment variables from: {env_file}")
if env_file.exists():
    load_dotenv(env_file)

def detection_integration():
    """Integration: Run detection on example tomato image.
    
    This test requires ROBOFLOW_API_KEY environment variable to be set.
    It will make a API call to Roboflow to detect diseases in the 
    Chlorosis tomato image.
    """
    api_key = os.getenv("ROBOFLOW_API_KEY")
    
    assert api_key is not None, "ROBOFLOW_API_KEY must be set"
    
    workflow = TomatoHealthDetectionWorkflow(
        api_key=api_key
    )
    
    # Run the detection
    result = workflow.run("data/data-input/PXL_20260922_124425712.jpg")

    with open("data/yolo-output/Tomato_Plant_Disease_Result.txt", "w") as f:
        f.write(str(result))
    
    # Verify result structure - Roboflow Serverless Cloud API returns predictions
    assert isinstance(result, dict)
    assert "predictions" in result
    
    # Get predictions array
    predictions = result.get("predictions", [])

    # write the predictions to a file for inspection
    with open("data/yolo-output/Tomato_Plant_Disease_Predictions.txt", "w") as f:
        f.write(str(predictions))
    
    # Get detection info
    if predictions:
        detection = predictions[0]
        detected_class = detection.get("class", "Unknown")
        confidence = detection.get("confidence", 0)
        print(f"\n✓ Detection successful!")
        print(f"✓ Detected class: {detected_class}")
        print(f"✓ Confidence: {confidence:.2%}")
        print(f"✓ Total detections: {len(predictions)}")
    else:
        # No predictions found - plant appears healthy
        print(f"\n✓ Inference completed successfully")
        print(f"✓ No diseases detected in image (plant appears healthy)")

detection_integration()