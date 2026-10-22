"""
SmartCityFarm Agent Dashboard - FastAPI backend
Serves web interface for aquaponics analysis
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import json
import base64
from datetime import datetime
from agent import agent_chat, load_pipeline_data

app = FastAPI(title="SmartCityFarm Agent Dashboard")

# Store analysis results
analysis_cache = {}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "online", "timestamp": datetime.now().isoformat()}

@app.get("/api/data-summary")
async def get_data_summary():
    """Get summary of available data"""
    data = load_pipeline_data()
    return {
        "csv_files": len(data['csv_summaries']) // 100,  # Rough count
        "images": len(data['images']),
        "total_context_chars": len(data['csv_summaries']) + len(data['json_data']) + len(data['raw_structure']),
        "data_loaded": True
    }

@app.get("/api/plots")
async def get_plots():
    """Get all available LSTM and analysis plots"""
    plots = []
    pipeline_dir = Path(__file__).parent / "pipelines" / "data"
    
    if not pipeline_dir.exists():
        return {"plots": []}
    
    # Get all PNG files from output directories
    for png_file in sorted(pipeline_dir.glob("*_output/*.png")):
        try:
            with open(png_file, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Parse filename to categorize
            name = png_file.name
            category = "LSTM" if "future_plot" in name or "parity_plot" in name else "Classification"
            
            plots.append({
                "name": name,
                "category": category,
                "base64": f"data:image/png;base64,{img_data}",
                "filename": name.replace(".png", "")
            })
        except Exception as e:
            print(f"Error loading {png_file}: {e}")
    
    return {
        "count": len(plots),
        "plots": plots
    }

@app.post("/api/analyze")
async def run_analysis(request: dict):
    """
    Run analysis with given parameters
    request: {
        "question": "string",
        "include_data": bool (default True),
        "max_tokens": int (default 4000)
    }
    """
    try:
        question = request.get("question", "Analyze the aquaponics system and provide recommendations.")
        include_data = request.get("include_data", True)
        max_tokens = request.get("max_tokens", 4000)
        
        # Run agent
        result = agent_chat(
            [],
            text=question,
            include_pipeline_data=include_data,
            max_tokens=max_tokens
        )
        
        # Cache result
        analysis_id = datetime.now().isoformat()
        analysis_cache[analysis_id] = {
            "question": question,
            "result": result,
            "timestamp": analysis_id
        }
        
        return {
            "success": True,
            "analysis_id": analysis_id,
            "result": result,
            "timestamp": analysis_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Retrieve cached analysis"""
    if analysis_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis_cache[analysis_id]

@app.get("/api/analyses")
async def list_analyses():
    """List all cached analyses"""
    return {
        "count": len(analysis_cache),
        "analyses": list(analysis_cache.keys())
    }

@app.get("/")
async def serve_index():
    """Serve main dashboard HTML"""
    dashboard_file = Path(__file__).parent / "dashboard.html"
    if dashboard_file.exists():
        return FileResponse(dashboard_file)
    return {"message": "Dashboard HTML not found. Create dashboard.html"}

if __name__ == "__main__":
    import uvicorn
    print("Starting SmartCityFarm Agent Dashboard...")
    print("Visit: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
