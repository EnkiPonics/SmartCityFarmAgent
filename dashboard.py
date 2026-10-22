"""
EnkiPonics Dashboard - FastAPI backend
Serves web interface for aquaponics analysis
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import json
from datetime import datetime
from agent import agent_chat, load_pipeline_data

app = FastAPI(title="EnkiPonics Dashboard")

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
    print("Starting EnkiPonics Dashboard...")
    print("Visit: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
