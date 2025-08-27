#!/usr/bin/env python3
"""
FastAPI Backend Server for AI Video Analysis Platform
Step 2: Basic backend that integrates with your existing modules
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import asyncio
import uuid
import logging
import shutil
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your existing modules
try:
    from integrated_configurable_pipeline import IntegratedConfigurableAnalysisPipeline
    from query_video_rag import RAGQueryInterface
    from configurable_agent_system import load_agent_template
    MODULES_AVAILABLE = True
    print("✅ All analysis modules imported successfully")
except ImportError as e:
    print(f"⚠️ Module import issue: {e}")
    MODULES_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Video Analysis Platform API",
    description="Backend API for multi-agent video analysis with RAG capabilities",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables
analysis_tasks: Dict[str, Dict] = {}
upload_directory = Path("uploads")
results_directory = Path("results")

# Create directories
upload_directory.mkdir(exist_ok=True)
results_directory.mkdir(exist_ok=True)

# Pydantic models for API
class AnalysisConfig(BaseModel):
    analysis_depth: str = "comprehensive"
    max_frames: int = 10
    fps_extract: float = 0.2
    discussion_rounds: int = 3
    selected_agents: List[str] = ["alex", "maya", "jordan"]
    agent_template: Optional[str] = None
    content_type: Optional[str] = None
    enable_rag: bool = True

class SearchQuery(BaseModel):
    query: str
    num_results: int = 5
    focus_on: str = "all"
    video_filter: Optional[str] = None

class AnalysisStatus(BaseModel):
    task_id: str
    status: str
    progress: float
    current_step: str
    results: Optional[Dict] = None
    error: Optional[str] = None

# Initialize analysis pipeline (with error handling)
analysis_pipeline = None
rag_interface = None

if MODULES_AVAILABLE:
    try:
        analysis_pipeline = IntegratedConfigurableAnalysisPipeline(
            analysis_depth="comprehensive",
            enable_rag=True
        )
        rag_interface = RAGQueryInterface()
        logger.info("✅ Analysis pipeline initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize analysis pipeline: {e}")
        logger.info("🔧 This might be due to missing API keys - the system will still work for testing")

# API Routes

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend application"""
    try:
        frontend_file = Path("static/index.html")
        if frontend_file.exists():
            with open(frontend_file, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        else:
            return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>AI Video Analysis Platform</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    .status {{ padding: 20px; border-radius: 8px; margin: 20px 0; }}
                    .success {{ background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
                    .warning {{ background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }}
                    .info {{ background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }}
                </style>
            </head>
            <body>
                <h1>🎬 AI Video Analysis Platform</h1>
                <div class="status info">
                    <h3>Backend API is Running!</h3>
                    <p>The FastAPI backend is working correctly.</p>
                </div>
                
                <div class="status {'success' if MODULES_AVAILABLE else 'warning'}">
                    <h3>Analysis Modules: {'✅ Available' if MODULES_AVAILABLE else '⚠️ Limited'}</h3>
                    <p>{'All your existing Python modules are loaded and ready.' if MODULES_AVAILABLE else 'Some modules may need API keys to be fully functional.'}</p>
                </div>
                
                <div class="status info">
                    <h3>📖 Available Endpoints:</h3>
                    <ul>
                        <li><strong>API Documentation:</strong> <a href="/docs">/docs</a></li>
                        <li><strong>Health Check:</strong> <a href="/api/health">/api/health</a></li>
                        <li><strong>List Agents:</strong> <a href="/api/agents">/api/agents</a></li>
                        <li><strong>Active Tasks:</strong> <a href="/api/tasks">/api/tasks</a></li>
                    </ul>
                </div>
                
                <div class="status info">
                    <h3>🚀 Next Steps:</h3>
                    <ol>
                        <li>The frontend HTML file will be added in Step 3</li>
                        <li>Test the API using <a href="/docs">/docs</a></li>
                        <li>Make sure your API keys are set in the .env file</li>
                    </ol>
                </div>
            </body>
            </html>
            """)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error loading frontend: {e}</h1>")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "modules_available": MODULES_AVAILABLE,
        "analysis_pipeline": analysis_pipeline is not None,
        "rag_interface": rag_interface is not None,
        "api_keys_set": {
            "fireworks": bool(os.getenv("FIREWORKS_API_KEY")) and len(os.getenv("FIREWORKS_API_KEY", "")) > 10,
            "pinecone": bool(os.getenv("PINECONE_API_KEY")) and len(os.getenv("PINECONE_API_KEY", "")) > 10
        }
    }

@app.post("/api/upload")
async def upload_video(
    video: UploadFile = File(...),
    subtitle: Optional[UploadFile] = File(None)
):
    """Upload video and optional subtitle file"""
    try:
        # Generate unique ID for this upload
        upload_id = str(uuid.uuid4())
        upload_path = upload_directory / upload_id
        upload_path.mkdir(exist_ok=True)

        # Save video file
        video_path = upload_path / f"video_{video.filename}"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Save subtitle file if provided
        subtitle_path = None
        if subtitle:
            subtitle_path = upload_path / f"subtitle_{subtitle.filename}"
            with open(subtitle_path, "wb") as buffer:
                shutil.copyfileobj(subtitle.file, buffer)

        logger.info(f"✅ Files uploaded for {upload_id}: {video.filename}")

        return {
            "upload_id": upload_id,
            "video_filename": video.filename,
            "video_size": video.size if hasattr(video, 'size') else 0,
            "subtitle_filename": subtitle.filename if subtitle else None,
            "status": "uploaded"
        }

    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/{upload_id}")
async def start_analysis(
    upload_id: str,
    config: AnalysisConfig,
    background_tasks: BackgroundTasks
):
    """Start video analysis for uploaded file"""
    try:
        if not MODULES_AVAILABLE or not analysis_pipeline:
            # For testing without full setup
            return {
                "task_id": f"demo_{uuid.uuid4()}",
                "status": "demo_mode",
                "message": "Demo mode - analysis pipeline not fully initialized. Check API keys and module setup."
            }

        upload_path = upload_directory / upload_id
        if not upload_path.exists():
            raise HTTPException(status_code=404, detail="Upload not found")

        # Find video file
        video_files = list(upload_path.glob("video_*"))
        if not video_files:
            raise HTTPException(status_code=404, detail="Video file not found")

        video_path = video_files[0]
        
        # Find subtitle file
        subtitle_files = list(upload_path.glob("subtitle_*"))
        subtitle_path = subtitle_files[0] if subtitle_files else None

        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Initialize task status
        analysis_tasks[task_id] = {
            "status": "queued",
            "progress": 0,
            "current_step": "Initializing...",
            "config": config.dict(),
            "upload_id": upload_id,
            "video_path": str(video_path),
            "subtitle_path": str(subtitle_path) if subtitle_path else None,
            "results": None,
            "error": None,
            "created_at": datetime.now().isoformat()
        }

        # Start analysis in background
        background_tasks.add_task(
            run_analysis_task,
            task_id,
            str(video_path),
            str(subtitle_path) if subtitle_path else None,
            config
        )

        logger.info(f"✅ Analysis started for task {task_id}")

        return {
            "task_id": task_id,
            "status": "started",
            "message": "Analysis started in background"
        }

    except Exception as e:
        logger.error(f"❌ Analysis start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status/{task_id}")
async def get_analysis_status(task_id: str):
    """Get analysis status and results"""
    if task_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = analysis_tasks[task_id]
    
    return AnalysisStatus(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        current_step=task["current_step"],
        results=task["results"],
        error=task["error"]
    )

@app.post("/api/search")
async def search_video_content(query: SearchQuery):
    """Search analyzed video content using RAG"""
    try:
        if not rag_interface:
            # Return demo results when RAG is not available
            demo_results = [
                {
                    "content": f"Demo search result for query: '{query.query}'",
                    "confidence": 0.85,
                    "document_type": "demo",
                    "video": "demo_video.mp4",
                    "timestamp": 15.0,
                    "frame_number": 3,
                    "agent": {"name": "Demo Agent", "role": "Demo Analyst"},
                    "context": "Demo mode - RAG system not fully initialized"
                }
            ]
            return {
                "query": query.query,
                "results": demo_results,
                "count": len(demo_results),
                "mode": "demo"
            }

        results = rag_interface.query_video_rag(
            query=query.query,
            num_results=query.num_results,
            focus_on=query.focus_on,
            video_filter=query.video_filter
        )

        logger.info(f"🔍 Search completed: '{query.query}' -> {len(results)} results")

        return {
            "query": query.query,
            "results": results,
            "count": len(results)
        }

    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents")
async def list_available_agents():
    """List available AI agents and templates"""
    try:
        # Default agents
        default_agents = [
            {
                "id": "alex",
                "name": "Alex",
                "role": "Technical Analyst",
                "emoji": "🎬",
                "description": "Focuses on technical aspects, cinematography, and production quality"
            },
            {
                "id": "maya",
                "name": "Maya", 
                "role": "Creative Interpreter",
                "emoji": "🎨",
                "description": "Explores artistic meanings, themes, and emotional impact"
            },
            {
                "id": "jordan",
                "name": "Jordan",
                "role": "Audience Advocate", 
                "emoji": "👥",
                "description": "Considers accessibility, engagement, and viewer experience"
            },
            {
                "id": "affan",
                "name": "Affan",
                "role": "Financial Marketing Analyst",
                "emoji": "🤖", 
                "description": "Analyzes marketability and financial appeal"
            }
        ]

        # Available templates
        templates = [
            {
                "id": "film_analysis",
                "name": "Film Analysis Specialists",
                "description": "Cinematographer, Film Critic, Sound Designer",
                "use_case": "Movie reviews, film analysis, cinematic content"
            },
            {
                "id": "educational",
                "name": "Educational Content Specialists",
                "description": "Learning Specialist, Subject Expert",
                "use_case": "Educational videos, tutorials, learning content"
            },
            {
                "id": "marketing",
                "name": "Marketing & Brand Specialists", 
                "description": "Brand Strategist, Conversion Specialist",
                "use_case": "Marketing videos, commercials, promotional content"
            }
        ]

        return {
            "default_agents": default_agents,
            "templates": templates
        }

    except Exception as e:
        logger.error(f"❌ Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tasks")
async def list_active_tasks():
    """List all active analysis tasks"""
    return {
        "tasks": [
            {
                "task_id": task_id,
                "status": task["status"],
                "progress": task["progress"],
                "created_at": task["created_at"],
                "video_path": task.get("video_path", "").split("\\")[-1]  # Just filename
            }
            for task_id, task in analysis_tasks.items()
        ],
        "count": len(analysis_tasks)
    }

# Background task for running analysis
async def run_analysis_task(
    task_id: str,
    video_path: str,
    subtitle_path: Optional[str],
    config: AnalysisConfig
):
    """Run video analysis in background"""
    try:
        # Update task status
        analysis_tasks[task_id]["status"] = "running"
        analysis_tasks[task_id]["progress"] = 10
        analysis_tasks[task_id]["current_step"] = "Initializing analysis pipeline..."

        if not analysis_pipeline:
            raise Exception("Analysis pipeline not available - check API keys and module setup")

        # Simulate progress updates
        for progress in [25, 50, 75, 90]:
            await asyncio.sleep(2)  # Simulate work
            analysis_tasks[task_id]["progress"] = progress
            analysis_tasks[task_id]["current_step"] = f"Processing... {progress}%"

        # Debug: Log the configuration being passed
        logger.info(f"🔧 Analysis config: {config.dict()}")
        logger.info(f"🤖 Selected agents: {config.selected_agents}")
        
        # Run the actual analysis
        results = await analysis_pipeline.analyze_video_with_configurable_agents(
            video_path=video_path,
            subtitle_path=subtitle_path,
            max_frames=config.max_frames,
            fps_extract=config.fps_extract,
            discussion_rounds=config.discussion_rounds,
            selected_agents=config.selected_agents if config.selected_agents else ["alex", "maya", "jordan"],
            content_type=config.content_type,
            agent_template=config.agent_template,
            output_dir=str(results_directory / task_id)
        )

        # Update task with completion
        analysis_tasks[task_id].update({
            "status": "completed",
            "progress": 100,
            "current_step": "Analysis completed successfully!",
            "results": {
                "summary": results,
                "task_id": task_id,
                "video_path": video_path
            }
        })

        logger.info(f"✅ Analysis completed for task {task_id}")

    except Exception as e:
        # Update task with error
        analysis_tasks[task_id].update({
            "status": "failed",
            "progress": 0,
            "current_step": f"Analysis failed: {str(e)}",
            "error": str(e)
        })

        logger.error(f"❌ Analysis failed for task {task_id}: {e}")

# Run the server
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting AI Video Analysis Platform Server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🎬 Frontend App: http://localhost:8000/")
    print("🔧 Health Check: http://localhost:8000/api/health")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )