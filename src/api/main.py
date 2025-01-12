# src/api/main.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import json
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.incident_classifier import IncidentClassifier
from src.config.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Incident Copilot API",
    description="AI-powered Incident Response Assistant API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize classifier
classifier = IncidentClassifier(Path(Config.MODELS_DIR))

class IncidentBase(BaseModel):
    """Base Pydantic model for incident data validation."""
    title: str = Field(..., description="Brief description of the incident")
    type: str = Field(..., description="Type of incident")
    severity: Optional[str] = Field(None, description="Incident severity level")
    description: str = Field(..., description="Detailed description of the incident")
    affected_systems: List[str] = Field(..., description="List of affected systems")
    detection_method: str = Field(..., description="How the incident was detected")
    affected_users: int = Field(0, description="Number of affected users")
    team_involved: List[str] = Field(..., description="Teams involved in resolution")

class IncidentResponse(BaseModel):
    """Pydantic model for API responses."""
    id: str
    severity_prediction: Dict
    type_prediction: Optional[Dict]
    recommended_actions: List[str]
    estimated_resolution_time: Optional[float]

def generate_recommendations(incident_data: Dict, predictions: Dict) -> List[str]:
    """Generate response recommendations based on incident data and predictions."""
    recommendations = []
    
    # Severity-based recommendations
    if predictions['confidence'] > 0.8:
        if predictions['prediction'] in ['critical', 'high']:
            recommendations.extend([
                "Immediately establish incident command structure",
                "Alert senior management and stakeholder teams",
                "Begin customer communication preparation",
                "Initialize war room and communication channels"
            ])
        elif predictions['prediction'] == 'medium':
            recommendations.extend([
                "Assign dedicated incident manager",
                "Set up monitoring dashboard",
                "Prepare initial assessment report"
            ])
    
    # System-based recommendations
    if 'database' in ' '.join(incident_data['affected_systems']).lower():
        recommendations.extend([
            "Check database replication status",
            "Review recent database changes",
            "Prepare rollback scripts if needed"
        ])
    
    if 'api' in ' '.join(incident_data['affected_systems']).lower():
        recommendations.extend([
            "Check API logs for error patterns",
            "Verify API dependencies status",
            "Review recent deployments"
        ])
    
    return recommendations

@app.post("/api/v1/incidents/analyze", response_model=IncidentResponse)
async def analyze_incident(
    incident: IncidentBase,
    background_tasks: BackgroundTasks
) -> IncidentResponse:
    """
    Analyze a new incident and provide recommendations.
    
    This endpoint:
    1. Validates the incident data
    2. Makes severity and type predictions
    3. Generates recommended actions
    4. Estimates resolution time
    """
    try:
        # Convert incident to dictionary
        incident_data = incident.dict()
        
        # Add timestamp
        incident_data['detected_at'] = datetime.now().isoformat()
        
        # Make predictions
        severity_prediction = classifier.predict(incident_data)
        
        # Generate recommendations
        recommendations = generate_recommendations(incident_data, severity_prediction)
        
        # Create response
        response = IncidentResponse(
            id=f"INC-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            severity_prediction=severity_prediction,
            type_prediction=None,  # Could add type prediction in future
            recommended_actions=recommendations,
            estimated_resolution_time=None  # Could add time estimation in future
        )
        
        # Add background task to save incident
        background_tasks.add_task(save_incident, incident_data, response)
        
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing incident: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def save_incident(incident_data: Dict, analysis_response: IncidentResponse):
    """Save incident data and analysis results for future training."""
    try:
        # Combine incident data and analysis
        saved_data = {
            **incident_data,
            "analysis": analysis_response.dict()
        }
        
        # Save to file
        output_path = Config.RAW_DATA_DIR / f"incident_{analysis_response.id}.json"
        with open(output_path, 'w') as f:
            json.dump(saved_data, f, indent=2)
            
        logger.info(f"Saved incident data to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving incident: {str(e)}")

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        workers=Config.API_WORKERS,
        reload=True
    )