import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
from src.data.incident_processor import IncidentProcessor
from src.models.incident_classifier import IncidentClassifier
from src.config.config import Config

def generate_and_save_sample_data():
    """Generate sample incident data for initial training."""
    from datetime import datetime, timedelta
    import random
    import uuid

    severities = ["critical", "high", "medium", "low"]
    types = ["outage", "performance", "security", "infrastructure"]
    systems = ["database-cluster", "api-gateway", "auth-service", "storage-service"]
    teams = ["Platform", "Security", "Database", "Frontend", "Backend"]

    incidents = []
    for _ in range(100):  # Generate 100 sample incidents
        detected_at = datetime.now() - timedelta(days=random.randint(1, 30))
        resolved_at = detected_at + timedelta(hours=random.randint(1, 48))
        
        incident = {
            "id": str(uuid.uuid4()),
            "title": f"{random.choice(types).title()} Issue in {random.choice(systems)}",
            "type": random.choice(types),
            "severity": random.choice(severities),
            "description": "Sample incident description for training purposes",
            "affected_systems": random.sample(systems, random.randint(1, 3)),
            "detection_method": random.choice(["monitoring_alert", "user_report"]),
            "affected_users": random.randint(100, 10000),
            "team_involved": random.sample(teams, random.randint(1, 3)),
            "detected_at": detected_at.isoformat(),
            "resolved_at": resolved_at.isoformat(),
            "timeline": [
                {
                    "timestamp": detected_at.isoformat(),
                    "status": "detected",
                    "description": "Issue detected",
                    "action_taken": "Investigation started",
                    "actor": "System"
                }
            ]
        }
        incidents.append(incident)
    
    # Create data directory if it doesn't exist
    data_dir = Path(Config.RAW_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sample data
    with open(data_dir / "sample_incidents.json", "w") as f:
        json.dump(incidents, f, indent=2)
    
    return incidents

def train_and_save_model():
    """Train the incident classifier and save the model."""
    # Generate sample data if it doesn't exist
    try:
        with open(Path(Config.RAW_DATA_DIR) / "sample_incidents.json", "r") as f:
            incidents = json.load(f)
    except FileNotFoundError:
        print("Generating sample data...")
        incidents = generate_and_save_sample_data()
    
    # Create model directory if it doesn't exist
    model_dir = Path(Config.MODELS_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize and train classifier
    print("Training model...")
    classifier = IncidentClassifier(model_dir)
    metrics = classifier.train(incidents)
    
    print("\nTraining metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nModel saved to {model_dir}")

if __name__ == "__main__":
    train_and_save_model()