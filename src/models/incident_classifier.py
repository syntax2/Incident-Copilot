from src.models.incident_classifier import IncidentClassifier
from pathlib import Path

# Initialize classifier
classifier = IncidentClassifier(Path('models/saved'))

# Train model
metrics = classifier.train(incidents, target='severity')

# Make predictions
prediction = classifier.predict(new_incident)