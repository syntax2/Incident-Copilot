from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
import json
import joblib
from typing import Tuple, Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IncidentProcessor:
    """
    Processes raw incident data and creates features for ML model training.
    This class handles all data preprocessing steps including text vectorization
    and feature engineering.
    """
    def __init__(self):
        self.text_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.label_encoders = {}
        self.feature_names = None
        
    def _combine_text_fields(self, incident: Dict) -> str:
        """Combines relevant text fields for analysis."""
        text_fields = [
            incident.get('title', ''),
            incident.get('description', ''),
            incident.get('root_cause', ''),
            ' '.join(incident.get('affected_systems', [])),
            ' '.join(incident.get('tags', []))
        ]
        return ' '.join(filter(None, text_fields))
    
    def _extract_numerical_features(self, incident: Dict) -> Dict:
        """Extracts numerical features from incident data."""
        # Calculate incident duration if resolved
        duration = 0
        if incident.get('resolved_at') and incident.get('detected_at'):
            detected = pd.to_datetime(incident['detected_at'])
            resolved = pd.to_datetime(incident['resolved_at'])
            duration = (resolved - detected).total_seconds() / 3600  # Duration in hours
            
        return {
            'affected_users': incident.get('affected_users', 0),
            'duration_hours': duration,
            'num_systems_affected': len(incident.get('affected_systems', [])),
            'num_teams_involved': len(incident.get('team_involved', [])),
            'timeline_steps': len(incident.get('timeline', [])),
        }
    
    def transform_incidents(self, incidents: List[Dict]) -> pd.DataFrame:
        """
        Transforms raw incident data into a feature matrix.
        
        Args:
            incidents: List of incident dictionaries
            
        Returns:
            DataFrame with processed features
        """
        # Extract features from incidents
        processed_data = []
        for incident in incidents:
            features = {
                'text_content': self._combine_text_fields(incident),
                'severity': incident.get('severity'),
                'type': incident.get('type'),
                'detection_method': incident.get('detection_method'),
                **self._extract_numerical_features(incident)
            }
            processed_data.append(features)
            
        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        
        # Encode categorical variables
        categorical_columns = ['severity', 'type', 'detection_method']
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])
            
        return df

class IncidentClassifier:
    """
    Trains and manages ML models for incident classification.
    This class handles model training, evaluation, and prediction.
    """
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.processor = IncidentProcessor()
        self.model = None
        self.feature_names = None
        
    def train(self, incidents: List[Dict], target: str = 'severity') -> Dict:
        """
        Trains the incident classification model.
        
        Args:
            incidents: List of incident dictionaries
            target: Target variable to predict ('severity', 'type')
            
        Returns:
            Dictionary containing training metrics
        """
        # Process incidents
        df = self.processor.transform_incidents(incidents)
        
        # Split features and target
        X = df.drop(target, axis=1)
        y = df[target]
        
        # Split training and test data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create feature processing pipeline
        text_features = ['text_content']
        numeric_features = ['affected_users', 'duration_hours', 'num_systems_affected',
                          'num_teams_involved', 'timeline_steps']
        categorical_features = ['detection_method']
        
        if target != 'type':
            categorical_features.append('type')
        if target != 'severity':
            categorical_features.append('severity')
        
        # Define preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', self.processor.text_vectorizer, text_features),
                ('num', 'passthrough', numeric_features),
                ('cat', 'passthrough', categorical_features)
            ]
        )
        
        # Create and train pipeline
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Train model
        logger.info("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        metrics = classification_report(y_test, y_pred, output_dict=True)
        
        # Save model and processors
        self._save_model()
        
        return metrics
    
    def predict(self, incident: Dict) -> Dict:
        """
        Makes predictions for a new incident.
        
        Args:
            incident: Dictionary containing incident data
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        if not self.model:
            raise ValueError("Model not trained. Please train the model first.")
            
        # Process incident
        df = self.processor.transform_incidents([incident])
        
        # Make prediction
        prediction = self.model.predict(df)[0]
        probabilities = self.model.predict_proba(df)[0]
        
        # Get prediction confidence
        confidence = max(probabilities)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': dict(zip(
                self.processor.label_encoders['severity'].classes_,
                probabilities
            ))
        }
    
    def _save_model(self):
        """Saves the trained model and processors."""
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True)
            
        model_path = self.model_dir / 'incident_classifier.joblib'
        processor_path = self.model_dir / 'incident_processor.joblib'
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.processor, processor_path)
        logger.info(f"Model saved to {model_path}")

def main():
    """Main function to demonstrate the ML pipeline."""
    # Load sample incidents
    with open('data/raw/sample_incidents.json', 'r') as f:
        incidents = json.load(f)
    
    # Initialize classifier
    classifier = IncidentClassifier(Path('models/saved'))
    
    # Train model
    metrics = classifier.train(incidents)
    print("\nTraining Metrics:")
    print(json.dumps(metrics, indent=2))
    
    # Make sample prediction
    sample_incident = incidents[0]
    prediction = classifier.predict(sample_incident)
    print("\nSample Prediction:")
    print(json.dumps(prediction, indent=2))

if __name__ == "__main__":
    main()