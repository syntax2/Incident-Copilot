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