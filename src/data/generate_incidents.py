from datetime import datetime, timedelta
import random
import json
import uuid
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class Severity(Enum):
    CRITICAL = "critical"    # Complete service outage or data breach
    HIGH = "high"           # Significant impact on service or security
    MEDIUM = "medium"       # Partial service degradation
    LOW = "low"            # Minor issues with minimal impact

class IncidentType(Enum):
    OUTAGE = "outage"                  # Service unavailability
    PERFORMANCE = "performance"         # Degraded performance
    SECURITY = "security"              # Security-related incidents
    DATA = "data"                      # Data-related issues
    INFRASTRUCTURE = "infrastructure"   # Infrastructure problems

class IncidentStatus(Enum):
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    POSTMORTEM = "postmortem"

@dataclass
class IncidentTimeline:
    timestamp: str
    status: str
    description: str
    action_taken: str
    actor: str

@dataclass
class Incident:
    id: str                            # Unique identifier
    title: str                         # Short, descriptive title
    type: str                          # Type of incident
    severity: str                      # Incident severity level
    status: str                        # Current status
    detected_at: str                   # Initial detection timestamp
    resolved_at: Optional[str]         # Resolution timestamp
    detection_method: str              # How the incident was detected
    description: str                   # Detailed incident description
    affected_systems: List[str]        # Impacted systems/services
    affected_users: int                # Number of users impacted
    timeline: List[IncidentTimeline]   # Incident progression
    tags: List[str]                    # Categorization tags
    primary_responder: str             # Main incident handler
    team_involved: List[str]           # All teams involved
    root_cause: Optional[str]          # Identified root cause
    resolution: Optional[str]          # How it was resolved
    mitigation_steps: List[str]        # Steps taken to resolve
    lessons_learned: List[str]         # Key takeaways
    follow_up_actions: List[str]       # Future prevention steps

def generate_sample_incidents(num_incidents: int = 50) -> List[Dict]:
    """Generate realistic sample incident data."""
    
    # Sample data components
    systems = ["authentication-service", "payment-gateway", "user-api", "database-cluster", 
              "message-queue", "load-balancer", "cache-service", "storage-service"]
    
    teams = ["Platform", "Security", "Database", "Frontend", "Backend", "DevOps", "SRE"]
    
    engineers = ["Alice Thompson", "Bob Martinez", "Carol Zhang", "David Kumar", 
                "Eva Patel", "Frank Rodriguez", "Grace Kim", "Henry Wilson"]
    
    detection_methods = ["monitoring_alert", "customer_report", "internal_user", 
                        "automated_test", "security_scan", "performance_monitoring"]
    
    common_tags = ["production", "customer-facing", "internal-tooling", "data-sensitive",
                  "compliance", "performance-critical", "high-traffic", "legacy-system"]
    
    incidents = []
    
    for _ in range(num_incidents):
        # Generate basic incident details
        incident_type = random.choice(list(IncidentType))
        severity = random.choice(list(Severity))
        
        # Generate realistic timestamps
        detected_at = datetime.now() - timedelta(days=random.randint(1, 90))
        resolution_time = timedelta(hours=random.randint(1, 48))
        resolved_at = detected_at + resolution_time if random.random() > 0.2 else None
        
        # Create timeline entries
        timeline = []
        current_time = detected_at
        
        for status in IncidentStatus:
            if status == IncidentStatus.POSTMORTEM and not resolved_at:
                continue
                
            timeline.append(IncidentTimeline(
                timestamp=current_time.isoformat(),
                status=status.value,
                description=f"Incident entered {status.value} stage",
                action_taken=f"Standard {status.value} procedures initiated",
                actor=random.choice(engineers)
            ))
            current_time += timedelta(hours=random.randint(1, 4))
        
        # Create incident record
        incident = Incident(
            id=str(uuid.uuid4()),
            title=f"{incident_type.value.title()} Incident: {random.choice(systems)}",
            type=incident_type.value,
            severity=severity.value,
            status=IncidentStatus.RESOLVED.value if resolved_at else IncidentStatus.INVESTIGATING.value,
            detected_at=detected_at.isoformat(),
            resolved_at=resolved_at.isoformat() if resolved_at else None,
            detection_method=random.choice(detection_methods),
            description=f"Major {incident_type.value} incident affecting {random.choice(systems)}",
            affected_systems=random.sample(systems, random.randint(1, 3)),
            affected_users=random.randint(100, 10000),
            timeline=[asdict(t) for t in timeline],
            tags=random.sample(common_tags, random.randint(2, 4)),
            primary_responder=random.choice(engineers),
            team_involved=random.sample(teams, random.randint(1, 3)),
            root_cause="Root cause analysis in progress" if not resolved_at else f"Identified issue in {random.choice(systems)}",
            resolution="Ongoing investigation" if not resolved_at else "Applied system patches and updated configurations",
            mitigation_steps=[
                "Isolated affected systems",
                "Applied emergency patches",
                "Increased monitoring",
                "Scaled up resources"
            ],
            lessons_learned=[
                "Need better monitoring",
                "Update disaster recovery plan",
                "Improve documentation"
            ] if resolved_at else [],
            follow_up_actions=[
                "Update incident response playbook",
                "Schedule preventive maintenance",
                "Review system architecture"
            ] if resolved_at else []
        )
        
        incidents.append(asdict(incident))
    
    return incidents

if __name__ == "__main__":
    # Generate sample incidents
    sample_incidents = generate_sample_incidents()
    
    # Save to file
    output_path = "data/raw/sample_incidents.json"
    with open(output_path, 'w') as f:
        json.dump(sample_incidents, f, indent=2)
    
    print(f"Generated {len(sample_incidents)} sample incidents and saved to {output_path}")