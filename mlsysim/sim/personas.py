# personas.py
"""
MLSys Personas
==============
Defines the Persona Archetypes (the rows of the lab matrix).
Each persona defines the scale multiplier and the primary engineering constraint
for a specific real-world deployment tier.
"""

from dataclasses import dataclass
from typing import Dict
from ..core.constants import ureg, Q_

@dataclass(frozen=True)
class Persona:
    """
    A Persona defining scaling factors and narrative focus for a simulation.
    
    Attributes:
        name: The display name of the persona (e.g., 'Cloud Titan').
        role: The job title/role (e.g., 'LLM Infrastructure Lead').
        description: A brief narrative overview of the persona's goal.
        scale_factor: The multiplier for device/node count in the simulation.
        primary_constraint: The 'Critical Wall' this persona must manage.
        unit_of_scale: The noun used for scaling (e.g., 'Fleet', 'Billion Devices').
    """
    name: str
    role: str
    description: str
    scale_factor: float
    primary_constraint: str
    unit_of_scale: str

class Personas:
    """The four canonical personas for the MLSys curriculum."""

    # --- CLOUD TITAN ---
    CloudTitan = Persona(
        name="Cloud Titan",
        role="LLM Infrastructure Lead",
        description="Responsible for utility-scale training and serving clusters.",
        scale_factor=1.0, 
        primary_constraint="Total Cost of Ownership (TCO) & Grid Stability",
        unit_of_scale="Cluster"
    )

    # --- EDGE GUARDIAN ---
    EdgeGuardian = Persona(
        name="Edge Guardian",
        role="Autonomous Systems Lead",
        description="Manages safety-critical real-time vehicle fleets.",
        scale_factor=10_000.0, 
        primary_constraint="Latency Determinism & Safety",
        unit_of_scale="Fleet"
    )

    # --- MOBILE NOMAD ---
    MobileNomad = Persona(
        name="Mobile Nomad",
        role="Smartphone App Architect",
        description="Optimizes global inference for consumer-scale applications.",
        scale_factor=100_000_000.0, 
        primary_constraint="Battery Life & UX Responsiveness",
        unit_of_scale="Global User Base"
    )

    # --- TINY PIONEER ---
    TinyPioneer = Persona(
        name="Tiny Pioneer",
        role="Smart Doorbell Product Lead",
        description="Scales always-on sensing to billions of sub-milliwatt devices.",
        scale_factor=10_000_000.0, 
        primary_constraint="Embodied Carbon & SRAM Limits",
        unit_of_scale="Installed Base"
    )

    @classmethod
    def get(cls, key: str) -> Persona:
        """Fetch a persona by its identifier key (cloud, edge, mobile, tiny).
        
        Args:
            key: Persona identifier.
            
        Returns:
            The corresponding Persona object. Defaults to CloudTitan.
        """
        lookup = {
            "cloud": cls.CloudTitan,
            "edge": cls.EdgeGuardian,
            "mobile": cls.MobileNomad,
            "tiny": cls.TinyPioneer
        }
        return lookup.get(key.lower(), cls.CloudTitan)
