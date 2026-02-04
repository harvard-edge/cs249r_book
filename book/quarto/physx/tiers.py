"""
tiers.py
Deployment and service tiers shared across volumes.
"""

DEPLOYMENT_TIERS = {
    "cloud": {"label": "Cloud", "latency_ms": "100-500", "power": "MW"},
    "edge": {"label": "Edge", "latency_ms": "10-100", "power": "100s W"},
    "mobile": {"label": "Mobile", "latency_ms": "5-50", "power": "2-5 W"},
    "tinyml": {"label": "TinyML", "latency_ms": "1-10", "power": "mW"},
}

SERVICE_TIERS = {
    "enterprise": {"label": "Enterprise", "slo": "99.9%"},
    "professional": {"label": "Professional", "slo": "99.5%"},
    "free": {"label": "Free", "slo": "best-effort"},
}
