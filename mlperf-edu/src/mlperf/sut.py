import abc
from typing import Any, List, Dict
import asyncio
from .loadgen import QuerySample

class SUT_Interface(abc.ABC):
    """
    MLPerf EDU: System Under Test (SUT) Protocol.
    
    Students must inherit from this class to submit optimizations.
    This strictly decouples the grading framework from the student implementation.
    """
    
    @abc.abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model, load weights, and prepare the execution device.
        """
        pass

    @abc.abstractmethod
    async def process_queries(self, samples: List[QuerySample]) -> Any:
        """
        The core asynchronous execution loop called by the LoadGen proxy.
        
        Args:
            samples: A list of QuerySample instances from LoadGen.
        
        Returns:
            Any pedagogical telemetry dictionary (e.g. {"ttft": 0.5}) or list of dicts.
        """
        pass
