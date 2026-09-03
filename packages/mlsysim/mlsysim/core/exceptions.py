# Exceptions for the MLSys Simulator

class MLSysError(Exception):
    """Base exception for all mlsysim simulation errors."""
    pass

class OOMError(MLSysError):
    """Raised when a workload's memory footprint exceeds the hardware capacity."""
    def __init__(self, message, required_bytes=None, available_bytes=None):
        super().__init__(message)
        self.required_bytes = required_bytes
        self.available_bytes = available_bytes

class ThermalThrottleWarning(UserWarning):
    """Warning for when continuous utilization might cause thermal downclocking."""
    pass

class SLAViolation(MLSysError):
    """Raised when a simulated system fails to meet a specified latency or throughput SLA."""
    pass
