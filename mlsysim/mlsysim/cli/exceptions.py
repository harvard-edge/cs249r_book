from enum import IntEnum
import sys
from contextlib import contextmanager

class ExitCode(IntEnum):
    """Semantic exit codes for CLI automation."""
    SUCCESS = 0
    BAD_INPUT = 1      # Syntax Error, Typo, Validation Failure
    PHYSICS_FAIL = 2   # Out of Memory, Pipeline Starved (Hardware Limitation)
    SLA_FAIL = 3       # P99 too high, TCO over budget (Business Limitation)

def exit_with_code(code: ExitCode, message: str = ""):
    if message:
        # Errors ALWAYS go to stderr to protect stdout data in CI pipelines
        print(message, file=sys.stderr)
    sys.exit(code.value)

@contextmanager
def error_shield(output_format: str = "text"):
    """
    Context manager to catch exceptions, format them properly via renderers,
    and exit with the correct semantic code. Promotes DRY principles.
    """
    from mlsysim.core.exceptions import OOMError
    from pydantic import ValidationError
    from mlsysim.cli.renderers import print_error
    
    try:
        yield
    except OOMError as e:
        print_error("Physics Violation (Out of Memory)", str(e), output_format=output_format)
        exit_with_code(ExitCode.PHYSICS_FAIL)
    except ValidationError as e:
        # Format Pydantic errors cleanly
        msg = "\n".join([f"- {err['msg']} (Input: {err.get('input')})" for err in e.errors()])
        print_error("Schema Validation Error", msg, output_format=output_format)
        exit_with_code(ExitCode.BAD_INPUT)
    except ValueError as e:
        print_error("Bad Input", str(e), output_format=output_format)
        exit_with_code(ExitCode.BAD_INPUT)
    except Exception as e:
        print_error("Evaluation Error", str(e), output_format=output_format)
        exit_with_code(ExitCode.BAD_INPUT)
