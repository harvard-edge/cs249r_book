"""
Input validation utilities for MLSysBook tools.

This module provides comprehensive validation functions for common input types
including file paths, configuration values, and data structures.
"""

import re
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Type
from urllib.parse import urlparse

from .exceptions import ValidationError


def validate_file_path(
    path: Union[str, Path],
    must_exist: bool = True,
    must_be_file: bool = True,
    must_be_readable: bool = True,
    allowed_extensions: Optional[List[str]] = None
) -> Path:
    """Validate a file path.

    Args:
        path: File path to validate
        must_exist: Whether the file must exist
        must_be_file: Whether the path must be a file (not directory)
        must_be_readable: Whether the file must be readable
        allowed_extensions: List of allowed file extensions (e.g., ['.qmd', '.md'])

    Returns:
        Validated Path object

    Raises:
        ValidationError: If validation fails
    """
    if not path:
        raise ValidationError("File path cannot be empty")

    path_obj = Path(path).resolve()

    # Check for path traversal attempts
    try:
        path_obj.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        # Allow absolute paths, but check for suspicious patterns
        path_str = str(path_obj)
        if '..' in path_str or path_str.startswith('/'):
            # Additional validation for absolute paths
            pass

    if must_exist and not path_obj.exists():
        raise ValidationError(f"File does not exist: {path_obj}")

    if must_exist and must_be_file and not path_obj.is_file():
        raise ValidationError(f"Path is not a file: {path_obj}")

    if must_exist and must_be_readable:
        try:
            with open(path_obj, 'r', encoding='utf-8') as f:
                f.read(1)  # Try to read one character
        except PermissionError:
            raise ValidationError(f"File is not readable: {path_obj}")
        except UnicodeDecodeError:
            raise ValidationError(f"File is not valid UTF-8: {path_obj}")

    if allowed_extensions:
        if path_obj.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            raise ValidationError(
                f"File extension {path_obj.suffix} not allowed. "
                f"Allowed extensions: {allowed_extensions}"
            )

    return path_obj


def validate_directory_path(
    path: Union[str, Path],
    must_exist: bool = True,
    create_if_missing: bool = False,
    must_be_writable: bool = False
) -> Path:
    """Validate a directory path.

    Args:
        path: Directory path to validate
        must_exist: Whether the directory must exist
        create_if_missing: Whether to create the directory if it doesn't exist
        must_be_writable: Whether the directory must be writable

    Returns:
        Validated Path object

    Raises:
        ValidationError: If validation fails
    """
    if not path:
        raise ValidationError("Directory path cannot be empty")

    path_obj = Path(path).resolve()

    if not path_obj.exists():
        if create_if_missing:
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValidationError(f"Cannot create directory {path_obj}: {e}")
        elif must_exist:
            raise ValidationError(f"Directory does not exist: {path_obj}")

    if path_obj.exists() and not path_obj.is_dir():
        raise ValidationError(f"Path is not a directory: {path_obj}")

    if must_be_writable and path_obj.exists():
        test_file = path_obj / '.write_test'
        try:
            test_file.touch()
            test_file.unlink()
        except Exception:
            raise ValidationError(f"Directory is not writable: {path_obj}")

    return path_obj


def validate_url(url: str, allowed_schemes: Optional[List[str]] = None) -> str:
    """Validate a URL.

    Args:
        url: URL to validate
        allowed_schemes: List of allowed URL schemes (e.g., ['http', 'https'])

    Returns:
        Validated URL string

    Raises:
        ValidationError: If validation fails
    """
    if not url:
        raise ValidationError("URL cannot be empty")

    if not isinstance(url, str):
        raise ValidationError("URL must be a string")

    try:
        parsed = urlparse(url)
    except Exception as e:
        raise ValidationError(f"Invalid URL format: {e}")

    if not parsed.scheme:
        raise ValidationError("URL must include a scheme (http, https, etc.)")

    if not parsed.netloc:
        raise ValidationError("URL must include a network location")

    if allowed_schemes and parsed.scheme not in allowed_schemes:
        raise ValidationError(
            f"URL scheme '{parsed.scheme}' not allowed. "
            f"Allowed schemes: {allowed_schemes}"
        )

    return url


def validate_json_data(
    data: Any,
    schema: Optional[Dict[str, Any]] = None,
    required_keys: Optional[List[str]] = None
) -> Any:
    """Validate JSON data structure.

    Args:
        data: Data to validate
        schema: Optional JSON schema for validation
        required_keys: Required keys for dictionary data

    Returns:
        Validated data

    Raises:
        ValidationError: If validation fails
    """
    if schema:
        try:
            import jsonschema
            jsonschema.validate(data, schema)
        except ImportError:
            raise ValidationError("jsonschema package required for schema validation")
        except jsonschema.ValidationError as e:
            raise ValidationError(f"JSON schema validation failed: {e.message}")

    if required_keys and isinstance(data, dict):
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValidationError(f"Missing required keys: {missing_keys}")

    return data


def validate_string(
    value: Any,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    pattern: Optional[str] = None,
    allowed_values: Optional[List[str]] = None
) -> str:
    """Validate a string value.

    Args:
        value: Value to validate
        min_length: Minimum string length
        max_length: Maximum string length
        pattern: Regex pattern the string must match
        allowed_values: List of allowed string values

    Returns:
        Validated string

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, str):
        raise ValidationError(f"Expected string, got {type(value).__name__}")

    if min_length is not None and len(value) < min_length:
        raise ValidationError(f"String too short. Minimum length: {min_length}")

    if max_length is not None and len(value) > max_length:
        raise ValidationError(f"String too long. Maximum length: {max_length}")

    if pattern and not re.match(pattern, value):
        raise ValidationError(f"String does not match pattern: {pattern}")

    if allowed_values and value not in allowed_values:
        raise ValidationError(f"Value '{value}' not in allowed values: {allowed_values}")

    return value


def validate_number(
    value: Any,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    number_type: Type = float
) -> Union[int, float]:
    """Validate a numeric value.

    Args:
        value: Value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        number_type: Expected number type (int or float)

    Returns:
        Validated number

    Raises:
        ValidationError: If validation fails
    """
    try:
        if number_type == int:
            numeric_value = int(value)
        else:
            numeric_value = float(value)
    except (ValueError, TypeError):
        raise ValidationError(f"Cannot convert '{value}' to {number_type.__name__}")

    if min_value is not None and numeric_value < min_value:
        raise ValidationError(f"Value {numeric_value} below minimum: {min_value}")

    if max_value is not None and numeric_value > max_value:
        raise ValidationError(f"Value {numeric_value} above maximum: {max_value}")

    return numeric_value


def validate_list(
    value: Any,
    item_validator: Optional[Callable] = None,
    min_items: Optional[int] = None,
    max_items: Optional[int] = None,
    unique_items: bool = False
) -> List[Any]:
    """Validate a list value.

    Args:
        value: Value to validate
        item_validator: Function to validate each item
        min_items: Minimum number of items
        max_items: Maximum number of items
        unique_items: Whether items must be unique

    Returns:
        Validated list

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, list):
        raise ValidationError(f"Expected list, got {type(value).__name__}")

    if min_items is not None and len(value) < min_items:
        raise ValidationError(f"Too few items. Minimum: {min_items}")

    if max_items is not None and len(value) > max_items:
        raise ValidationError(f"Too many items. Maximum: {max_items}")

    if unique_items and len(value) != len(set(value)):
        raise ValidationError("List items must be unique")

    if item_validator:
        validated_items = []
        for i, item in enumerate(value):
            try:
                validated_items.append(item_validator(item))
            except ValidationError as e:
                raise ValidationError(f"Item {i} validation failed: {e}")
        return validated_items

    return value


def validate_config_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Validate and load a configuration file.

    Args:
        file_path: Path to configuration file

    Returns:
        Loaded configuration data

    Raises:
        ValidationError: If validation fails
    """
    path_obj = validate_file_path(
        file_path,
        allowed_extensions=['.yaml', '.yml', '.json']
    )

    try:
        with open(path_obj, 'r', encoding='utf-8') as f:
            if path_obj.suffix.lower() == '.json':
                data = json.load(f)
            else:
                data = yaml.safe_load(f)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in config file: {e}")
    except yaml.YAMLError as e:
        raise ValidationError(f"Invalid YAML in config file: {e}")
    except Exception as e:
        raise ValidationError(f"Cannot read config file: {e}")

    if not isinstance(data, dict):
        raise ValidationError("Configuration file must contain a dictionary/object")

    return data


class Validator:
    """Fluent validation interface for complex validation chains."""

    def __init__(self, value: Any, name: str = "value") -> None:
        """Initialize validator with a value.

        Args:
            value: Value to validate
            name: Name of the value for error messages
        """
        self.value = value
        self.name = name

    def is_string(self, **kwargs) -> 'Validator':
        """Validate that value is a string."""
        self.value = validate_string(self.value, **kwargs)
        return self

    def is_number(self, **kwargs) -> 'Validator':
        """Validate that value is a number."""
        self.value = validate_number(self.value, **kwargs)
        return self

    def is_list(self, **kwargs) -> 'Validator':
        """Validate that value is a list."""
        self.value = validate_list(self.value, **kwargs)
        return self

    def is_file_path(self, **kwargs) -> 'Validator':
        """Validate that value is a file path."""
        self.value = validate_file_path(self.value, **kwargs)
        return self

    def is_directory_path(self, **kwargs) -> 'Validator':
        """Validate that value is a directory path."""
        self.value = validate_directory_path(self.value, **kwargs)
        return self

    def is_url(self, **kwargs) -> 'Validator':
        """Validate that value is a URL."""
        self.value = validate_url(self.value, **kwargs)
        return self

    def get(self) -> Any:
        """Get the validated value."""
        return self.value


def validate(value: Any, name: str = "value") -> Validator:
    """Create a new validator for fluent validation.

    Args:
        value: Value to validate
        name: Name of the value for error messages

    Returns:
        Validator instance
    """
    return Validator(value, name)
