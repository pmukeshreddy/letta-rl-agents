# Error Handling

## Exception Hierarchy
```python
# Catch specific exceptions first, then general
try:
    result = risky_operation()
except FileNotFoundError:
    handle_missing_file()
except PermissionError:
    handle_permission_denied()
except OSError as e:
    handle_os_error(e)
except Exception as e:
    log_unexpected_error(e)
    raise
```

## Custom Exceptions
```python
class AppError(Exception):
    """Base exception for application."""
    pass

class ValidationError(AppError):
    """Raised when input validation fails."""
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")

class NotFoundError(AppError):
    """Raised when resource not found."""
    pass
```

## Retry Pattern
```python
from functools import wraps
import time

def retry(max_attempts=3, delay=1, backoff=2, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator

@retry(max_attempts=3, exceptions=(ConnectionError,))
def fetch_data():
    ...
```

## Common Pitfalls
- Catching too broad (bare `except:`)
- Swallowing exceptions without logging
- Not re-raising when appropriate
- Missing `finally` for cleanup

## Logging Errors
```python
import logging

logger = logging.getLogger(__name__)

try:
    operation()
except Exception as e:
    logger.exception("Operation failed")  # Includes traceback
    raise
```

## Context Managers for Cleanup
```python
from contextlib import contextmanager

@contextmanager
def managed_resource():
    resource = acquire_resource()
    try:
        yield resource
    finally:
        resource.cleanup()
```
