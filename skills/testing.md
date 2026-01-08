# Testing

Write and run tests for code quality.

## When to Use
- Validating function behavior
- Regression testing
- Integration testing
- Test-driven development

## Approach
1. Use `pytest` as the test framework
2. Write unit tests for individual functions
3. Use fixtures for setup/teardown
4. Mock external dependencies

## Code Examples

```python
import pytest
from unittest.mock import Mock, patch

# Basic test
def test_add():
    assert add(2, 3) == 5

# Parametrized tests
@pytest.mark.parametrize("input,expected", [
    (1, 1),
    (2, 4),
    (3, 9),
])
def test_square(input, expected):
    assert square(input) == expected

# Fixtures
@pytest.fixture
def database():
    db = create_test_db()
    yield db
    db.cleanup()

def test_query(database):
    result = database.query("SELECT 1")
    assert result == 1

# Mocking
@patch("module.external_api")
def test_with_mock(mock_api):
    mock_api.return_value = {"status": "ok"}
    result = function_that_calls_api()
    assert result["status"] == "ok"
```

## Pitfalls
- Don't test implementation details
- Avoid flaky tests (random, time-dependent)
- Mock at the right level
- Keep tests fast — slow tests don't get run
