# Debugging

Find and fix bugs in code.

## When to Use
- Code produces wrong output
- Unexpected exceptions
- Performance issues
- Memory leaks

## Approach
1. Reproduce the bug consistently
2. Use print/logging to trace execution
3. Use debugger for step-through
4. Binary search to isolate the problem

## Code Examples

```python
# Logging for debugging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def problematic_function(data):
    logger.debug(f"Input: {data}")
    result = process(data)
    logger.debug(f"Output: {result}")
    return result

# Using pdb
import pdb

def buggy_function():
    x = calculate()
    pdb.set_trace()  # Breakpoint
    return x * 2

# Exception context
try:
    risky_operation()
except Exception as e:
    logger.exception(f"Failed: {e}")
    raise

# Memory profiling
from memory_profiler import profile

@profile
def memory_hungry():
    data = [i for i in range(1000000)]
    return sum(data)
```

## Pitfalls
- Don't debug in production
- Remove debug statements before commit
- Check edge cases (empty, None, large inputs)
- Verify assumptions — they're often wrong
