# API Integration

## Approaches
- Use `requests` for simple sync HTTP calls
- Use `httpx` for async HTTP (preferred for concurrent calls)
- Use `aiohttp` for high-performance async scenarios

## Authentication Patterns
- Bearer tokens: `headers={"Authorization": f"Bearer {token}"}`
- API keys: Query params or headers depending on API
- OAuth2: Use `authlib` or `requests-oauthlib`

## Common Pitfalls
- Not handling rate limits (implement exponential backoff)
- Missing timeout parameters (always set `timeout=30`)
- Not validating response status codes
- Ignoring pagination in list endpoints

## Error Handling
```python
import httpx

async def api_call(url: str, retries: int = 3):
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=30)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:  # Rate limited
                await asyncio.sleep(2 ** attempt)
                continue
            raise
        except httpx.TimeoutException:
            if attempt == retries - 1:
                raise
            continue
```

## Verification
- Check response status code is 2xx
- Validate response schema matches expectations
- Test with invalid inputs to verify error handling
