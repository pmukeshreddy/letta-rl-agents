# File Operations

## Reading Files
- Text: `Path(file).read_text(encoding='utf-8')`
- Binary: `Path(file).read_bytes()`
- Large files: Use `open()` with iteration for memory efficiency
- CSV: `pandas.read_csv()` or `csv.reader()`
- JSON: `json.load(open(file))` or `Path(file).read_text()` then `json.loads()`

## Writing Files
- Always use context managers: `with open(file, 'w') as f:`
- Atomic writes: Write to temp file, then rename
- Create parent directories: `Path(file).parent.mkdir(parents=True, exist_ok=True)`

## Common Pitfalls
- Not specifying encoding (use `utf-8` explicitly)
- Not handling `FileNotFoundError`
- Writing without creating parent directories
- Not closing file handles (use `with` statements)
- Race conditions with concurrent writes

## Safe File Writing
```python
from pathlib import Path
import tempfile
import shutil

def safe_write(path: str, content: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temp file first
    with tempfile.NamedTemporaryFile(
        mode='w', 
        dir=path.parent, 
        delete=False,
        suffix='.tmp'
    ) as f:
        f.write(content)
        temp_path = f.name
    
    # Atomic rename
    shutil.move(temp_path, path)
```

## Verification
- Check file exists after write: `Path(file).exists()`
- Verify content: Read back and compare
- Check file size is reasonable
