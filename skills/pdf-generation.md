# PDF Generation

## Approaches
- Use `reportlab` for programmatic PDF creation
- Use `weasyprint` for HTML-to-PDF conversion
- Use `fpdf2` for simple text-based PDFs

## Common Pitfalls
- Forgetting to close file handles
- Font embedding issues on different systems
- Memory issues with large documents

## Verification
- Check file size > 0
- Validate with `pdfinfo` command
- Open and verify page count

## Example
```python
from reportlab.pdfgen import canvas

def create_pdf(filename, content):
    c = canvas.Canvas(filename)
    c.drawString(100, 750, content)
    c.save()
```
