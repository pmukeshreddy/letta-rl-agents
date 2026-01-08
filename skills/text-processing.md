# Text Processing

Process, clean, and transform text data.

## When to Use
- Cleaning messy text data
- Extracting patterns with regex
- Text normalization
- NLP preprocessing

## Approach
1. Use regex for pattern matching
2. Use `unicodedata` for normalization
3. Use `nltk` or `spacy` for NLP tasks
4. Chain transformations in pipelines

## Code Examples

```python
import re
import unicodedata

def clean_text(text):
    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove special characters
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower()

# Extract emails
def extract_emails(text):
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    return re.findall(pattern, text)

# Tokenize with nltk
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
```

## Pitfalls
- Unicode edge cases (emojis, RTL text)
- Regex can be slow on large texts — compile patterns
- Language-specific tokenization rules
- Don't over-clean — preserve meaningful punctuation
