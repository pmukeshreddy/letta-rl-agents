# Web Scraping

Extract data from websites.

## When to Use
- Collecting data from web pages
- Monitoring website changes
- Extracting structured data from HTML
- Automating data collection

## Approach
1. Use `requests` for simple HTTP fetching
2. Use `BeautifulSoup` for HTML parsing
3. Use `Selenium` for JavaScript-rendered pages
4. Respect robots.txt and rate limits

## Code Examples

```python
import requests
from bs4 import BeautifulSoup

def scrape_page(url):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Extract all links
    links = [a["href"] for a in soup.find_all("a", href=True)]
    
    # Extract text from specific element
    title = soup.find("h1").text.strip()
    
    return {"title": title, "links": links}

# For JavaScript pages
from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()
driver.get(url)
element = driver.find_element(By.CSS_SELECTOR, ".dynamic-content")
```

## Pitfalls
- Always check robots.txt compliance
- Implement rate limiting (1-2 req/sec)
- Handle pagination properly
- Some sites block scrapers — use rotating proxies
- JavaScript rendering requires Selenium/Playwright
