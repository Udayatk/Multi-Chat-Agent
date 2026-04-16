"""Web scraping module for extracting content from URLs."""

from typing import Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import streamlit as st


class WebScraper:
    """Web scraper for extracting content from URLs"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def extract_from_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract text content from a URL"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "meta", "link"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Get title
            title = soup.title.string if soup.title else url
            
            return {
                'url': url,
                'title': title,
                'content': text,
                'source_type': 'website'
            }
        except Exception as e:
            st.error(f"Error scraping {url}: {str(e)}")
            return None
