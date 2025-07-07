import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import time
import re

def parse_paper_entry(entry_text):
    # Regular expression pattern to match paper entries
    title_pattern = r'<a href="([^"]+)">([^<]+)</a>'
    
    # Find title and forum link
    title_match = re.search(title_pattern, entry_text)
    if not title_match:
        return None
        
    forum_link = title_match.group(1)
    pdf_link = forum_link.replace('forum', 'pdf')
    title = title_match.group(2).strip()
    
    # Extract authors - everything between </b> and the next <br>
    authors_text = entry_text.split('</b>')[1].split('<br>')[1] if '</b>' in entry_text else ""
    authors = authors_text.strip()
    
    return {
        'title': title,
        'authors': authors,
        'forum_link': forum_link,
        'pdf_link': pdf_link
    }

def scrape_corl_papers():
    url = "https://2024.corl.org/program/papers"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Send GET request to the URL
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all paper entries
        papers = []
        
        # Find all div elements with class "YMEQtf"
        # TODO: class name is not equivalent in different collections, check it before scrap it.
        paper_elements = soup.find_all('div', class_="YMEQtf")
        
        for paper in paper_elements:
            try:
                # Find the link element within the div, link_element is str now.
                link_element = paper.find('div').attrs['data-code']
                entries = re.findall(r'<b>\d+\.\s*<a.*?<br>\s*<br>', link_element, re.DOTALL)

                for entry in entries:
                    paper_info = parse_paper_entry(entry)
                    if paper_info:
                        formatted = f"- [{paper_info['title']}]({paper_info['pdf_link']}), CoRL, 2024"
                        papers.append(formatted)

            except Exception as e:
                print(f"Error processing paper: {e}")
                continue
        
        # Save results to a Markdown file
        output_file = f'CoRL2024_Papers.md'
        with open(output_file, "w") as file:
            file.write("\n".join(papers))
        print("Markdown file created: RSS2024_Papers.md")
        print(f"Successfully scraped {len(papers)} papers")
        print(f"Results saved to {output_file}")
        
    except requests.RequestException as e:
        print(f"Error fetching the webpage: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    scrape_corl_papers()
