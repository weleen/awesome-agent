import requests
from bs4 import BeautifulSoup

# URL of the RSS 2024 papers page
# url = "https://roboticsconference.org/2024/program/papers/"
# url of the RSS papers page
url = "https://roboticsconference.org/program/papers/"

# Fetch the page content
response = requests.get(url)
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all paper entries (adjust selectors as needed)
    paper_links = soup.find_all('a', href=True)
    
    # List to store formatted entries
    papers = []

    for link in paper_links:
        if "papers/" in link['href']:  # Filter only paper links
            paper_title = link.text.strip()
            paper_url = link['href']
            formatted = f"- [{paper_title}]({paper_url}), RSS, 2025"
            papers.append(formatted)
    
    # Save or print the result
    with open("RSS2025_Papers.md", "w") as file:
        file.write("\n".join(papers))
    print("Markdown file created: RSS2025_Papers.md")
else:
    print(f"Failed to fetch the page. Status code: {response.status_code}")

