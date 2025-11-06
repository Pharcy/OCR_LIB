import json
import requests
import re

def call_ollama_llm(text, model="gpt-oss:latest"):
    """
    Send text to Ollama LLM and get response
    """
    url = "http://localhost:11434/api/generate"
    
    prompt = f"""You are tasked with separating newspaper articles from the following text. 
Each article should be clearly delimited and formatted with a title.

Format your response EXACTLY like this for each article:
<<<ARTICLE_START>>>
TITLE: [Extract the article's headline/title here]
CONTENT:
[Article content goes here]
<<<ARTICLE_END>>>

IMPORTANT: 
- Every article MUST have a TITLE: line at the start
- After TITLE:, add a CONTENT: line before the article text
- Use <<<ARTICLE_START>>> and <<<ARTICLE_END>>> as delimiters for EACH article

Here is the text to process:

{text}"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return None

def parse_articles(llm_response, source_metadata):
    """
    Parse individual articles from LLM response using delimiters
    Extract title and content, and include source metadata
    """
    # Split by article markers
    article_pattern = r'<<<ARTICLE_START>>>(.*?)<<<ARTICLE_END>>>'
    articles = re.findall(article_pattern, llm_response, re.DOTALL)
    
    # Clean up each article and extract title
    parsed_articles = []
    for i, article in enumerate(articles, 1):
        article = article.strip()
        if not article:
            continue
        
        # Try to extract title and content
        title = f"Article {i}"  # Default title
        content = article
        
        # Look for TITLE: and CONTENT: markers
        title_match = re.search(r'TITLE:\s*(.+?)(?:\n|$)', article, re.IGNORECASE)
        if title_match:
            title = title_match.group(1).strip()
            # Remove the TITLE: line from content
            content = re.sub(r'TITLE:\s*.+?(?:\n|$)', '', article, flags=re.IGNORECASE)
        
        # Remove CONTENT: marker if present
        content = re.sub(r'^\s*CONTENT:\s*', '', content, flags=re.IGNORECASE | re.MULTILINE)
        content = content.strip()
        
        parsed_articles.append({
            "article_id": i,
            "title": title,
            "content": content,
            "source_file": source_metadata.get("source_file", "Unknown"),
            "source_page": source_metadata.get("page_number", "Unknown"),
            "total_source_pages": source_metadata.get("total_pages", "Unknown"),
            "processed_date": source_metadata.get("processed_date", "Unknown")
        })
    
    return parsed_articles

def extract_text_from_json(data):
    """
    Extract text from the JSON structure
    Handles the nested pages structure
    """
    # Check if 'pages' exists and has content
    if 'pages' in data and isinstance(data['pages'], list) and len(data['pages']) > 0:
        # Concatenate text from all pages
        all_text = []
        for page in data['pages']:
            if 'text' in page:
                all_text.append(page['text'])
        
        return '\n\n'.join(all_text)
    
    # Fallback: check for direct 'text' field
    elif 'text' in data:
        return data['text']
    
    else:
        return None

def process_json_file(json_file_path, output_dir="output_articles"):
    """
    Main function to process JSON file and extract articles
    Saves each article as a separate JSON file
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read JSON file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{json_file_path}'")
        return
    
    # Extract text from JSON
    text_content = extract_text_from_json(data)
    
    if not text_content:
        print("Error: Could not find text content in JSON")
        print("Available keys:", list(data.keys()))
        return
    
    # Prepare source metadata
    source_metadata = {
        "source_file": data.get('source_file', 'Unknown'),
        "total_pages": data.get('total_pages', 'Unknown'),
        "processed_date": data.get('processed_date', 'Unknown'),
        "page_number": data.get('pages', [{}])[0].get('page_number', 1) if data.get('pages') else 1
    }
    
    print(f"Extracted {len(text_content.split())} words from JSON")
    print(f"Source file: {source_metadata['source_file']}")
    print(f"Total pages: {source_metadata['total_pages']}")
    
    # Send to LLM
    print("\nSending text to Ollama LLM (gpt-oss)...")
    llm_response = call_ollama_llm(text_content)
    
    if not llm_response:
        print("Failed to get response from LLM")
        return
    
    print("Received response from LLM")
    
    # Parse articles
    print("\nParsing articles...")
    articles = parse_articles(llm_response, source_metadata)
    
    print(f"\nFound {len(articles)} articles")
    
    # Save each article as a separate JSON file
    saved_files = []
    for article in articles:
        # Create safe filename from title
        safe_title = re.sub(r'[^\w\s-]', '', article['title'])
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        safe_title = safe_title[:50]  # Limit length
        
        filename = f"article_{article['article_id']}_{safe_title}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(article, f, indent=2, ensure_ascii=False)
        
        saved_files.append(filename)
        
        # Display preview
        preview = article['content'][:150] + "..." if len(article['content']) > 150 else article['content']
        print(f"\n--- Article {article['article_id']}: {article['title']} ---")
        print(f"File: {filename}")
        print(preview)
    
    # # Also save a master index file
    # index_file = os.path.join(output_dir, "_index.json")
    # index_data = {
    #     "source_file": source_metadata['source_file'],
    #     "total_pages": source_metadata['total_pages'],
    #     "processed_date": source_metadata['processed_date'],
    #     "total_articles": len(articles),
    #     "articles": [
    #         {
    #             "article_id": a['article_id'],
    #             "title": a['title'],
    #             "filename": saved_files[i]
    #         }
    #         for i, a in enumerate(articles)
    #     ]
    # }
    
    # with open(index_file, 'w', encoding='utf-8') as f:
    #     json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ {len(articles)} articles saved to '{output_dir}/' directory")
    print(f"✓ Index saved to '{index_file}'")
    
    return articles

if __name__ == "__main__":
    # Replace with your JSON file path
    json_file_path = "output/wyubdi_20040901_0006.json"
    
    # Process the file (will save to 'output_articles' directory by default)
    articles = process_json_file(json_file_path)
