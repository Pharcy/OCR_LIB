#!/usr/bin/env python3
"""
Batch Article Separation Script (Ollama Only)

This script processes existing OCR JSON files and uses Ollama LLM to separate
them into individual article files. Run this AFTER you've completed OCR processing.

Usage:
    python batch_article_separate.py <json_directory> [output_directory] [--max-files N]
    
Examples:
    python batch_article_separate.py ./ocr_json
    python batch_article_separate.py ./ocr_json ./articles --max-files 100
    python batch_article_separate.py ./ocr_json/newspapers ./output/articles --max-files 50
"""

import os
import sys
import json
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import requests


def call_ollama_llm(text_content: str) -> Optional[str]:
    """
    Call Ollama LLM to separate articles from newspaper text.
    Uses the gpt-oss model (or fallback to llama2).
    """
    url = "http://localhost:11434/api/generate"
    
    prompt = f"""You are analyzing a newspaper page that contains one or more articles. Your task is to:

1. Identify each distinct article in the text
2. For each article, extract:
   - A clear, descriptive title (create one if not present)
   - The complete article content

Format your response EXACTLY like this for EACH article:

ARTICLE_START
TITLE: [Article Title Here]
CONTENT:
[Full article content here]
ARTICLE_END

Here is the newspaper text to analyze:

{text_content}

Remember: Use ARTICLE_START and ARTICLE_END markers for each article. Include a TITLE and CONTENT for each one."""

    payload = {
        "model": "gpt-oss",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        return result.get('response', '')
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama: {e}")
        return None


def parse_articles(llm_response: str, source_metadata: dict) -> List[dict]:
    """
    Parse the LLM response to extract individual articles.
    """
    articles = []
    
    # Split by ARTICLE_START markers
    article_blocks = llm_response.split('ARTICLE_START')
    
    for i, block in enumerate(article_blocks[1:], 1):  # Skip first empty split
        # Extract content before ARTICLE_END
        if 'ARTICLE_END' in block:
            block = block.split('ARTICLE_END')[0]
        
        # Extract title
        title_match = re.search(r'TITLE:\s*(.+?)(?:\n|CONTENT:)', block, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else f"Article {i}"
        
        # Extract content
        content_match = re.search(r'CONTENT:\s*(.+)', block, re.IGNORECASE | re.DOTALL)
        content = content_match.group(1).strip() if content_match else block.strip()
        
        # Clean up content
        content = re.sub(r'\n{3,}', '\n\n', content)  # Normalize whitespace
        
        if content and len(content) > 50:  # Only include substantial articles
            article = {
                "article_id": f"{i:03d}",
                "title": title,
                "content": content,
                "word_count": len(content.split()),
                "source_file": source_metadata.get('source_file'),
                "source_date": source_metadata.get('source_date'),
                "page_number": source_metadata.get('page_number', 1),
                "extraction_date": datetime.now().isoformat()
            }
            articles.append(article)
    
    return articles


def extract_text_from_json(json_path: str) -> tuple:
    """
    Extract text content from OCR JSON file.
    Returns: (text_content, source_metadata)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract all text from all pages
    text_parts = []
    for page in data.get('pages', []):
        if 'text' in page:
            text_parts.append(page['text'])
    
    text_content = '\n\n'.join(text_parts)
    
    # Extract metadata
    source_metadata = {
        "source_file": data.get('source_file', os.path.basename(json_path)),
        "total_pages": len(data.get('pages', [])),
        "source_date": data.get('processed_date', 'Unknown'),
        "page_number": data.get('pages', [{}])[0].get('page_number', 1) if data.get('pages') else 1
    }
    
    return text_content, source_metadata


def process_single_json(json_path: Path, output_base_dir: Path, relative_path: Path) -> dict:
    """
    Process a single JSON file to extract and separate articles.
    Returns statistics dictionary.
    """
    result = {
        "json_path": str(json_path),
        "success": False,
        "articles_extracted": 0,
        "error": None,
        "skipped": False
    }
    
    # Create output directory maintaining folder structure
    output_dir = output_base_dir / relative_path.parent / json_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed (look for any article files)
    existing_articles = list(output_dir.glob("article_*.json"))
    if existing_articles:
        result["skipped"] = True
        result["success"] = True
        result["articles_extracted"] = len(existing_articles)
        return result
    
    try:
        # Extract text from JSON
        text_content, source_metadata = extract_text_from_json(str(json_path))
        
        if not text_content or len(text_content.split()) < 50:
            result["error"] = "Insufficient text content"
            return result
        
        # Send to LLM
        llm_response = call_ollama_llm(text_content)
        
        if not llm_response:
            result["error"] = "Failed to get LLM response"
            return result
        
        # Parse articles
        articles = parse_articles(llm_response, source_metadata)
        
        if not articles:
            result["error"] = "No articles parsed from LLM response"
            return result
        
        # Save each article as separate JSON file
        for article in articles:
            # Create safe filename from title
            safe_title = re.sub(r'[^\w\s-]', '', article['title'])
            safe_title = re.sub(r'[-\s]+', '_', safe_title)
            safe_title = safe_title[:50]  # Limit length
            
            filename = f"article_{article['article_id']}_{safe_title}.json"
            filepath = output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(article, f, indent=2, ensure_ascii=False)
        
        # Save index file
        index_data = {
            "source_file": source_metadata['source_file'],
            "total_articles": len(articles),
            "articles": [
                {
                    "article_id": a['article_id'],
                    "title": a['title'],
                    "word_count": a['word_count']
                }
                for a in articles
            ],
            "processed_date": datetime.now().isoformat()
        }
        
        with open(output_dir / "_index.json", 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        
        result["success"] = True
        result["articles_extracted"] = len(articles)
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def find_all_json_files(input_dir: Path) -> List[Path]:
    """Recursively find all JSON files in input directory"""
    json_files = []
    json_files.extend(input_dir.rglob("*.json"))
    # Exclude index files
    json_files = [f for f in json_files if not f.name.startswith('_index')]
    return sorted(json_files)


def batch_process_articles(json_dir: str, output_dir: Optional[str] = None, max_files: Optional[int] = None):
    """
    Main batch processing function for article separation.
    
    Args:
        json_dir: Directory containing OCR JSON files
        output_dir: Directory for article outputs (default: json_dir/articles)
        max_files: Maximum number of files to process (None = all files)
    """
    input_path = Path(json_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory not found: {json_dir}")
        sys.exit(1)
    
    # Setup output directory
    if output_dir is None:
        output_path = input_path.parent / "articles"
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files
    print(f"Scanning for JSON files in: {input_path}")
    json_files = find_all_json_files(input_path)
    
    if not json_files:
        print("No JSON files found!")
        sys.exit(1)
    
    # Apply max_files limit
    total_found = len(json_files)
    if max_files is not None and max_files < len(json_files):
        json_files = json_files[:max_files]
        print(f"Found {total_found} files, processing first {max_files} as requested")
    else:
        print(f"Found {len(json_files)} JSON files")
    
    # Initialize statistics
    stats = {
        "total_files": len(json_files),
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "total_articles": 0,
        "start_time": datetime.now()
    }
    
    print(f"Output directory: {output_path}")
    print(f"\n{'='*80}")
    print(f"ARTICLE SEPARATION (Ollama LLM Processing)")
    print(f"{'='*80}\n")
    
    # Process each JSON file
    for idx, json_file in enumerate(json_files, 1):
        # Get relative path for maintaining structure
        try:
            relative_path = json_file.relative_to(input_path)
        except ValueError:
            relative_path = Path(json_file.name)
        
        print(f"[{idx}/{len(json_files)}] Processing: {json_file.name}")
        
        start_time = time.time()
        result = process_single_json(json_file, output_path, relative_path)
        process_time = time.time() - start_time
        
        # Update statistics
        if result["success"]:
            if result["skipped"]:
                stats["skipped"] += 1
                stats["total_articles"] += result["articles_extracted"]
                print(f"  ⊙ Skipped (already processed): {result['articles_extracted']} articles (cached)")
            else:
                stats["successful"] += 1
                stats["total_articles"] += result["articles_extracted"]
                print(f"  ✓ Success: {result['articles_extracted']} articles extracted ({process_time:.1f}s)")
        else:
            stats["failed"] += 1
            print(f"  ✗ Failed: {result['error']}")
        
        # Add small delay between requests to avoid overwhelming Ollama
        if idx < len(json_files):
            time.sleep(0.5)
    
    # Final statistics
    stats["end_time"] = datetime.now()
    total_time = (stats["end_time"] - stats["start_time"]).total_seconds()
    
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total files processed: {stats['total_files']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Cached/Skipped: {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")
    print(f"Total articles extracted: {stats['total_articles']}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    if stats['successful'] > 0:
        print(f"Average time per file: {total_time/stats['successful']:.2f}s")
    
    # Save summary
    summary_file = output_path / "processing_summary.json"
    summary_data = {
        "statistics": {
            "total_files": stats["total_files"],
            "successful": stats["successful"],
            "skipped": stats["skipped"],
            "failed": stats["failed"],
            "total_articles_extracted": stats["total_articles"]
        },
        "processing_info": {
            "start_time": stats["start_time"].isoformat(),
            "end_time": stats["end_time"].isoformat(),
            "total_seconds": total_time,
            "input_directory": str(input_path),
            "output_directory": str(output_path),
            "max_files_limit": max_files
        }
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSummary saved to: {summary_file}")


def main():
    """Main entry point with argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch process OCR JSON files to separate articles using Ollama LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_article_separate.py ./ocr_json
  python batch_article_separate.py ./ocr_json ./articles --max-files 100
  python batch_article_separate.py ./data/newspapers ./output --max-files 50
        """
    )
    
    parser.add_argument('json_dir', help='Directory containing OCR JSON files')
    parser.add_argument('output_dir', nargs='?', help='Output directory for articles (default: json_dir/articles)')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to process')
    
    args = parser.parse_args()
    
    # Validate max_files
    if args.max_files is not None and args.max_files < 1:
        print("Error: --max-files must be at least 1")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"BATCH ARTICLE SEPARATION")
    print(f"{'='*80}\n")
    
    batch_process_articles(args.json_dir, args.output_dir, args.max_files)


if __name__ == "__main__":
    main()