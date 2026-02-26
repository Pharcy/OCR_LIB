"""
PARALLEL Batch PDF to Articles Processor
Optimized version using multiprocessing for handling hundreds to thousands of files.

This version parallelizes OCR processing for significant speed improvements.
LLM calls remain serial due to Ollama limitations

Usage:
    python batch_pdf_to_articles_parallel.py <input_directory> [output_directory] [--workers N]
    
Example:
    python batch_pdf_to_articles_parallel.py ./newspapers ./output --workers 4
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import time
from typing import Optional, List, Tuple, Dict
from multiprocessing import Pool, Manager, cpu_count
from functools import partial

# Import functions from the original scripts
from Pdf_to_json import ocr_pdf, preprocess_image
from article_seperate import call_ollama_llm, parse_articles, extract_text_from_json


class ParallelBatchProcessor:
    """Parallel batch processor for PDF to articles workflow"""
    
    def __init__(self, input_dir: str, output_dir: Optional[str] = None, 
                 num_workers: Optional[int] = None):
        self.input_dir = Path(input_dir)
        
        # Set number of workers (default: CPU count - 1, min 1)
        if num_workers is None:
            self.num_workers = max(1, cpu_count() - 1)
        else:
            self.num_workers = max(1, min(num_workers, cpu_count()))
        
        # Set up output directories
        if output_dir is None:
            self.base_output_dir = self.input_dir / "processed_output"
        else:
            self.base_output_dir = Path(output_dir)
        
        self.ocr_output_dir = self.base_output_dir / "ocr_json"
        self.articles_output_dir = self.base_output_dir / "articles"
        
        # Create output directories
        self.ocr_output_dir.mkdir(parents=True, exist_ok=True)
        self.articles_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics (will be managed by multiprocessing.Manager)
        self.stats = {
            "total_pdfs": 0,
            "ocr_success": 0,
            "ocr_failed": 0,
            "ocr_skipped": 0,
            "total_articles_extracted": 0,
            "article_separation_failed": 0,
            "start_time": None,
            "end_time": None
        }
    
    def find_all_pdfs(self) -> List[Path]:
        """Recursively find all PDF files in input directory"""
        pdf_files = []
        pdf_files.extend(self.input_dir.rglob("*.pdf"))
        pdf_files.extend(self.input_dir.rglob("*.PDF"))
        return sorted(pdf_files)
    
    def get_relative_path(self, pdf_path: Path) -> Path:
        """Get relative path from input directory"""
        try:
            return pdf_path.relative_to(self.input_dir)
        except ValueError:
            return Path(pdf_path.name)
    
    @staticmethod
    def process_ocr_worker(args: Tuple[Path, Path, Path, Path]) -> Dict:
        """
        Worker function for parallel OCR processing.
        This is a static method to ensure it's picklable for multiprocessing.
        
        Returns: dictionary with results
        """
        pdf_path, input_dir, ocr_output_dir, articles_output_dir = args
        
        result = {
            "pdf_path": str(pdf_path),
            "success": False,
            "ocr_success": False,
            "ocr_skipped": False,
            "num_articles": 0,
            "error": None,
            "json_path": None
        }
        
        try:
            # Get relative path
            try:
                relative_path = pdf_path.relative_to(input_dir)
            except ValueError:
                relative_path = Path(pdf_path.name)
            
            pdf_stem = pdf_path.stem
            rel_dir = relative_path.parent
            
            # Create subdirectories
            ocr_subdir = ocr_output_dir / rel_dir
            ocr_subdir.mkdir(parents=True, exist_ok=True)
            
            json_output_path = ocr_subdir / f"{pdf_stem}.json"
            result["json_path"] = str(json_output_path)
            
            # Check if OCR already done
            if json_output_path.exists():
                result["ocr_skipped"] = True
                result["ocr_success"] = True
                result["success"] = True
            else:
                # Perform OCR
                ocr_success = ocr_pdf(pdf_path, ocr_subdir)
                result["ocr_success"] = ocr_success
                result["success"] = ocr_success
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def process_articles_for_json(self, json_path: Path, relative_pdf_path: Path) -> Tuple[bool, int]:
        """
        Process a single JSON file to extract articles.
        This is done serially due to Ollama LLM limitations.
        """
        try:
            pdf_stem = relative_pdf_path.stem
            rel_dir = relative_pdf_path.parent
            
            articles_subdir = self.articles_output_dir / rel_dir / pdf_stem
            articles_subdir.mkdir(parents=True, exist_ok=True)
            
            # Load OCR JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
            
            # Extract text
            text_content = extract_text_from_json(ocr_data)
            
            if not text_content:
                return False, 0
            
            # Prepare source metadata
            source_metadata = {
                "source_file": str(relative_pdf_path),
                "total_pages": ocr_data.get('total_pages', 'Unknown'),
                "processed_date": ocr_data.get('processed_date', 'Unknown'),
                "page_number": ocr_data.get('pages', [{}])[0].get('page_number', 1) if ocr_data.get('pages') else 1
            }
            
            # Call LLM
            llm_response = call_ollama_llm(text_content)
            
            if not llm_response:
                return False, 0
            
            # Parse articles
            articles = parse_articles(llm_response, source_metadata)
            
            if not articles:
                return False, 0
            
            # Save articles
            import re
            for article in articles:
                safe_title = re.sub(r'[^\w\s-]', '', article['title'])
                safe_title = re.sub(r'[-\s]+', '_', safe_title)
                safe_title = safe_title[:50]
                
                filename = f"article_{article['article_id']:03d}_{safe_title}.json"
                filepath = articles_subdir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(article, f, indent=2, ensure_ascii=False)
            
            return True, len(articles)
            
        except Exception as e:
            print(f"    [ARTICLES] Error processing {json_path}: {str(e)}")
            return False, 0
    
    def process_all(self):
        """Process all PDFs using parallel OCR and serial article extraction"""
        print(f"\n{'#'*80}")
        print(f"# PARALLEL BATCH PDF TO ARTICLES PROCESSOR")
        print(f"{'#'*80}")
        print(f"\nInput directory:  {self.input_dir}")
        print(f"Output directory: {self.base_output_dir}")
        print(f"  - OCR JSONs:    {self.ocr_output_dir}")
        print(f"  - Articles:     {self.articles_output_dir}")
        print(f"\nParallel workers: {self.num_workers} (for OCR phase)")
        print(f"Article extraction: Serial (Ollama limitation)")
        
        # Find all PDFs
        pdf_files = self.find_all_pdfs()
        self.stats["total_pdfs"] = len(pdf_files)
        
        if not pdf_files:
            print(f"\n✗ No PDF files found in '{self.input_dir}'")
            return
        
        print(f"\n✓ Found {len(pdf_files)} PDF files")
        
        self.stats["start_time"] = time.time()
        
        # ============================================================
        # PHASE 1: PARALLEL OCR PROCESSING
        # ============================================================
        print(f"\n{'='*80}")
        print(f"PHASE 1: OCR PROCESSING ({self.num_workers} parallel workers)")
        print(f"{'='*80}\n")
        
        # Prepare arguments for parallel processing
        ocr_args = [
            (pdf_path, self.input_dir, self.ocr_output_dir, self.articles_output_dir)
            for pdf_path in pdf_files
        ]
        
        # Process PDFs in parallel
        ocr_start = time.time()
        ocr_results = []
        
        with Pool(processes=self.num_workers) as pool:
            # Use imap_unordered for better performance with progress tracking
            total = len(pdf_files)
            completed = 0
            
            for result in pool.imap_unordered(self.process_ocr_worker, ocr_args):
                completed += 1
                ocr_results.append(result)
                
                # Update statistics
                if result["ocr_success"]:
                    if result["ocr_skipped"]:
                        self.stats["ocr_skipped"] += 1
                    else:
                        self.stats["ocr_success"] += 1
                else:
                    self.stats["ocr_failed"] += 1
                
                # Print progress
                status = "✓" if result["ocr_success"] else "✗"
                skip_note = " (cached)" if result.get("ocr_skipped") else ""
                print(f"[{completed}/{total}] {status} OCR: {Path(result['pdf_path']).name}{skip_note}")
        
        ocr_time = time.time() - ocr_start
        print(f"\nOCR Phase Complete: {ocr_time:.2f}s ({ocr_time/60:.2f} minutes)")
        print(f"  Success: {self.stats['ocr_success']}")
        print(f"  Cached:  {self.stats['ocr_skipped']}")
        print(f"  Failed:  {self.stats['ocr_failed']}")
        
        # ============================================================
        # PHASE 2: SERIAL ARTICLE EXTRACTION
        # ============================================================
        print(f"\n{'='*80}")
        print(f"PHASE 2: ARTICLE EXTRACTION (serial processing)")
        print(f"{'='*80}\n")
        print(f"Note: This phase is serial due to Ollama LLM limitations.")
        print(f"Each document makes a synchronous call to the LLM.\n")
        
        article_start = time.time()
        
        # Process each successful OCR result for article extraction
        successful_ocr = [r for r in ocr_results if r["success"] and r["json_path"]]
        
        for idx, ocr_result in enumerate(successful_ocr, 1):
            pdf_path = Path(ocr_result["pdf_path"])
            json_path = Path(ocr_result["json_path"])
            relative_path = self.get_relative_path(pdf_path)
            
            print(f"[{idx}/{len(successful_ocr)}] Processing: {relative_path.name}")
            
            success, num_articles = self.process_articles_for_json(json_path, relative_path)
            
            if success and num_articles > 0:
                self.stats["total_articles_extracted"] += num_articles
                print(f"    ✓ Extracted {num_articles} articles")
            else:
                self.stats["article_separation_failed"] += 1
                print(f"    ✗ No articles extracted")
        
        article_time = time.time() - article_start
        print(f"\nArticle Extraction Complete: {article_time:.2f}s ({article_time/60:.2f} minutes)")
        
        self.stats["end_time"] = time.time()
        self.print_summary()
    
    def print_summary(self):
        """Print processing summary"""
        total_time = self.stats["end_time"] - self.stats["start_time"]
        
        print(f"\n\n{'#'*80}")
        print(f"# PROCESSING COMPLETE")
        print(f"{'#'*80}")
        print(f"\nTotal PDFs:              {self.stats['total_pdfs']}")
        print(f"  OCR successful:        {self.stats['ocr_success']}")
        print(f"  OCR cached (skipped):  {self.stats['ocr_skipped']}")
        print(f"  OCR failed:            {self.stats['ocr_failed']}")
        print(f"  Total articles found:  {self.stats['total_articles_extracted']}")
        print(f"  Article sep. failed:   {self.stats['article_separation_failed']}")
        
        print(f"\nTotal processing time:   {total_time:.2f}s ({total_time/60:.2f} minutes)")
        
        if self.stats['total_pdfs'] > 0:
            avg_time = total_time / self.stats['total_pdfs']
            print(f"Average per PDF:         {avg_time:.2f}s")
        
        print(f"\nOutput locations:")
        print(f"  OCR JSONs:    {self.ocr_output_dir}")
        print(f"  Articles:     {self.articles_output_dir}")
        
        # Save summary
        summary_path = self.base_output_dir / "processing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                "input_directory": str(self.input_dir),
                "output_directory": str(self.base_output_dir),
                "processing_date": datetime.now().isoformat(),
                "num_workers": self.num_workers,
                "statistics": self.stats,
                "total_time_seconds": total_time,
                "total_time_minutes": total_time / 60
            }, f, indent=2)
        
        print(f"\n✓ Summary saved to: {summary_path}")
        print(f"\n{'#'*80}\n")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExample:")
        print("  python batch_pdf_to_articles_parallel.py ./newspapers")
        print("  python batch_pdf_to_articles_parallel.py ./newspapers ./output --workers 4")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_directory = None
    num_workers = None
    
    # Parse arguments
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--workers":
            if i + 1 < len(sys.argv):
                try:
                    num_workers = int(sys.argv[i + 1])
                    i += 2
                    continue
                except ValueError:
                    print(f"✗ Error: --workers must be followed by a number")
                    sys.exit(1)
        elif not arg.startswith("--"):
            output_directory = arg
        i += 1
    
    # Validate input directory
    if not os.path.exists(input_directory):
        print(f"✗ Error: Input directory '{input_directory}' does not exist.")
        sys.exit(1)
    
    # Create and run processor
    processor = ParallelBatchProcessor(input_directory, output_directory, num_workers)
    processor.process_all()


if __name__ == "__main__":
    main()


