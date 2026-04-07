# Pharcy

A pipeline for extracting and structuring data from PDF documents into JSON format.

## Core Files

These are the only files that matter for running the pipeline:

| File | Description |
|------|-------------|
| `Batch_pdf_parallel.py` | **Main entry point.** Runs the full pipeline in parallel across multiple PDFs. |
| `Pdf_to_json.py` | Dependency of `Batch_pdf_parallel.py` — converts raw PDFs into JSON. |
| `article_seperate.py` | Dependency of `Batch_pdf_parallel.py` — handles article separation logic. |

### Running the pipeline

```bash
 python batch_pdf_to_articles_parallel.py <input_directory> [output_directory] [--workers N]
```

---

## Everything Else

The remaining files are experimental or legacy test scripts. They are **not part of the main workflow** and may or may not be useful as reference:

- `article_seperate_update.py`
- `grayscale_parse.py`
- `lay_parse.py`
- `layparse_testing.py`
- `scan_doc.py`
