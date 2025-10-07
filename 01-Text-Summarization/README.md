# Text Summarization

A simple extractive text summarizer that generates concise summaries from longer text documents.

## Setup
- Create and activate a Python 3.9+ environment
- Install dependencies from the root of the repo:
  - `pip install -r ../requirements.txt` (if running from this folder)
  - Or from repository root: `pip install -r requirements.txt`

## Inputs
- `sample.txt` — plain text sample
- `sample.pdf` — PDF sample (requires `pdfminer.six`/`PyPDF2` depending on implementation)
- `sample.doc` — Word document sample (requires `python-docx`)

## Usage
Run from this folder:

```bash
python text_summarizer.py --input sample.txt --ratio 0.2
```

Common flags:
- `--input PATH` — path to input file (.txt/.pdf/.doc)
- `--ratio FLOAT` — target summary length as a fraction of the original (e.g., 0.2)
- `--max_words INT` — alternatively, cap the summary by word count

If flags are not provided, the script defaults to example inputs where applicable.

## Outputs
- `output_summary.txt` — raw summarized text
- `output_text_summary.png` — optional visual preview of the summary
- `fianl_output.png` — final visualization (typo preserved in filename as in repo)

## Examples
```bash
# Summarize plain text to ~20% of original length
python text_summarizer.py --input sample.txt --ratio 0.2

# Summarize a PDF capped at 120 words
python text_summarizer.py --input sample.pdf --max_words 120
```

## Notes
- For PDFs/Docs, ensure relevant libraries are installed (see `requirements.txt`).
- Results are deterministic if the underlying algorithm is deterministic; otherwise, set a random seed if supported by the script.
