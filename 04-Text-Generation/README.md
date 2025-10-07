# Text Generation

Generate coherent text continuations using a simple generative language model.

## Setup
- Python 3.9+
- Install dependencies from repo root:
  - `pip install -r ../requirements.txt` (from this folder)
  - or `pip install -r requirements.txt` (from repo root)

## Data
- `examples.txt` — small corpus used for quick demos/testing

## Usage
Run from this folder:

```bash
python generative_text_model.py --prompt "Once upon a time" --max_tokens 80 --temperature 0.8
```

Common flags:
- `--prompt STRING` — starting text
- `--max_tokens INT` — number of tokens/words to generate
- `--temperature FLOAT` — sampling temperature (0.0-1.5)
- `--top_k INT` — limit sampling to top-k candidates (if supported)
- `--top_p FLOAT` — nucleus sampling (if supported)

## Outputs
- `output_text_generation.png` — optional visualization of generated text
- Console/text output with the generated continuation

## Examples
```bash
# Short creative continuation
python generative_text_model.py --prompt "In a quiet village," --max_tokens 60 --temperature 1.0

# More deterministic continuation
python generative_text_model.py --prompt "The algorithm converges when" --max_tokens 50 --temperature 0.2
```

## Notes
- Set a random seed if deterministic runs are needed (if supported by the script).
