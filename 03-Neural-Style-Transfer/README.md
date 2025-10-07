# Neural Style Transfer

Generate an artwork by blending the content of one image with the style of another using neural style transfer.

## Setup
- Python 3.9+
- GPU recommended for faster results (CUDA/cuDNN if using PyTorch/TensorFlow)
- Install dependencies from repo root:
  - `pip install -r ../requirements.txt` (from this folder)
  - or `pip install -r requirements.txt` (from repo root)

## Inputs
- `examples/ref_content_image.jpg` — example content image
- `examples/vector-illustration-avatar-dummy.jpg` — example style image

## Usage
Run from this folder:

```bash
python neural_style_transfer.py --content examples/ref_content_image.jpg \
  --style examples/vector-illustration-avatar-dummy.jpg --steps 500 --style_weight 1e6
```

Common flags:
- `--content PATH` — content image path
- `--style PATH` — style image path
- `--steps INT` — optimization steps/iterations
- `--style_weight FLOAT` — style loss weight
- `--content_weight FLOAT` — content loss weight
- `--output PATH` — output image path

If flags are omitted, the script may run with default example images.

## Outputs
- `output_neural_art_1.png` — generated stylized image

## Examples
```bash
# Fast test run
python neural_style_transfer.py --content examples/ref_content_image.jpg \
  --style examples/vector-illustration-avatar-dummy.jpg --steps 200 --output output_neural_art_1.png

# Higher quality
python neural_style_transfer.py --content my_photo.jpg --style my_style.jpg \
  --steps 1000 --style_weight 2e6 --content_weight 1e5 --output my_art.png
```

## Notes
- Larger images and higher step counts produce better results but take longer.
- Ensure images are reasonably sized; scripts often resize to manageable dimensions to fit memory.
