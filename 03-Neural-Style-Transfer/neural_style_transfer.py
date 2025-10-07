"""
NEURAL STYLE TRANSFER
A system that applies artistic styles from one image to another image
For example: Make your photo look like a Van Gogh painting!
"""

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import copy

# ============================================
# PART 1: CHECK IF GPU IS AVAILABLE
# ============================================
def setup_device():
    """
    Check if we can use GPU (faster) or CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for faster processing!")
    else:
        device = torch.device("cpu")
        print("Using CPU (this might be slower)")
    
    return device


# ============================================
# PART 2: LOAD AND PREPARE IMAGES
# ============================================
def load_image(image_path, max_size=400):
    """
    Load an image and prepare it for neural network
    image_path: path to your image file
    max_size: maximum size to resize image (smaller = faster)
    """
    print(f"Loading image: {image_path}")
    
    # Open the image
    image = Image.open(image_path)
    
    # Resize image if too large
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    # Transform image to tensor
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Add batch dimension and return
    image = transform(image).unsqueeze(0)
    return image


def save_image(tensor, output_path):
    """
    Convert tensor back to image and save it
    """
    # Remove batch dimension
    image = tensor.clone().squeeze(0)
    
    # Unnormalize
    image = image.cpu().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = image.clip(0, 1)
    
    # Convert to PIL Image and save
    image = Image.fromarray((image * 255).astype(np.uint8))
    image.save(output_path)
    print(f"Styled image saved to: {output_path}")


def show_image(tensor, title="Image"):
    """
    Display an image from tensor
    """
    image = tensor.clone().squeeze(0)
    image = image.cpu().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = image.clip(0, 1)
    
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()


# ============================================
# PART 3: CALCULATE CONTENT LOSS
# ============================================
class ContentLoss(nn.Module):
    """
    Measures how different the content is from target
    We want to keep the content of our original photo
    """
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = 0
    
    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input


# ============================================
# PART 4: CALCULATE STYLE LOSS
# ============================================
def calculate_gram_matrix(input):
    """
    Gram matrix captures the style (textures, colors, patterns)
    """
    batch, channels, height, width = input.size()
    
    # Reshape to matrix
    features = input.view(batch * channels, height * width)
    
    # Calculate gram matrix
    gram = torch.mm(features, features.t())
    
    # Normalize
    return gram.div(batch * channels * height * width)


class StyleLoss(nn.Module):
    """
    Measures how different the style is from target
    We want to match the artistic style
    """
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = calculate_gram_matrix(target_feature).detach()
        self.loss = 0
    
    def forward(self, input):
        gram = calculate_gram_matrix(input)
        self.loss = nn.functional.mse_loss(gram, self.target)
        return input


# ============================================
# PART 5: BUILD THE NEURAL NETWORK MODEL
# ============================================
def build_style_transfer_model(content_img, style_img, device):
    """
    Build the neural network model for style transfer
    Uses pre-trained VGG19 network
    """
    print("Building neural network model...")
    
    # Load pre-trained VGG19 model
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    
    # Layers to extract content and style
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    
    # Lists to store losses
    content_losses = []
    style_losses = []
    
    # Build new model
    model = nn.Sequential()
    
    i = 0  # Counter for conv layers
    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue
        
        model.add_module(name, layer)
        
        # Add content loss
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)
        
        # Add style loss
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)
    
    # Trim layers after last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]
    
    return model, style_losses, content_losses


# ============================================
# PART 6: RUN STYLE TRANSFER
# ============================================
def run_style_transfer(content_img, style_img, input_img, device,
                       num_steps=300, style_weight=1000000, content_weight=1):
    """
    Main function that performs style transfer
    num_steps: how many iterations (more = better quality but slower)
    style_weight: how much to apply style
    content_weight: how much to preserve content
    """
    print("Starting style transfer process...")
    
    # Build model
    model, style_losses, content_losses = build_style_transfer_model(
        content_img, style_img, device
    )
    
    # We want to optimize the input image
    input_img.requires_grad_(True)
    model.requires_grad_(False)
    
    # Use LBFGS optimizer (works well for style transfer)
    optimizer = optim.LBFGS([input_img])
    
    print(f"Running optimization for {num_steps} steps...")
    
    step = [0]  # Counter
    
    while step[0] <= num_steps:
        
        def closure():
            # Correct the values of input image
            with torch.no_grad():
                input_img.clamp_(0, 1)
            
            optimizer.zero_grad()
            model(input_img)
            
            style_score = 0
            content_score = 0
            
            # Calculate losses
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            
            # Total loss
            style_score *= style_weight
            content_score *= content_weight
            total_loss = style_score + content_score
            total_loss.backward()
            
            step[0] += 1
            
            # Print progress every 50 steps
            if step[0] % 50 == 0:
                print(f"Step {step[0]}/{num_steps} - Style Loss: {style_score.item():.2f} - Content Loss: {content_score.item():.2f}")
            
            return total_loss
        
        optimizer.step(closure)
    
    # Final correction
    with torch.no_grad():
        input_img.clamp_(0, 1)
    
    return input_img


# ============================================
# PART 7: MAIN FUNCTION - RUNS EVERYTHING
# ============================================
def main():
    """
    Main function that brings everything together
    """
    print("=" * 60)
    print("WELCOME TO NEURAL STYLE TRANSFER")
    print("=" * 60)
    
    # Setup device
    device = setup_device()
    
    # Get image paths from user
    print("\n")
    content_path = input("Enter path to content image (your photo): ")
    style_path = input("Enter path to style image (artistic image): ")
    output_path = input("Enter output file name (e.g., styled_output.jpg): ")
    
    # Optional parameters
    print("\n--- Optional Settings (press Enter for defaults) ---")
    
    image_size_input = input("Image size (default 400, smaller=faster): ")
    image_size = int(image_size_input) if image_size_input else 400
    
    steps_input = input("Number of steps (default 300, more=better): ")
    num_steps = int(steps_input) if steps_input else 300
    
    style_weight_input = input("Style weight (default 1000000): ")
    style_weight = int(style_weight_input) if style_weight_input else 1000000
    
    # Load images
    print("\n" + "=" * 60)
    content_img = load_image(content_path, max_size=image_size).to(device)
    style_img = load_image(style_path, max_size=image_size).to(device)
    
    # Start with content image
    input_img = content_img.clone()
    
    # Run style transfer
    print("\n" + "=" * 60)
    output_img = run_style_transfer(
        content_img, style_img, input_img, device,
        num_steps=num_steps, style_weight=style_weight
    )
    
    # Save result
    print("\n" + "=" * 60)
    save_image(output_img, output_path)
    
    # Display images
    print("\nDisplaying results...")
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    show_image(content_img, "Content Image")
    
    plt.subplot(1, 3, 2)
    show_image(style_img, "Style Image")
    
    plt.subplot(1, 3, 3)
    show_image(output_img, "Styled Output")
    
    plt.tight_layout()
    plt.savefig("comparison.jpg")
    print("Comparison saved as: comparison.jpg")
    
    print("\n" + "=" * 60)
    print("Style transfer complete!")
    print("=" * 60)


# ============================================
# EXAMPLE FUNCTION - QUICK STYLE TRANSFER
# ============================================
def quick_style_transfer(content_path, style_path, output_path="styled_image.jpg"):
    """
    Quick function to apply style transfer with default settings
    Perfect for beginners!
    """
    device = setup_device()
    
    # Load images
    content_img = load_image(content_path, max_size=400).to(device)
    style_img = load_image(style_path, max_size=400).to(device)
    input_img = content_img.clone()
    
    # Run style transfer
    output_img = run_style_transfer(
        content_img, style_img, input_img, device,
        num_steps=300, style_weight=1000000, content_weight=1
    )
    
    # Save result
    save_image(output_img, output_path)
    
    return output_img


# ============================================
# RUN THE PROGRAM
# ============================================
if __name__ == "__main__":
    main()
    
    # Uncomment below for quick usage:
    # quick_style_transfer("my_photo.jpg", "van_gogh.jpg", "styled_output.jpg")