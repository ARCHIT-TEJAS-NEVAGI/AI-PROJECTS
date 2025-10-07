"""
GENERATIVE TEXT MODEL
A system that generates coherent text using AI models
Two methods: LSTM (train your own) and GPT-2 (pre-trained)
"""

# Import necessary libraries
import torch
import torch.nn as nn
import numpy as np
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PART 1: SIMPLE LSTM TEXT GENERATOR
# ============================================

class LSTMTextGenerator(nn.Module):
    """
    LSTM Neural Network for text generation
    LSTM = Long Short-Term Memory (remembers patterns in text)
    """
    def __init__(self, vocab_size, embedding_size=256, hidden_size=512, num_layers=2):
        super(LSTMTextGenerator, self).__init__()
        
        # Embedding layer: converts words to numbers
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        # LSTM layer: learns patterns in text
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.3)
        
        # Output layer: predicts next word
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    def forward(self, x, hidden):
        # Embed words
        embedded = self.embedding(x)
        
        # Pass through LSTM
        output, hidden = self.lstm(embedded, hidden)
        
        # Predict next word
        output = self.fc(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))


# ============================================
# PART 2: PREPARE TEXT DATA FOR TRAINING
# ============================================

class TextDataPreparator:
    """
    Prepares text data for training the LSTM model
    """
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
    
    def build_vocabulary(self, text):
        """
        Create a dictionary of all unique words
        """
        print("Building vocabulary...")
        
        # Split text into words
        words = text.lower().split()
        
        # Get unique words
        unique_words = sorted(set(words))
        
        # Create word-to-index mapping
        self.word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(unique_words)
        
        print(f"Vocabulary size: {self.vocab_size} unique words")
        
        return self.vocab_size
    
    def text_to_sequence(self, text):
        """
        Convert text to sequence of numbers
        """
        words = text.lower().split()
        sequence = [self.word_to_idx.get(word, 0) for word in words]
        return sequence
    
    def sequence_to_text(self, sequence):
        """
        Convert sequence of numbers back to text
        """
        words = [self.idx_to_word.get(idx, '<UNK>') for idx in sequence]
        return ' '.join(words)


# ============================================
# PART 3: TRAIN LSTM MODEL
# ============================================

def train_lstm_model(text, epochs=50, sequence_length=30):
    """
    Train the LSTM model on your text data
    epochs: how many times to train on the data
    sequence_length: how many words to look at once
    """
    print("=" * 60)
    print("TRAINING LSTM TEXT GENERATOR")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Prepare data
    preparator = TextDataPreparator()
    vocab_size = preparator.build_vocabulary(text)
    sequence = preparator.text_to_sequence(text)
    
    # Create training sequences
    sequences = []
    targets = []
    
    for i in range(0, len(sequence) - sequence_length):
        seq = sequence[i:i + sequence_length]
        target = sequence[i + 1:i + sequence_length + 1]
        sequences.append(seq)
        targets.append(target)
    
    print(f"Created {len(sequences)} training sequences\n")
    
    # Create model
    model = LSTMTextGenerator(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("Starting training...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        hidden = model.init_hidden(1, device)
        
        # Shuffle data
        combined = list(zip(sequences, targets))
        random.shuffle(combined)
        sequences_shuffled, targets_shuffled = zip(*combined)
        
        for seq, target in zip(sequences_shuffled, targets_shuffled):
            # Convert to tensors
            seq_tensor = torch.LongTensor([seq]).to(device)
            target_tensor = torch.LongTensor([target]).to(device)
            
            # Reset hidden state
            hidden = tuple([h.detach() for h in hidden])
            
            # Forward pass
            output, hidden = model(seq_tensor, hidden)
            loss = criterion(output.view(-1, vocab_size), target_tensor.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(sequences)
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
    
    print("\nTraining complete!")
    
    return model, preparator, device


# ============================================
# PART 4: GENERATE TEXT WITH LSTM
# ============================================

def generate_text_lstm(model, preparator, device, prompt, length=50, temperature=0.8):
    """
    Generate text using trained LSTM model
    prompt: starting text
    length: how many words to generate
    temperature: creativity (higher = more random, lower = more predictable)
    """
    model.eval()
    
    # Prepare prompt
    words = prompt.lower().split()
    sequence = [preparator.word_to_idx.get(word, 0) for word in words]
    
    # Initialize hidden state
    hidden = model.init_hidden(1, device)
    
    # Generate text
    generated = words.copy()
    
    with torch.no_grad():
        for _ in range(length):
            # Prepare input
            input_seq = torch.LongTensor([sequence[-30:]]).to(device)
            
            # Get prediction
            output, hidden = model(input_seq, hidden)
            
            # Apply temperature
            output = output[:, -1, :] / temperature
            probs = torch.softmax(output, dim=-1)
            
            # Sample next word
            next_word_idx = torch.multinomial(probs, 1).item()
            
            # Add to sequence
            sequence.append(next_word_idx)
            next_word = preparator.idx_to_word.get(next_word_idx, '<UNK>')
            generated.append(next_word)
    
    return ' '.join(generated)


# ============================================
# PART 5: GPT-2 TEXT GENERATOR (PRE-TRAINED)
# ============================================

class GPT2TextGenerator:
    """
    Uses pre-trained GPT-2 model (doesn't need training!)
    GPT-2 is trained on millions of web pages
    """
    def __init__(self, model_size='gpt2'):
        """
        model_size options: 'gpt2', 'gpt2-medium', 'gpt2-large'
        (larger = better but slower)
        """
        print("Loading GPT-2 model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_size)
        self.model = GPT2LMHeadModel.from_pretrained(model_size)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"GPT-2 model loaded on {self.device}")
    
    def generate(self, prompt, max_length=100, temperature=0.7, 
                 top_k=50, top_p=0.9, num_return=1):
        """
        Generate text using GPT-2
        prompt: starting text
        max_length: maximum words to generate
        temperature: creativity (0.7 = balanced)
        top_k: consider top k words
        top_p: nucleus sampling
        num_return: how many different versions to generate
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode generated text
        generated_texts = []
        for sequence in output:
            text = self.tokenizer.decode(sequence, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts


# ============================================
# PART 6: EXAMPLE TRAINING DATA
# ============================================

SAMPLE_TRAINING_TEXT = """
Artificial intelligence is transforming the world. Machine learning algorithms can now 
recognize patterns in data. Deep learning models are becoming more powerful every day. 
Neural networks learn from examples. Natural language processing helps computers understand 
human language. Computer vision allows machines to see and interpret images. AI is used in 
healthcare to diagnose diseases. Self-driving cars use AI to navigate roads. Chatbots 
provide customer service using AI. Recommendation systems suggest products based on preferences. 
AI assists in scientific research and discovery. Robots use AI to perform complex tasks. 
Speech recognition converts voice to text. AI helps in financial trading and analysis. 
Machine translation breaks down language barriers. AI is revolutionizing education with 
personalized learning. Climate models use AI to predict weather patterns. AI enhances 
cybersecurity by detecting threats. Virtual assistants make daily tasks easier. 
The future of AI holds endless possibilities for innovation and progress.
"""


# ============================================
# PART 7: INTERACTIVE TEXT GENERATION
# ============================================

def interactive_gpt2_mode():
    """
    Interactive mode for GPT-2 text generation
    """
    print("=" * 60)
    print("GPT-2 INTERACTIVE TEXT GENERATOR")
    print("=" * 60)
    
    # Initialize GPT-2
    generator = GPT2TextGenerator()
    
    print("\nGPT-2 is ready! Type 'quit' to exit.\n")
    
    while True:
        # Get prompt from user
        prompt = input("Enter your prompt: ")
        
        if prompt.lower() == 'quit':
            print("Thank you for using GPT-2 Text Generator!")
            break
        
        if not prompt.strip():
            print("Please enter a prompt!")
            continue
        
        # Get parameters
        print("\n--- Generation Settings ---")
        length = input("Max length (default 100): ")
        length = int(length) if length else 100
        
        temp = input("Temperature 0.1-2.0 (default 0.7): ")
        temp = float(temp) if temp else 0.7
        
        num = input("Number of versions (default 1): ")
        num = int(num) if num else 1
        
        # Generate text
        print("\n" + "=" * 60)
        print("GENERATING TEXT...")
        print("=" * 60 + "\n")
        
        results = generator.generate(
            prompt, 
            max_length=length,
            temperature=temp,
            num_return=num
        )
        
        # Display results
        for i, text in enumerate(results, 1):
            print(f"--- Version {i} ---")
            print(text)
            print()
        
        print("=" * 60 + "\n")


# ============================================
# PART 8: MAIN MENU
# ============================================

def main():
    """
    Main function with menu options
    """
    print("=" * 60)
    print("WELCOME TO GENERATIVE TEXT MODEL")
    print("=" * 60)
    
    print("\nChoose generation method:")
    print("1. GPT-2 (Pre-trained - No training needed)")
    print("2. LSTM (Train on custom text)")
    print("3. Quick GPT-2 Demo")
    
    choice = input("\nEnter your choice (1/2/3): ")
    
    if choice == "1":
        # GPT-2 Interactive Mode
        interactive_gpt2_mode()
    
    elif choice == "2":
        # LSTM Training and Generation
        print("\n" + "=" * 60)
        print("LSTM MODE - TRAIN ON CUSTOM TEXT")
        print("=" * 60)
        
        use_sample = input("\nUse sample text? (yes/no): ")
        
        if use_sample.lower() == "yes":
            training_text = SAMPLE_TRAINING_TEXT
        else:
            print("Enter your training text (paste and press Enter twice):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            training_text = " ".join(lines)
        
        # Train model
        model, preparator, device = train_lstm_model(training_text, epochs=50)
        
        # Generate text
        print("\n" + "=" * 60)
        prompt = input("Enter starting prompt: ")
        length = int(input("How many words to generate? (default 50): ") or "50")
        
        generated = generate_text_lstm(model, preparator, device, prompt, length)
        
        print("\n" + "=" * 60)
        print("GENERATED TEXT:")
        print("=" * 60)
        print(generated)
    
    elif choice == "3":
        # Quick GPT-2 Demo
        print("\n" + "=" * 60)
        print("QUICK GPT-2 DEMO")
        print("=" * 60)
        
        generator = GPT2TextGenerator()
        
        demo_prompts = [
            "The future of artificial intelligence is",
            "Once upon a time in a magical forest",
            "Climate change is affecting our planet by",
            "The benefits of exercise include"
        ]
        
        print("\nGenerating text for demo prompts...\n")
        
        for prompt in demo_prompts:
            print(f"Prompt: {prompt}")
            result = generator.generate(prompt, max_length=80)[0]
            print(f"Generated: {result}")
            print("-" * 60 + "\n")


# ============================================
# QUICK USAGE FUNCTIONS
# ============================================

def quick_generate(prompt, max_length=100):
    """
    Quick function to generate text (one line of code!)
    """
    generator = GPT2TextGenerator()
    return generator.generate(prompt, max_length=max_length)[0]


# ============================================
# EXAMPLE DEMONSTRATIONS
# ============================================

def run_examples():
    """
    Run example text generations
    """
    print("=" * 60)
    print("TEXT GENERATION EXAMPLES")
    print("=" * 60 + "\n")
    
    generator = GPT2TextGenerator()
    
    examples = [
        ("Story", "In a distant galaxy, a young hero discovered"),
        ("Tech", "Artificial intelligence will revolutionize"),
        ("Science", "The most fascinating thing about quantum physics is"),
        ("Business", "The key to successful entrepreneurship is")
    ]
    
    for category, prompt in examples:
        print(f"[{category}] Prompt: {prompt}")
        text = generator.generate(prompt, max_length=70)[0]
        print(f"Generated: {text}\n")
        print("-" * 60 + "\n")


# ============================================
# RUN THE PROGRAM
# ============================================

if __name__ == "__main__":
    main()
    
    # Uncomment for quick usage:
    # print(quick_generate("The future of technology is"))
    
    # Uncomment to run examples:
    # run_examples()