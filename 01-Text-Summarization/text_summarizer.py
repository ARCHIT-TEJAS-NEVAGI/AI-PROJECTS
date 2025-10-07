"""
TEXT SUMMARIZATION TOOL
A simple tool to summarize lengthy articles from PDF, Word files, or direct text
"""

# Import necessary libraries
import PyPDF2
import docx
from transformers import pipeline

# ============================================
# PART 1: READ PDF FILES
# ============================================
def read_pdf_file(file_path):
    """
    This function reads text from a PDF file
    """
    print(f"Reading PDF file: {file_path}")
    
    # Open the PDF file
    pdf_file = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    # Store all text here
    all_text = ""
    
    # Go through each page
    for page in pdf_reader.pages:
        all_text += page.extract_text()
    
    pdf_file.close()
    return all_text


# ============================================
# PART 2: READ WORD FILES
# ============================================
def read_word_file(file_path):
    """
    This function reads text from a Word document
    """
    print(f"Reading Word file: {file_path}")
    
    # Open the Word document
    doc = docx.Document(file_path)
    
    # Store all text here
    all_text = ""
    
    # Go through each paragraph
    for paragraph in doc.paragraphs:
        all_text += paragraph.text + "\n"
    
    return all_text


# ============================================
# PART 3: SUMMARIZE TEXT
# ============================================
def summarize_text(text, max_summary_length=150, min_summary_length=50):
    """
    This function takes text and creates a summary
    max_summary_length: maximum words in summary
    min_summary_length: minimum words in summary
    """
    print("Creating summary...")
    
    # Create the summarizer (using Facebook's BART model)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Split text into smaller chunks if it's too long (model limit is 1024 tokens)
    max_chunk_length = 1000
    
    if len(text.split()) > max_chunk_length:
        # Split into sentences
        sentences = text.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk.split()) + len(sentence.split()) < max_chunk_length:
                current_chunk += sentence + "."
            else:
                chunks.append(current_chunk)
                current_chunk = sentence + "."
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # Summarize each chunk
        all_summaries = []
        for chunk in chunks:
            if len(chunk.strip()) > 0:
                summary = summarizer(chunk, max_length=max_summary_length, 
                                   min_length=min_summary_length, do_sample=False)
                all_summaries.append(summary[0]['summary_text'])
        
        # Combine all summaries
        final_summary = " ".join(all_summaries)
        return final_summary
    else:
        # Text is short enough to summarize directly
        summary = summarizer(text, max_length=max_summary_length, 
                           min_length=min_summary_length, do_sample=False)
        return summary[0]['summary_text']


# ============================================
# PART 4: MAIN FUNCTION - RUNS EVERYTHING
# ============================================
def main():
    """
    Main function that brings everything together
    """
    print("=" * 60)
    print("WELCOME TO TEXT SUMMARIZATION TOOL")
    print("=" * 60)
    
    # Ask user what type of input they want
    print("\nChoose input type:")
    print("1. PDF file")
    print("2. Word file (.docx)")
    print("3. Type or paste text directly")
    
    choice = input("\nEnter your choice (1/2/3): ")
    
    # Get the text based on user choice
    if choice == "1":
        file_path = input("Enter PDF file path: ")
        text = read_pdf_file(file_path)
    
    elif choice == "2":
        file_path = input("Enter Word file path: ")
        text = read_word_file(file_path)
    
    elif choice == "3":
        print("Enter or paste your text (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        text = "\n".join(lines)
    
    else:
        print("Invalid choice!")
        return
    
    # Show the original text
    print("\n" + "=" * 60)
    print("ORIGINAL TEXT:")
    print("=" * 60)
    print(text[:500] + "..." if len(text) > 500 else text)
    print(f"\nTotal words: {len(text.split())}")
    
    # Create summary
    print("\n" + "=" * 60)
    print("CREATING SUMMARY...")
    print("=" * 60)
    
    summary = summarize_text(text)
    
    # Show the summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(summary)
    print(f"\nSummary words: {len(summary.split())}")
    
    # Save summary to file
    save_option = input("\nDo you want to save summary to file? (yes/no): ")
    if save_option.lower() == "yes":
        output_file = input("Enter output file name (e.g., summary.txt): ")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("ORIGINAL TEXT:\n")
            f.write("=" * 60 + "\n")
            f.write(text + "\n\n")
            f.write("SUMMARY:\n")
            f.write("=" * 60 + "\n")
            f.write(summary)
        print(f"Summary saved to {output_file}")
    
    print("\nThank you for using Text Summarization Tool!")


# ============================================
# RUN THE PROGRAM
# ============================================
if __name__ == "__main__":
    main()