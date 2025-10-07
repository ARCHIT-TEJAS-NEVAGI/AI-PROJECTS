"""
SPEECH RECOGNITION SYSTEM
A system that converts speech (audio) to text using different methods
"""

# Import necessary libraries
import speech_recognition as sr
import os
from pydub import AudioSegment
import wave

# ============================================
# PART 1: CONVERT AUDIO FILES TO COMPATIBLE FORMAT
# ============================================
def convert_audio_to_wav(input_file):
    """
    Converts any audio file (mp3, m4a, etc.) to WAV format
    WAV format works best with speech recognition
    """
    print(f"Converting {input_file} to WAV format...")
    
    # Get file name without extension
    file_name = os.path.splitext(input_file)[0]
    output_file = file_name + "_converted.wav"
    
    # Load the audio file
    audio = AudioSegment.from_file(input_file)
    
    # Export as WAV
    audio.export(output_file, format="wav")
    
    print(f"Converted file saved as: {output_file}")
    return output_file


# ============================================
# PART 2: RECORD AUDIO FROM MICROPHONE
# ============================================
def record_audio_from_microphone(duration=5):
    """
    Records audio directly from your microphone
    duration: how many seconds to record
    """
    recognizer = sr.Recognizer()
    
    print(f"Recording for {duration} seconds...")
    print("Speak now!")
    
    # Use microphone as audio source
    with sr.Microphone() as source:
        # Adjust for background noise
        print("Adjusting for background noise... Please wait")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        # Record audio
        audio_data = recognizer.listen(source, timeout=duration)
        
    print("Recording complete!")
    return audio_data


# ============================================
# PART 3: SPEECH TO TEXT - GOOGLE (ONLINE)
# ============================================
def speech_to_text_google(audio_data):
    """
    Converts speech to text using Google's speech recognition
    Requires internet connection
    """
    recognizer = sr.Recognizer()
    
    print("Converting speech to text using Google...")
    
    try:
        # Use Google Speech Recognition
        text = recognizer.recognize_google(audio_data)
        return text
    
    except sr.UnknownValueError:
        return "Could not understand the audio"
    
    except sr.RequestError as e:
        return f"Error with Google service: {e}"


# ============================================
# PART 4: SPEECH TO TEXT - SPHINX (OFFLINE)
# ============================================
def speech_to_text_sphinx(audio_data):
    """
    Converts speech to text using Sphinx (works offline)
    No internet required but less accurate than Google
    """
    recognizer = sr.Recognizer()
    
    print("Converting speech to text using Sphinx (offline)...")
    
    try:
        # Use Sphinx Speech Recognition (offline)
        text = recognizer.recognize_sphinx(audio_data)
        return text
    
    except sr.UnknownValueError:
        return "Could not understand the audio"
    
    except sr.RequestError as e:
        return f"Error with Sphinx: {e}"


# ============================================
# PART 5: PROCESS AUDIO FILE
# ============================================
def process_audio_file(file_path, recognition_method="google"):
    """
    Reads an audio file and converts it to text
    file_path: path to your audio file
    recognition_method: 'google' or 'sphinx'
    """
    recognizer = sr.Recognizer()
    
    print(f"Processing audio file: {file_path}")
    
    # Check if file needs conversion
    if not file_path.endswith('.wav'):
        file_path = convert_audio_to_wav(file_path)
    
    # Load the audio file
    with sr.AudioFile(file_path) as source:
        # Read the audio data
        audio_data = recognizer.record(source)
    
    # Convert to text based on method
    if recognition_method == "google":
        text = speech_to_text_google(audio_data)
    else:
        text = speech_to_text_sphinx(audio_data)
    
    return text


# ============================================
# PART 6: GET AUDIO FILE INFO
# ============================================
def get_audio_info(file_path):
    """
    Shows information about the audio file
    """
    print("\n" + "=" * 60)
    print("AUDIO FILE INFORMATION")
    print("=" * 60)
    
    try:
        with wave.open(file_path, 'rb') as audio_file:
            # Get audio properties
            channels = audio_file.getnchannels()
            sample_width = audio_file.getsampwidth()
            frame_rate = audio_file.getframerate()
            frames = audio_file.getnframes()
            duration = frames / float(frame_rate)
            
            print(f"File: {file_path}")
            print(f"Channels: {channels}")
            print(f"Sample Width: {sample_width} bytes")
            print(f"Frame Rate: {frame_rate} Hz")
            print(f"Duration: {duration:.2f} seconds")
            print("=" * 60 + "\n")
    
    except Exception as e:
        print(f"Could not read audio info: {e}")


# ============================================
# PART 7: SAVE TRANSCRIPTION TO FILE
# ============================================
def save_transcription(text, output_file="transcription.txt"):
    """
    Saves the transcribed text to a file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("TRANSCRIPTION RESULT\n")
        f.write("=" * 60 + "\n\n")
        f.write(text)
    
    print(f"Transcription saved to: {output_file}")


# ============================================
# PART 8: MAIN FUNCTION - RUNS EVERYTHING
# ============================================
def main():
    """
    Main function that runs the speech recognition system
    """
    print("=" * 60)
    print("WELCOME TO SPEECH RECOGNITION SYSTEM")
    print("=" * 60)
    
    # Ask user for input method
    print("\nChoose input method:")
    print("1. Record from microphone")
    print("2. Upload audio file (WAV, MP3, etc.)")
    
    choice = input("\nEnter your choice (1/2): ")
    
    # Choose recognition method
    print("\nChoose recognition method:")
    print("1. Google (online - more accurate)")
    print("2. Sphinx (offline - works without internet)")
    
    method_choice = input("\nEnter your choice (1/2): ")
    recognition_method = "google" if method_choice == "1" else "sphinx"
    
    # Process based on user choice
    if choice == "1":
        # Record from microphone
        duration = int(input("How many seconds to record? (e.g., 5): "))
        audio_data = record_audio_from_microphone(duration)
        
        # Convert to text
        if recognition_method == "google":
            transcribed_text = speech_to_text_google(audio_data)
        else:
            transcribed_text = speech_to_text_sphinx(audio_data)
    
    elif choice == "2":
        # Process audio file
        file_path = input("Enter audio file path: ")
        
        # Show audio info if it's a WAV file
        if file_path.endswith('.wav'):
            get_audio_info(file_path)
        
        # Process the file
        transcribed_text = process_audio_file(file_path, recognition_method)
    
    else:
        print("Invalid choice!")
        return
    
    # Display results
    print("\n" + "=" * 60)
    print("TRANSCRIPTION RESULT")
    print("=" * 60)
    print(transcribed_text)
    print("=" * 60)
    
    # Ask if user wants to save
    save_option = input("\nDo you want to save transcription? (yes/no): ")
    if save_option.lower() == "yes":
        output_file = input("Enter output file name (e.g., transcript.txt): ")
        save_transcription(transcribed_text, output_file)
    
    print("\nThank you for using Speech Recognition System!")


# ============================================
# EXAMPLE USAGE FUNCTIONS
# ============================================
def example_transcribe_file():
    """
    Example: Transcribe an audio file
    """
    file_path = "sample_audio.wav"
    text = process_audio_file(file_path, recognition_method="google")
    print(f"Transcription: {text}")


def example_record_and_transcribe():
    """
    Example: Record audio and transcribe
    """
    audio = record_audio_from_microphone(duration=5)
    text = speech_to_text_google(audio)
    print(f"You said: {text}")


# ============================================
# RUN THE PROGRAM
# ============================================
if __name__ == "__main__":
    main()
    
    # Uncomment below to run examples directly:
    # example_transcribe_file()
    # example_record_and_transcribe()