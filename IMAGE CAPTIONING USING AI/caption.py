from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import time
from gtts import gTTS
import pygame

# Initialize the BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

# Folder containing input images
input_folder = "input_images"

# Output folder for audio files
output_audio_folder = "audio_output"
os.makedirs(output_audio_folder, exist_ok=True)

# Output file for captions
output_captions_file = "captions.txt"

# Initialize Pygame for audio playback
pygame.mixer.init()

# Function to play audio and wait for it to finish
def play_audio_wait(audio_file):
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pass

# Initialize a list to store captions
captions = []

# Process and generate captions for each image in the input folder
for image_filename in os.listdir(input_folder):
    if image_filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
        image_path = os.path.join(input_folder, image_filename)

        # Load the image
        raw_image = Image.open(image_path).convert('RGB')

        # Conditional image captioning
        text = "a photography of"  # Modify the prompt as needed
        inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        print(f"Processing Image: {image_filename}")
        print("Caption:", caption)

        # Add the caption to the list
        captions.append(caption)

        # Convert caption to audio using gTTS
        tts = gTTS(caption, lang='en')
        audio_file = os.path.join(output_audio_folder, f"{os.path.splitext(image_filename)[0]}.mp3")

        tts.save(audio_file)
        print(f"Audio saved as: {audio_file}")

        # Play the generated audio and wait for it to finish
        play_audio_wait(audio_file)

# Save the captions to a text file (notepad)
with open(output_captions_file, 'w') as file:
    for caption in captions:
        file.write(f"{caption}\n")

print(f"Captions saved to: {output_captions_file}")
