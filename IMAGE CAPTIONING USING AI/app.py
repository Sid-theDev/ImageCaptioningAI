import cv2
import os
import time
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import pygame

cap = cv2.VideoCapture(0)

capture_interval = 3  # Capture an image every 5 seconds

# Initializinvg a counter for image filenames
image_count = 0

# Initializing the Blip model and processor
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large").to("cuda")

image_output_folder = "captured_images"
audio_output_folder = "audio_output"
os.makedirs(image_output_folder, exist_ok=True)
os.makedirs(audio_output_folder, exist_ok=True)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Display the webcam feed
    cv2.imshow('Webcam Feed', frame)

    # Generate a unique filename for the captured image
    image_filename = os.path.join(
        image_output_folder, f"captured_image_{image_count}.jpg")

    # Save the captured image to the output folder
    cv2.imwrite(image_filename, frame)
    print(f"Image saved as: {image_filename}")

    # Increment the image count
    image_count += 1

    # Load the captured image
    raw_image = Image.open(image_filename).convert('RGB')

    # Conditional image captioning
    text = f"in this image"
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    conditional_caption = processor.decode(out[0], skip_special_tokens=True)

    print(f"Image Caption: {conditional_caption}")

    # Convert caption to audio using gTTS
    tts = gTTS(conditional_caption, lang='en')
    audio_filename = os.path.join(
        audio_output_folder, f"captured_audio_{image_count}.mp3")
    tts.save(audio_filename)
    print(f"Audio saved as: {audio_filename}\n")

    # Play the generated audio immediately
    pygame.mixer.init()
    pygame.mixer.music.load(audio_filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        # Wait for audio to finish playing before proceeding
        pygame.time.wait(100)

    time.sleep(capture_interval)

    # Check for a key press (press 'q' to exit)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
