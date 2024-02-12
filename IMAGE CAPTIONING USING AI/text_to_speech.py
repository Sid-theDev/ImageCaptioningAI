import cv2
import os
import time
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import pygame

cap = cv2.VideoCapture(0)

capture_interval = 5 

image_count = 0

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

output_folder = "captured_images"
os.makedirs(output_folder, exist_ok=True)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    image_filename = os.path.join(output_folder, f"captured_image_{image_count}.jpg")

    cv2.imwrite(image_filename, frame)
    print(f"Image saved as: {image_filename}")

    image_count += 1

    raw_image = Image.open(image_filename).convert('RGB')

    text = f"in this image"  
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    conditional_caption = processor.decode(out[0], skip_special_tokens=True)

    print(f"Image Caption: {conditional_caption}")

    tts = gTTS(conditional_caption, lang='en')
    audio_filename = os.path.join(output_folder, f"captured_audio_{image_count}.mp3")
    tts.save(audio_filename)
    print(f"Audio saved as: {audio_filename}\n")

    pygame.mixer.init()
    pygame.mixer.music.load(audio_filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)  

    time.sleep(capture_interval)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
