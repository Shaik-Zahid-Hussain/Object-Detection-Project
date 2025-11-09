import os
import cv2
import io
from PIL import Image

import ollama
from ollama import generate

def adjust_prompt(frame_width, frame_height):
    """
    Adjust the prompt size based on the frame size.
    Larger frames get a more detailed prompt; smaller frames get a simpler prompt.
    """
    frame_area = frame_width * frame_height
    
    # Set thresholds for different frame sizes
    if frame_area > 1000000:  # For large frames (e.g., 1080p)
        prompt = ("Describe in detail the types of vehicles in the bounded box, including the brand, color, their direction of movement, and any visible logos. Write only relevant things don't give any extra information.")
    elif frame_area > 500000:  # For medium-sized frames (e.g., 720p)
        prompt = ("Describe the types of vehicles in the bounded box, mention their brand, color, and direction of movement. Write only relevant things don't give any extra information.")
    else:  # For smaller frames (e.g., low resolution)
        prompt = "Identify the vehicles in the image and mention their color. Write only relevant things don't give any extra information."
    
    # If you want to further control the number of characters, truncate or expand the prompt.
    max_prompt_length = 200  # You can adjust this limit as needed
    if len(prompt) > max_prompt_length:
        prompt = prompt[:max_prompt_length]  # Truncate the prompt if it's too long
    
    return prompt

def preprocess_image(frame):
        # ollama.pull('llava')
        height, width, _ = frame.shape
        prompt = adjust_prompt(width, height)

        is_success, buffer = cv2.imencode(".jpg", frame)
        io_buf = io.BytesIO(buffer)
        
        full_res = ''
        for response in generate(model='llava', 
                             prompt=prompt,
                             images=[io_buf], 
                             stream=True):
        # Print the response to the console and add it to the full response
            print(response['response'], end='', flush=True)
            full_res += response['response']
            
        return full_res

# img = cv2.imread('images.jpg')
# print(preprocess_image(img))