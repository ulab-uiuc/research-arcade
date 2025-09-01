import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.paragraph_generation_local_vllm import visual_adaptation

image_paths = ["./Konigsberg2_page1.jpg", "./nekopen.png"]

descriptions = visual_adaptation(image_paths)


print("-----")
print(descriptions)
