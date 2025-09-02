import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.paragraph_generation_local_vllm import visual_adaptation

image_paths = ["./Konigsberg2_page1.jpg", "./something_that_does_not_exist.jpg", "./hello_world.txt", "Konigsberg2.pdf"]

generated_description = visual_adaptation(image_paths = image_paths)

print(generated_description)
