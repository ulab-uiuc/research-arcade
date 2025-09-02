# Preprpss the data so that we can use it directly as LLM's input

import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from tasks.paragraph_generation_local_vllm import