import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.generated_paragraph_evaluation import answer_evaluation

generated_answers = ["Hi"]
original_answers = ["Hello"]

res = answer_evaluation(generated_answers=generated_answers, original_answers=original_answers)

print(res)