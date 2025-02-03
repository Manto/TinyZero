import lighteval as le
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Load a base LLM (not instruct-tuned)
model_name = "EleutherAI/gpt-neo-1.3B"  # Replace with your preferred base model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load the AIME_2024 dataset from Hugging Face
dataset = load_dataset("Maxwell-Jia/AIME_2024", split="test")

# Define a math evaluation task
class AIME2024Task(le.Task):
    def __init__(self):
        super().__init__(name="AIME_2024_math")

    def dataset(self):
        return [{"input": sample["problem"], "expected": str(sample["answer"])} for sample in dataset]

    def evaluate(self, sample, model_output):
        return sample["expected"].strip() == model_output.strip()

# Define a basic wrapper for non-instruct-tuned models
class BaseLLMEvaluator(le.Model):
    def __init__(self, model, tokenizer):
        super().__init__(name="base_llm")
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        output_ids = self.model.generate(input_ids, max_length=50)
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response[len(prompt):].strip()  # Extract generated text after prompt

# Run the evaluation
task = AIME2024Task()
model = BaseLLMEvaluator(model, tokenizer)
results = le.evaluate(model, [task])

# Print results
print(results)
