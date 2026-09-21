from datetime import date
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = Path("models/SmolLM2-360M-Instruct")
PROMPT = "Give me a brief, creative idea for a new product or service. Return exactly one sentence."
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
MODEL = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)

def generate_idea() -> str:
	inputs = TOKENIZER.apply_chat_template(
		[{"role": "user", "content": PROMPT}],
		add_generation_prompt=True,
		return_tensors="pt",
		return_dict=True,
	)

	seed = int(date.today().strftime("%Y%m%d"))  # Seed based on the current date
	torch.manual_seed(seed)
	with torch.inference_mode():
		output = MODEL.generate(
			**inputs,
			max_new_tokens=56,
			do_sample=False,
			pad_token_id=TOKENIZER.eos_token_id,
		)
	new_tokens = output[0, inputs["input_ids"].shape[1] :]
	return TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()

if __name__ == "__main__":
	print(generate_idea())