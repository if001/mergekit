from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./sample_output")
print(model)