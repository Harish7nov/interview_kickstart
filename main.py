from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import json
from pydantic import BaseModel
import torch

# load the base model tokenizer
checkpoint = "HuggingFaceTB/SmolLM-360M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# config for smollm 360m
# set the pad token as eos token so that the finetuned model knows when to stop generating tokens
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# load the base model and attach the adapter to it
model_path = "lora_weights/checkpoint-500"
peft_model = AutoPeftModelForCausalLM.from_pretrained(model_path, device_map="cpu", torch_dtype=torch.float16)

print(f"Total No of params (Model + Adapter) : {peft_model.num_parameters() / 1e6} M Params")

# Create FastAPI app
app = FastAPI()

# Request schema
class PromptRequest(BaseModel):
    query: str
    max_new_tokens: int = 100

# Inference endpoint
@app.post("/generate")
def generate_text(request: PromptRequest):
    try:
        input_text = f"<user>{request.query}</user><output>"
        inputs = tokenizer(input_text, return_tensors="pt")

        output_tokens = peft_model.generate(**inputs, max_new_tokens=request.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)

        truncated_output = tokenizer.decode(output_tokens[0][len(inputs[0]):])
        output = tokenizer.decode(output_tokens[0])

        resp = output.split("</output>")[0].split("<output>")[1]

        pred = json.loads(resp)

        return {"response": pred}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
