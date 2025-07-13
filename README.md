This folder contains the code for finetuning a SmolLM for a given task using LoRA method.
Given a query for scheduling a meeting specifying multiple entities, the LLM is fine tuned to extract those entities in a JSON format.

# Environment Setup
Recommended Python version : 3.11.13
The fine tuning was done on Google Colab notebook, so the package installation requirements are mentioned in the first cell.

However the requirements for local inference and deployments is given in `requirements.txt`.

Post download the checkpoint folder, place it inside the `lora_weights` folder

# Dataset Creation
The code for creating the dataset is in `create_dataset.ipynb`

# Finetuning
The code for finetuning can be found in `finetune_smollm.ipynb`

# Validation
## Base model
A standard prompt was tested using the base model, however it couldn't generate the responses even in JSON format.
The code for testing out the base model can be found in `base_model_evaludation.ipynb`.

## Fine tuned model evaluation

Since the task is to extract the given entities from a json format, a string comparison is used as an evaluation metric.

Accuracy is calculated across the validation dataset on a strict string match bases.
Similiarly precision and recall are also calculated.

| Model Name                 | Training Technique   | Precision | Recall   | Accuracy |
|---------------------------|----------------------|-----------|----------|----------|
| HuggingFaceTB/SmolLM-360M | Base Model           | 0.00 %    | 0.00 %   | 0.00 %   |
| HuggingFaceTB/SmolLM-360M | PEFT (LoRA)          | 96.00 %   | 95.47 %  | 73.58 %  |


However a key level accuracy is also computed across the validation dataset.

| Field      | Accuracy (%)         |
|------------|----------------------|
| action     | 91.19                |
| attendees  | 97.48                |
| date       | 93.71                |
| duration   | 96.86                |
| location   | 94.34                |
| notes      | 99.37                |
| recurrence | 97.48                |
| time       | 96.86                |



# Deployment
The fine tuned model is deployed using fast api. This is a local deployment.
However with more compute and Nginx used for port forwarding, the app can be exposed to the internet.

Navigate to the cwd with the environment activated in a terminal. Deploy the app by using the following command
```
uvicorn main:app --reload
```

To test the deployment, open another terminal and navigate to the cwd and run the following command.
Please feel free to change the user query for more experiments.
```
python3 api_inference.py
```
