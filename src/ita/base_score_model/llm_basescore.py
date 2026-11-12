import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

model = None
tokenizer = None

# Special token ID for </think>
THINK_END_TOKEN_ID = 151668

_bs_device = "cuda:0"

def load_model(device: str = None):
    """Load the base-score LLM with standard HuggingFace transformers."""
    global model, tokenizer, _bs_device
    target_device = device or _bs_device
    if model is None or tokenizer is None:
        model_name = "Qwen/Qwen3-8B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": target_device},
        )
        model.eval()
        print(f"[LLM BS] Loaded {model_name} on {target_device}")


def generate_qwen_response(
    prompt: str, max_new_tokens: int = 512, device: str | None = None
):
    load_model(device=device)
    messages = [{"role": "user", "content": prompt}]

    # Format input using chat template
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # Extract thinking content
    try:
        index = len(output_ids) - output_ids[::-1].index(THINK_END_TOKEN_ID)
    except ValueError:
        index = 0  # </think> token not found

    thinking_content = tokenizer.decode(
        output_ids[:index], skip_special_tokens=True
    ).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    return {"thinking_content": thinking_content, "content": content}


def analyst(statement, claim=None, support=False, verbal=False, topic=False, **_):
    if not topic and claim is None:
        raise ValueError(
            "Claim is required for the analyst prompt without topic flag, but was None"
        )

    if verbal:

        def formatter(output):
            likelihood = output.replace("Confidence in argument:", "").strip()
            likelihood_dict = {
                "fully confident": 0.95,
                "highly confident": 0.8,
                "quite confident": 0.65,
                "moderately confident": 0.5,
                "slightly confident": 0.35,
                "not very confident": 0.2,
                "not confident at all": 0.05,
            }
            return likelihood_dict[likelihood]

        relation = "supports" if support else "refutes"
        options = [
            "fully confident",
            "highly confident",
            "quite confident",
            "moderately confident",
            "slightly confident",
            "not very confident",
            "not confident at all",
        ]
        q = '"'
        constraints = {
            "constraint_prefix": "Confidence in argument:",
            "constraint_options": options,
            "constraint_end_after_options": True,
        }

        if topic:
            instructions = (
                f"You are an analyst evaluating the validity of statements. "
                f'For the statement:\n\nStatement: "{statement}"\n\nplease give your confidence '
                f"that the statement is correct, accurate and truthful. "
            )
        else:
            instructions = (
                f"You are an analyst evaluating the validity and relevance of arguments. "
                f'For the argument:\n\nArgument: "{statement}"\n\nplease give your confidence '
                f"that the argument presents a compelling case {'in favour of' if support else 'against'} "
                f'the statement:\n\nStatement: "{claim}"\n\nYour assessment should be based '
                f"on how well the argument {'supports' if support else 'refutes'} the considered "
                "statement as well as the correctness, accuracy and truthfulness of the given argument. "
            )

        return (
            instructions
            + (
                f"Your response should be chosen out of the options: "
                f"{', '.join([q + o + q for o in options])}. "
                "Please respond in the following form:"
                f"\n\nConfidence in {'argument' if not topic else 'statement'}: "
                f"Your confidence in the {'argument' if not topic else 'statement'} validity"
            ),
            constraints,
            formatter,
        )

    def formatter(output):
        print(f"------------------------Raw output from LLM: {output}")
        likelihood = output.replace("Likelihood:", "").replace("%", "").strip()
        likelihood = likelihood.split("\n")[0]
        print(f"---------------------Likelihood: {likelihood} ---------------------")
        return int(likelihood) / 100

    constraints = {
        "constraint_prefix": "Likelihood:",
        "constraint_options": [f" {l}%" for l in range(0, 101)],
        "constraint_end_after_options": True,
    }

    if topic:
        instructions = (
            "You are an analyst evaluating the validity of statements. "
            f'For the statement:\n\nStatement: "{statement}"\n\nplease give your confidence '
            f"that the statement is correct, accurate and truthful. "
            f"Your response should be between 0% and 100% with 0% indicating that the "
            f"considered statement is definitely invalid, 100% indicating that the considered statement is "
        )
    else:
        instructions = (
            "You are an analyst evaluating the validity and relevance of arguments. "
            f'For the argument:\n\nArgument: "{statement}"\n\nplease give your confidence '
            f"that the argument presents a compelling case {'in favour of' if support else 'against'} "
            f'the statement:\n\nStatement: "{claim}"\n\nYour assessment should be based '
            f"on how well the argument {'supports' if support else 'refutes'} the considered "
            "statement as well as the correctness, accuracy and truthfulness of the given argument. "
            f"Your response should be between 0% and 100% with 0% indicating that the "
            f"considered argument is definitely invalid, 100% indicating that the considered argument is "
        )

    return (
        instructions
        + (
            "definitely valid and values in between indicating various levels of "
            "uncertainty. Your estimates should be well-calibrated, so feel free to "
            "err on the side of caution and output moderate probabilities if you are "
            "not completely sure in your assessment. "
            "Please respond in the following form:"
            "\n\nLikelihood: The predicted likelihood that the considered "
            f"{'argument' if not topic else 'statement'} is valid"
        ),
        constraints,
        formatter,
    )


def base_score_func(claim, statement, type, device: str | None = None):
    prompt, constraints, formatter = analyst(
        statement, claim=claim, support=type, verbal=False, topic=False
    )
    llm_response = generate_qwen_response(
        "/no_think" + prompt + "/no_think", device=device
    )
    print(f"------------------------Raw output from LLM: {llm_response['content']}")
    # while True:
    #     llm_response = generate_qwen_response("/no_think"+prompt+"/no_think")
    #     content = llm_response["content"].strip()

    #     # Define the required pattern: starts with "Likelihood: ",
    #     # followed by one or more digits, and ends with "%"
    #     pattern = r"^Likelihood: \d+%$"

    #     # Check 1: Ensure there are no "<think>" tags
    #     # Check 2: Ensure the content matches the required pattern
    #     if "<think>" not in content and re.match(pattern, content):
    #         break # If both checks pass, exit the loop
    return formatter(llm_response["content"])
