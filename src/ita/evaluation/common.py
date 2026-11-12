import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


SYSTEM_PROMPT = """
        
            Please provide a set of short arguments supporting and attacking the following claim. 
            Construct the arguments so they refer to the truthfulness of the claim. 
            The arguments should be short and concise, ideally one sentence per argument.
            The arguments should be based on factual information and logical reasoning.
            Only provide supporting/attacking arguments if you think there is a valid and convincing support/attack for this claim (there is a non-zero probability that the claim is true/false), otherwise return: N/A.
            Provide as many uniqute supporting/attacking arguments as you think are correct and applicable to the claim.
            
            
            Respond in the following format:
            
                claim: 
                    "<CLAIM>".
                Output:
                    {
                        'support': [
                            "<SUPPORT ARGUMENT 1>",
                            "<SUPPORT ARGUMENT 2>",
                            ...,
                        ],
                        'attack': [
                            "<ATTACK ARGUMENT 1>",
                            "<ATTACK ARGUMENT 2>",
                            ...,
                        ]
                    }
        """

PROMPT_TEMPLATE = {
    "qwen": lambda user_input: f"<|im_start|>system\nYou are a helpful assistant. {SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
}


@dataclass
class GeneratedSample:
    idx: int
    claim: str
    label: str
    arguments: Optional[Dict[str, List[str]]]
    is_parse_error: bool = False
    error_message: Optional[str] = None
    argument_scores: Optional[Dict[str, List[float]]] = None  # {"support": [...], "attack": [...]}


@dataclass
class GenerationMetadata:
    model_id: str
    eval_data_paths: List[str]
    timestamp: str
    total_samples: int
    parse_errors: int
    parse_error_rate: float
    claims_with_retries: int = 0
    total_retries: int = 0
    avg_retries: float = 0.0


@dataclass
class GenerationOutput:
    metadata: GenerationMetadata
    samples: List[GeneratedSample]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": asdict(self.metadata),
            "samples": [asdict(s) for s in self.samples],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationOutput":
        metadata = GenerationMetadata(**data["metadata"])
        samples = [GeneratedSample(**s) for s in data["samples"]]
        return cls(metadata=metadata, samples=samples)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GenerationOutput":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
