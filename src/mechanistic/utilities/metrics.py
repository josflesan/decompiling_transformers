import torch
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float, Int

def get_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Int[Tensor, "batch 2"],
    per_prompt: bool = False,
) -> Float[Tensor, "*batch"]:
    """
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    """
    
    prediction = logits[:, -1, :].squeeze(1)
    logitCorrect = prediction[torch.arange(prediction.size(0)), answer_tokens[:, 0]]
    logitIncorrect = prediction[torch.arange(prediction.size(0)), answer_tokens[:, 1]]
    
    logitDiff = logitCorrect - logitIncorrect
    
    if per_prompt:
        return logitDiff
    
    return torch.mean(logitDiff, axis=0)

def logit_diff_metric(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Int[Tensor, "batch 2"],
    corrupted_logit_diff: float,
    clean_logit_diff: float,
) -> Float[Tensor, "..."]:
    """
    Linear function of logit diff, calibrated so that it equals 0 when performance is same as on
    corrupted input, and 1 when performance is same as on clean input. This is used for activation
    and path patching.
    """
    
    patched_logit_diff = get_logit_diff(logits, answer_tokens)
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

def ablation_metric(ablated_logits, clean_logits, answer_tokens, **kwargs) -> Float[Tensor, "..."]:
    """
    Linear function of logit diff, calibrated so that it equals 0 when performance is completely
    degraded and 1 when performance is relatively unaffected. This is used for ablation experiments.
    
    Negative results suggest the component was critical for performance, causing the model to prefer
    the wrong token when ablated.
    """
    ablated_logit_diff = get_logit_diff(ablated_logits, answer_tokens)
    clean_logit_diff = get_logit_diff(clean_logits, answer_tokens)
    
    return ablated_logit_diff / clean_logit_diff


# def logsoftmax(logits: torch.Tensor, target_ids: torch.Tensor, position: int) -> float:
#     """Log probability of target token at position"""
#     log_probs = F.log_softmax(logits[:, position, :], dim=-1)
#     return log_probs.gather(1, target_ids.unsqueeze(1)).mean().item()

# def softmax(logits: torch.Tensor, target_ids: torch.Tensor, position: int) -> float:
#     """Probability of target token at position (0-1 scale)"""
#     probs = F.softmax(logits[:, position, :], dim=-1)
#     return probs.gather(1, target_ids.unsqueeze(1)).mean().item()
