import os
import warnings
from transformer_lens.model_bridge import TransformerBridge

from data.CountDataset import CountCorruption
from mechanistic.core.attribution import (
    residual_stream_attribution,
    layerwise_attribution,
    attention_head_attribution
)
from tasks.registry import get_task
from utilities.core import TaskConfig

os.environ['TRANSFORMERLENS_ALLOW_MPS'] = '1'
warnings.filterwarnings('ignore')

EXP_NAME = 'prune_test'
MODEL_PATH = "models/count-%2l4h256d4lr01drop"

def main():
    # Test function to verify that the functions in core are working
    model = TransformerBridge.boot_transformers(
        MODEL_PATH,
        device='mps',
    )
    model.eval()
    
    # Get tokenizer and corrupted dataset
    config = TaskConfig(
        name='counting',
        train_length_range=[50, 150],
        val_length_range=[50, 150],
        max_test_length=150
    )
    task = get_task('counting', config)
    tokenizer, dataset = task.build()
    dataset = dataset['train']

    corrupted_data = dataset.get_corrupted(
        CountCorruption.CHANGE_START,
        batch_size=128
    )
    
    # Run attribution experiments
    residual_stream_attribution(
        model=model,
        tokens=corrupted_data.clean_tokens,
        position_ids=corrupted_data.clean_pos,
        answer_tokens=corrupted_data.answer_tokens[:, 0],
        exp_name=EXP_NAME
    )
    
    layerwise_attribution(
        model=model,
        tokens=corrupted_data.clean_tokens,
        position_ids=corrupted_data.clean_pos,
        answer_tokens=corrupted_data.answer_tokens[:, 0],
        exp_name=EXP_NAME
    )
    
    attention_head_attribution(
        model=model,
        tokens=corrupted_data.clean_tokens,
        position_ids=corrupted_data.clean_pos,
        answer_tokens=corrupted_data.answer_tokens[:, 0],
        exp_name=EXP_NAME
    )


if __name__ == "__main__":
    main()