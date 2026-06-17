from dataclasses import dataclass

from primitives_att.utilities.att_primitive_dataclasses import AttPrimitivesConfig
from primitives_mlp.utilities.mlp_primitive_dataclasses import MLPPrimitivesConfig
from pruning.utilities.pruning_dataclasses import PruningRunConfig


@dataclass
class DecompilationRunConfig:
    pruning: PruningRunConfig
    mlp_primitives: MLPPrimitivesConfig
    att_primitives: AttPrimitivesConfig
    run_pruning: bool = True
    run_mlp_primitives: bool = True
    run_att_primitives: bool = True
