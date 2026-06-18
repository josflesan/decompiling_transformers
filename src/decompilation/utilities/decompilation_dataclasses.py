from dataclasses import dataclass

from primitives_att.utilities.att_primitive_dataclasses import AttPrimitivesConfig
from primitives_mlp.utilities.mlp_primitive_dataclasses import MLPPrimitivesConfig
from pruning.utilities.pruning_dataclasses import PruningRunConfig
from rasp.utilities.rasp_dataclasses import RaspRunConfig

@dataclass
class DecompilationRunConfig:
    pruning: PruningRunConfig
    mlp_primitives: MLPPrimitivesConfig
    att_primitives: AttPrimitivesConfig
    rasp: RaspRunConfig
    run_pruning: bool = True
    run_mlp_primitives: bool = True
    run_att_primitives: bool = True
    run_conversion: bool = True
