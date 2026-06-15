from pathlib import Path

OUTPUT_DIRECTORY = 'src/out'

def get_output_path(exp_name: str, subfolder: str, artifact_subdir: str | None = None) -> Path:
    exp_path = Path(OUTPUT_DIRECTORY) / exp_name
    
    if not exp_path.exists():
        raise RuntimeError(f"The experiment {exp_name} does not exist")

    # If the experiment exists, create subfolders as required
    full_path = exp_path / "mechanistic" / subfolder
    full_path.mkdir(exist_ok=True, parents=True)
    
    if artifact_subdir:
        full_path = full_path / artifact_subdir
        full_path.mkdir(exist_ok=True, parents=True)
    
    return full_path
