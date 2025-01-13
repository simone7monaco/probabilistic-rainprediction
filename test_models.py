from training import main, get_args
from pathlib import Path
import shutil

if __name__ == "__main__":
    from utils import io
    import numpy as np
    import pandas as pd

    for model in ['unet', 'sde_unet', 'ensemble_unet', 'mcd_unet']:
        output_path = 'foo'
        print(f"""
              \n\n=========================================
              >> Testing model {model}\n""")
        args = get_args(['-m', model, '-e', '1', '-o', output_path, '--n_split', '1'])
        output_path = Path(output_path) / f'{args.network_model}' / f"split_{args.n_split}"
        assert not output_path.exists(), f"Output path {output_path} already exists"
        main(args)
        shutil.rmtree(output_path)