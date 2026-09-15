"""Write the predeclared primary comparison set after training or unpacking."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='results/value_v2/models')
    p.add_argument('--v1',default='results/value/direct/best.pt')
    a=p.parse_args();root=Path(a.root)
    models={name:str(root/(name+'_17')/'best.pt') for name in ('prior','mlp','residual','no_vision','vision')}
    models['v1_fixed']=a.v1
    for path in models.values():
        if not Path(path).is_file():raise FileNotFoundError(path)
    (root/'online_models.json').write_text(json.dumps(models,indent=2))
    all_models={folder.name:str(folder/'best.pt') for folder in root.iterdir()
                if folder.is_dir() and (folder/'best.pt').is_file()}
    all_models['v1_fixed']=a.v1
    (root/'all_models.json').write_text(json.dumps(all_models,indent=2))


if __name__=='__main__':main()
