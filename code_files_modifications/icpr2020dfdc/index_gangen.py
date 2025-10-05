import sys
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, help='Source dir (ex: dataset/GANGen-Detection)', required=True)
    parser.add_argument('--output', type=Path, default='data/gangen_faces.pkl',
                        help='Path to save the faces DataFrame')
    return parser.parse_args(argv)

def main(argv):
    args = parse_args(argv)
    source_dir: Path = args.source
    output_path: Path = args.output

    if output_path.exists():
        print(f'Loading existing DataFrame: {output_path}')
        df = pd.read_pickle(output_path)
    else:
        print('Creating DataFrame from images...')

        data = []
        # Example: AttGAN/0_real, AttGAN/1_fake, etc.
        for folder in tqdm(sorted(source_dir.glob('*')), desc='Model folders'):
            if folder.is_dir():
                for class_folder in sorted(folder.glob('*')):
                    if class_folder.is_dir():
                        label = 0 if '0_real' in class_folder.name else 1
                        for img_path in class_folder.glob('*'):
                            if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                                data.append({
                                    'path': str(img_path.relative_to(source_dir)),
                                    'label': label,
                                    'folder': folder.name
                                })

        df = pd.DataFrame(data)
        print(f'Saving DataFrame with {len(df)} entries to {output_path}')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(output_path)

    print('Real:', (df['label'] == 0).sum())
    print('Fake:', (df['label'] == 1).sum())

if __name__ == '__main__':
    main(sys.argv[1:])
