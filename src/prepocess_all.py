from clean_data import clean
from pathlib import Path
import shutil


if __name__ == '__main__':
    p = Path(__file__).parent / Path('../data')
    for x in p.rglob('*'):
        for file in x.glob('*.csv'):
            # prepare output directory
            outPath = Path(str(file.parent).replace('data', 'data_clean'))
            if not outPath.exists():
                outPath.mkdir(parents=True)

            if not 'annotations' in str(file):
                # clean and save data
                df = clean(file)
                df.to_csv(outPath / file.name, index=False)
            else:
                # copy annotation
                shutil.copy(file, outPath / file.name)

