"""Step 0 - build data/<dataset>/{features,groundTruth,mapping} from the public releases (downloaded if missing).

    python scripts/prepare_data.py --datasets FSeval,FS,desktop_assembly,Breakfast,YTI --raw-root /path/to/downloads
"""
import argparse

import _path  # noqa: F401
from pa_decode import config as cfg
from pa_decode.data import prepare_datasets


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--datasets', default=','.join(cfg.ALL_DATASETS))
    p.add_argument('--raw-root', action='append', default=[],
                   help='folder(s) that already contain the raw datasets (searched recursively); can be repeated')
    p.add_argument('--data-dir', default='data', help='output folder in the layout expected by the ASOT loader')
    p.add_argument('--download-dir', default='data_raw', help='where missing datasets are downloaded')
    p.add_argument('--no-download', action='store_true')
    a = p.parse_args()
    ready = prepare_datasets(a.datasets.split(','), a.raw_root, a.data_dir, a.download_dir, allow_download=not a.no_download)
    print('Ready:', ready)


if __name__ == '__main__':
    main()
