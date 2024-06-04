from pathlib import Path

import hydra

from src.utils import chngdir, remove_dir


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg):
    chngdir()
    images_dir = Path(cfg.folders.raw_images)
    class_paths = list(images_dir.iterdir())

    remove_dir(cfg.no_fold.root.train)
    remove_dir(cfg.no_fold.root.val)
    remove_dir(cfg.no_fold.root.test)


if __name__ == "__main__":
    main()
