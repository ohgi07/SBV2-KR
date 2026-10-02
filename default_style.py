import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import numpy as np

from style_bert_vits2.constants import DEFAULT_STYLE
from style_bert_vits2.logging import logger


def collect_npy_files(
    wav_dir: Path, list_paths: Optional[Sequence[Union[Path, str]]]
) -> list[Path]:
    """
    Return the style vector files (`{wav}.npy`) to average.
    With list_paths (train.list / val.list), only the audio files in those lists are used, so that
    files left in wav_dir by earlier preprocessing runs (e.g. a style folder removed from raw/,
    since resampling does not clear wavs/) do not come back as styles or leak into the mean.
    """
    if list_paths is None:
        return list(wav_dir.rglob("*.npy"))
    files = []
    for list_path in list_paths:
        with open(list_path, encoding="utf-8") as f:
            files.extend(Path(f"{line.split('|')[0]}.npy") for line in f if line.strip())
    return files


def mean_vector(files: Sequence[Path]) -> np.ndarray:
    return np.mean(np.stack([np.load(file) for file in files]), axis=0)  # (256,)


def save_styles_by_dirs(
    wav_dir: Union[Path, str],
    output_dir: Union[Path, str],
    config_path: Union[Path, str],
    config_output_path: Union[Path, str],
    list_paths: Optional[Sequence[Union[Path, str]]] = None,
):
    wav_dir = Path(wav_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = collect_npy_files(wav_dir, list_paths)
    # Group by the first subdirectory under wav_dir (files directly under it belong to no style)
    root = wav_dir.resolve()
    styles: dict[str, list[Path]] = defaultdict(list)
    for file in npy_files:
        path = file.resolve()
        parts = path.relative_to(root).parts if path.is_relative_to(root) else ()
        if len(parts) > 1:
            styles[parts[0]].append(file)

    # Neutral is the mean of all
    names = [DEFAULT_STYLE]
    style_vectors = [mean_vector(npy_files)]
    if len(styles) in (0, 1):
        logger.info(
            f"At least 2 subdirectories are required for generating style vectors with respect to them, found {len(styles)}."
        )
        logger.info("Generating only neutral style vector instead.")
    else:
        for name in sorted(styles, key=lambda name: wav_dir / name):
            names.append(name)
            style_vectors.append(mean_vector(styles[name]))

    # Stack them to make (num_styles, 256)
    np.save(output_dir / "style_vectors.npy", np.stack(style_vectors, axis=0))
    logger.info(f"Saved style vectors to {output_dir / 'style_vectors.npy'}")

    # Save style2id config to json
    with open(config_path, encoding="utf-8") as f:
        json_dict = json.load(f)
    json_dict["data"]["num_styles"] = len(names)
    json_dict["data"]["style2id"] = {name: i for i, name in enumerate(names)}
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved style config to {config_output_path}")
