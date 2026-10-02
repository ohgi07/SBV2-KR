import glob
import os
import re
from pathlib import Path
from typing import Any, Optional, Union

import torch

from style_bert_vits2.logging import logger
from style_bert_vits2.nlp.korean.warm_start import warm_start_new_rows


def expand_embedding_if_needed(
    key: str, saved_tensor: torch.Tensor, model_tensor: torch.Tensor
) -> Optional[torch.Tensor]:
    """
    シンボルテーブル拡張 (韓国語対応など) の後方互換処理。
    保存済みテンソルの 0 次元目だけがモデルより小さい場合 (音素・トーン・言語の
    埋め込みテーブルが拡張された場合)、既存の行を先頭にコピーした拡張済みテンソルを返す。
    新規行のうち韓国語の音素・トーン・言語の行は JP 行の加重結合で warm-start 初期化し
    (nlp/korean/warm_start.py)、それ以外はモデルの初期値のままにする。
    それ以外の形状不一致の場合は None を返す。
    """
    if (
        saved_tensor.dim() == model_tensor.dim()
        and saved_tensor.shape[0] < model_tensor.shape[0]
        and saved_tensor.shape[1:] == model_tensor.shape[1:]
    ):
        expanded = model_tensor.clone()
        expanded[: saved_tensor.shape[0]] = saved_tensor
        if warm_start_new_rows(key, expanded, saved_tensor.shape[0]):
            new_rows = "KO rows are warm-started from JP rows"
        else:
            new_rows = "new rows keep their initial values"
        logger.info(
            f"Expanded {key} from {tuple(saved_tensor.shape)} to "
            f"{tuple(model_tensor.shape)} ({new_rows})"
        )
        return expanded
    return None


def __expand_optimizer_state_if_needed(
    optimizer: torch.optim.Optimizer, saved_optimizer: dict
) -> None:
    """
    シンボルテーブル拡張前のチェックポイントの optimizer state (Adam の exp_avg 等) を
    拡張後のモデルで読み込めるようにする後方互換処理。埋め込みに対応する state テンソルの
    0 次元目が小さい場合、既存行を保持し新規行をゼロ (新規パラメータの初期 state) で拡張する。
    """
    params = [p for group in optimizer.param_groups for p in group["params"]]
    for index, state in saved_optimizer.get("state", {}).items():
        i = int(index)
        if not (0 <= i < len(params)):
            continue
        param = params[i]
        for key, tensor in state.items():
            if not isinstance(tensor, torch.Tensor) or tensor.shape == param.shape:
                continue
            expanded = expand_embedding_if_needed(
                f"optimizer state [{i}].{key}",
                tensor,
                torch.zeros(param.shape, dtype=tensor.dtype, device=tensor.device),
            )
            if expanded is not None:
                state[key] = expanded


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    skip_optimizer: bool = False,
    for_infer: bool = False,
    device: Union[str, torch.device] = "cpu",
) -> tuple[torch.nn.Module, Optional[torch.optim.Optimizer], float, int]:
    """
    指定されたパスからチェックポイントを読み込み、モデルとオプティマイザーを更新する。

    Args:
        checkpoint_path (Union[str, Path]): チェックポイントファイルのパス
        model (torch.nn.Module): 更新するモデル
        optimizer (Optional[torch.optim.Optimizer]): 更新するオプティマイザー。None の場合は更新しない
        skip_optimizer (bool): オプティマイザーの更新をスキップするかどうかのフラグ
        for_infer (bool): 推論用に読み込むかどうかのフラグ

    Returns:
        tuple[torch.nn.Module, Optional[torch.optim.Optimizer], float, int]: 更新されたモデルとオプティマイザー、学習率、イテレーション回数
    """

    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location=device)
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    logger.info(
        f"Loading model and optimizer at iteration {iteration} from {checkpoint_path}"
    )
    if (
        optimizer is not None
        and not skip_optimizer
        and checkpoint_dict["optimizer"] is not None
    ):
        __expand_optimizer_state_if_needed(optimizer, checkpoint_dict["optimizer"])
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    elif optimizer is None and not skip_optimizer:
        # else:      Disable this line if Infer and resume checkpoint,then enable the line upper
        new_opt_dict = optimizer.state_dict()  # type: ignore
        new_opt_dict_params = new_opt_dict["param_groups"][0]["params"]
        new_opt_dict["param_groups"] = checkpoint_dict["optimizer"]["param_groups"]
        new_opt_dict["param_groups"][0]["params"] = new_opt_dict_params
        optimizer.load_state_dict(new_opt_dict)  # type: ignore

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            # assert "emb_g" not in k
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (
                saved_state_dict[k].shape,
                v.shape,
            )
        except:
            # For upgrading from the old version
            if "ja_bert_proj" in k:
                v = torch.zeros_like(v)
                logger.warning(
                    f"Seems you are using the old version of the model, the {k} is automatically set to zero for backward compatibility"
                )
            elif "enc_q" in k and for_infer:
                continue
            elif k in saved_state_dict:
                # シンボルテーブル拡張 (韓国語対応など) による埋め込みサイズ差の吸収
                expanded = expand_embedding_if_needed(k, saved_state_dict[k], v)
                if expanded is not None:
                    v = expanded
                else:
                    logger.error(
                        f"Shape mismatch for {k}: "
                        f"{tuple(saved_state_dict[k].shape)} != {tuple(v.shape)}"
                    )
            else:
                logger.error(f"{k} is not in the checkpoint {checkpoint_path}")

            new_state_dict[k] = v

    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)

    logger.info(f"Loaded '{checkpoint_path}' (iteration {iteration})")

    return model, optimizer, learning_rate, iteration


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Union[torch.optim.Optimizer, torch.optim.AdamW],
    learning_rate: float,
    iteration: int,
    checkpoint_path: Union[str, Path],
) -> None:
    """
    モデルとオプティマイザーの状態を指定されたパスに保存する。

    Args:
        model (torch.nn.Module): 保存するモデル
        optimizer (Union[torch.optim.Optimizer, torch.optim.AdamW]): 保存するオプティマイザー
        learning_rate (float): 学習率
        iteration (int): イテレーション回数
        checkpoint_path (Union[str, Path]): 保存先のパス
    """
    logger.info(
        f"Saving model and optimizer state at iteration {iteration} to {checkpoint_path}"
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def clean_checkpoints(
    model_dir_path: Union[str, Path] = "logs/44k/",
    n_ckpts_to_keep: int = 2,
    sort_by_time: bool = True,
) -> None:
    """
    指定されたディレクトリから古いチェックポイントを削除して空き容量を確保する

    Args:
        model_dir_path (Union[str, Path]): モデルが保存されているディレクトリのパス
        n_ckpts_to_keep (int): 保持するチェックポイントの数（G_0.pth と D_0.pth を除く）
        sort_by_time (bool): True の場合、時間順に削除。False の場合、名前順に削除
    """

    ckpts_files = [
        f
        for f in os.listdir(model_dir_path)
        if os.path.isfile(os.path.join(model_dir_path, f))
    ]

    def name_key(_f: str) -> int:
        return int(re.compile("._(\\d+)\\.pth").match(_f).group(1))  # type: ignore

    def time_key(_f: str) -> float:
        return os.path.getmtime(os.path.join(model_dir_path, _f))

    sort_key = time_key if sort_by_time else name_key

    def x_sorted(_x: str) -> list[str]:
        return sorted(
            [f for f in ckpts_files if f.startswith(_x) and not f.endswith("_0.pth")],
            key=sort_key,
        )

    to_del = [
        os.path.join(model_dir_path, fn)
        for fn in (
            x_sorted("G_")[:-n_ckpts_to_keep]
            + x_sorted("D_")[:-n_ckpts_to_keep]
            + x_sorted("WD_")[:-n_ckpts_to_keep]
            + x_sorted("DUR_")[:-n_ckpts_to_keep]
        )
    ]

    def del_info(fn: str) -> None:
        return logger.info(f"Free up space by deleting ckpt {fn}")

    def del_routine(x: str) -> list[Any]:
        return [os.remove(x), del_info(x)]

    [del_routine(fn) for fn in to_del]


def get_latest_checkpoint_path(
    model_dir_path: Union[str, Path], regex: str = "G_*.pth"
) -> str:
    """
    指定されたディレクトリから最新のチェックポイントのパスを取得する

    Args:
        model_dir_path (Union[str, Path]): モデルが保存されているディレクトリのパス
        regex (str): チェックポイントのファイル名の正規表現

    Returns:
        str: 最新のチェックポイントのパス
    """

    f_list = glob.glob(os.path.join(str(model_dir_path), regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    try:
        x = f_list[-1]
    except IndexError:
        raise ValueError(f"No checkpoint found in {model_dir_path} with regex {regex}")

    return x


def find_pretrained_paths(
    model_dir_path: Union[str, Path], prefixes: list[str]
) -> dict[str, Path]:
    """
    学習開始時に読み込む事前学習モデル ({prefix}_0.safetensors) のパスを返す。
    1 つでも欠けていれば、事前学習なしのゼロからの学習が黙って始まらないよう FileNotFoundError を送出する。
    """
    paths = {p: Path(model_dir_path) / f"{p}_0.safetensors" for p in prefixes}
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Pretrained models not found in {model_dir_path}: {', '.join(missing)}")
    return paths


def find_resume_checkpoints(
    model_dir_path: Union[str, Path], prefixes: list[str]
) -> tuple[int, dict[str, Path]]:
    """
    学習再開に使うチェックポイント ({prefix}_{step}.pth) を、全 prefix がそろった最新ステップで選んで返す。
    保存中の中断などで最新の G のステップに欠けがあれば警告し、そろっている直前のステップを使う。
    そろったステップが 1 つもなければ FileNotFoundError を送出する。
    """
    model_dir = Path(model_dir_path)
    steps = {
        p: {int(m.group(1)) for f in model_dir.glob(f"{p}_*.pth") if (m := re.fullmatch(rf"{p}_(\d+)\.pth", f.name))}
        for p in prefixes
    }
    complete = set.intersection(*steps.values())
    if not complete:
        raise FileNotFoundError(f"No complete checkpoint set ({', '.join(f'{p}_*.pth' for p in prefixes)}) in {model_dir}")
    step = max(complete)
    latest = max(steps[prefixes[0]])
    if latest > step:
        missing = [f"{p}_{latest}.pth" for p in prefixes if latest not in steps[p]]
        logger.warning(f"{', '.join(missing)} not found, so resuming from step {step} instead of {latest}")
    return step, {p: model_dir / f"{p}_{step}.pth" for p in prefixes}
