from __future__ import annotations

import gc
import math
import time
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import numpy as np
import onnxruntime
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.streaming import DecoderGeometry, decoder_geometry, float_to_pcm16_fixed, plan_chunks
from style_bert_vits2.voice import adjust_voice


if TYPE_CHECKING:
    from style_bert_vits2.models.models import SynthesizerTrn
    from style_bert_vits2.models.models_jp_extra import (
        SynthesizerTrn as SynthesizerTrnJPExtra,
    )


class NullModelParam(BaseModel):
    """
    ヌルモデルのパラメータを表す Pydantic モデル。
    各パラメータは 0.0 から 1.0 の範囲で指定する。
    """

    name: str  # モデル名
    path: Path  # モデルファイルのパス
    weight: float = Field(ge=0.0, le=1.0)  # 声質の重み
    pitch: float = Field(ge=0.0, le=1.0)  # 声の高さの重み
    style: float = Field(ge=0.0, le=1.0)  # 話し方の重み
    tempo: float = Field(ge=0.0, le=1.0)  # テンポの重み


@dataclass
class AudioStream:
    """
    ストリーミング音声合成の結果。for 文で 16bit PCM (int16) の音声チャンクを生成順に取り出せる。
    num_samples は全体のサンプル数 (改行分割で複数行を生成する場合は事前に分からないため None)。
    """

    sample_rate: int
    num_samples: Optional[int]
    chunks: Generator[NDArray[np.int16], None, None]

    def __iter__(self) -> Iterator[NDArray[np.int16]]:
        return self.chunks

    def close(self) -> None:
        """未消費のチャンク生成を打ち切り、保持している中間テンソルを解放する (複数回呼んでも安全)"""
        self.chunks.close()


class TTSModel:
    """
    Style-Bert-VITS2 の音声合成モデルを操作するクラス。
    モデル/ハイパーパラメータ/スタイルベクトルのパスとデバイスを指定して初期化し、model.infer() メソッドを呼び出すと音声合成を行える。
    """

    def __init__(
        self,
        model_path: Path,
        config_path: Union[Path, HyperParameters],
        style_vec_path: Union[Path, NDArray[Any]],
        device: str = "cpu",
        onnx_providers: Sequence[Union[str, tuple[str, dict[str, Any]]]] = [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})],
    ) -> None:  # fmt: skip
        """
        Style-Bert-VITS2 の音声合成モデルを初期化する。
        この時点ではモデルはロードされていない (明示的にロードしたい場合は model.load() を呼び出す)。

        Args:
            model_path (Path): モデル (.safetensors / .onnx) のパス
            config_path (Union[Path, HyperParameters]): ハイパーパラメータ (config.json) のパス (直接 HyperParameters を指定することも可能)
            style_vec_path (Union[Path, NDArray[Any]]): スタイルベクトル (style_vectors.npy) のパス (直接 NDArray を指定することも可能)
            device (str): PyTorch 推論での音声合成時に利用するデバイス (cpu, cuda, mps など)
            onnx_providers (list[str]): ONNX 推論で利用する ExecutionProvider (CPUExecutionProvider, CUDAExecutionProvider など)
        """

        self.model_path: Path = model_path
        self.device: str = device
        self.onnx_providers: Sequence[Union[str, tuple[str, dict[str, Any]]]] = onnx_providers  # fmt: skip

        # ONNX 形式のモデルかどうか
        if self.model_path.suffix == ".onnx":
            self.is_onnx_model = True
        else:
            self.is_onnx_model = False

        # ハイパーパラメータの Pydantic モデルが直接指定された
        if isinstance(config_path, HyperParameters):
            self.config_path: Path = Path("")  # 互換性のため空の Path を設定
            self.hyper_parameters: HyperParameters = config_path
        # ハイパーパラメータのパスが指定された
        else:
            self.config_path: Path = config_path
            self.hyper_parameters: HyperParameters = HyperParameters.load_from_json(
                self.config_path
            )

        # スタイルベクトルの NDArray が直接指定された
        if isinstance(style_vec_path, np.ndarray):
            self.style_vec_path: Path = Path("")  # 互換性のため空の Path を設定
            self.style_vectors: NDArray[Any] = style_vec_path
        # スタイルベクトルのパスが指定された
        else:
            self.style_vec_path: Path = style_vec_path
            self.style_vectors: NDArray[Any] = np.load(self.style_vec_path)

        self.spk2id: dict[str, int] = self.hyper_parameters.data.spk2id
        self.id2spk: dict[int, str] = {v: k for k, v in self.spk2id.items()}

        num_styles: int = self.hyper_parameters.data.num_styles
        if hasattr(self.hyper_parameters.data, "style2id"):
            self.style2id: dict[str, int] = self.hyper_parameters.data.style2id
        else:
            self.style2id: dict[str, int] = {str(i): i for i in range(num_styles)}
        if len(self.style2id) != num_styles:
            raise ValueError(
                f"Number of styles ({num_styles}) does not match the number of style2id ({len(self.style2id)})"
            )

        if self.style_vectors.shape[0] != num_styles:
            raise ValueError(
                f"The number of styles ({num_styles}) does not match the number of style vectors ({self.style_vectors.shape[0]})"
            )
        self.style_vector_inference: Optional[Any] = None

        # net_g / null_model_params は PyTorch 推論時のみ遅延初期化される
        self.net_g: Union[SynthesizerTrn, SynthesizerTrnJPExtra, None] = None
        self.null_model_params: Optional[dict[int, NullModelParam]] = None

        # onnx_session は ONNX 推論時のみ遅延初期化される
        self.onnx_session: Optional[onnxruntime.InferenceSession] = None

    def load(self) -> None:
        """
        音声合成モデルをデバイスにロードする。
        """

        start_time = time.time()

        # PyTorch 推論時
        if not self.is_onnx_model:
            from style_bert_vits2.models.infer import get_net_g

            # PyTorch モデルをロード
            self.net_g = get_net_g(
                model_path=str(self.model_path),
                version=self.hyper_parameters.version,
                device=self.device,
                hps=self.hyper_parameters,
            )
            logger.info(
                f'Model loaded successfully from {self.model_path} to "{self.device}" device ({time.time() - start_time:.2f}s)'
            )

            # ここからはヌルモデルのロード用パラメータが指定されている場合のみ
            if self.null_model_params is None:
                return

            # 推論対象のモデルの重みとヌルモデルの重みをマージ
            for null_model_info in self.null_model_params.values():
                logger.info(f"Adding null model: {null_model_info.path}...")
                null_model_add = get_net_g(
                    model_path=str(null_model_info.path),
                    version=self.hyper_parameters.version,
                    device=self.device,
                    hps=self.hyper_parameters,
                )
                # 愚直。もっと上手い方法ありそう
                params = zip(
                    self.net_g.dec.parameters(), null_model_add.dec.parameters()
                )
                for v in params:
                    v[0].data.add_(v[1].data, alpha=float(null_model_info.weight))
                params = zip(
                    self.net_g.flow.parameters(), null_model_add.flow.parameters()
                )
                for v in params:
                    v[0].data.add_(v[1].data, alpha=float(null_model_info.pitch))

                params = zip(
                    self.net_g.enc_p.parameters(), null_model_add.enc_p.parameters()
                )
                for v in params:
                    v[0].data.add_(v[1].data, alpha=float(null_model_info.style))
                # テンポは sdp と dp 二つあるからとりあえずどっちも足す
                params = zip(
                    self.net_g.sdp.parameters(), null_model_add.sdp.parameters()
                )
                for v in params:
                    v[0].data.add_(v[1].data, alpha=float(null_model_info.tempo))
                params = zip(self.net_g.dp.parameters(), null_model_add.dp.parameters())
                for v in params:
                    v[0].data.add_(v[1].data, alpha=float(null_model_info.tempo))

            logger.info(
                f"Null models merged successfully ({time.time() - start_time:.2f}s)"
            )

        # ONNX 推論時
        else:
            # 推論時に一番優先される ExecutionProvider の名前を取得
            assert len(self.onnx_providers) > 0
            first_provider_name = (
                self.onnx_providers[0]
                if type(self.onnx_providers[0]) is str
                else self.onnx_providers[0][0]
            )

            # 推論セッションの設定
            sess_options = onnxruntime.SessionOptions()
            ## ONNX モデルの作成時にすでに onnxsim により最適化されていることから、ロード高速化のため最適化を無効にする
            ## DmlExecutionProvider が先頭に指定されているときのみ、DirectML 推論の高速化のためすべての最適化を有効にする
            if first_provider_name == "DmlExecutionProvider":
                sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL  # fmt: skip
            else:
                sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL  # fmt: skip
            ## エラー以外のログを出力しない
            ## 本来は log_severity_level = 3 だけで効くはずだが、なぜか CUDA 系のログが抑制できないので set_default_logger_severity() も呼び出している
            sess_options.log_severity_level = 3
            onnxruntime.set_default_logger_severity(3)

            # ONNX モデルをロードし、推論セッションを初期化
            self.onnx_session = onnxruntime.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=self.onnx_providers,
            )
            logger.info(
                f"Model loaded successfully from {self.model_path} to {self.onnx_session.get_providers()[0]} ({time.time() - start_time:.2f}s)"
            )

    def unload(self) -> None:
        """
        音声合成モデルをデバイスからアンロードする。
        PyTorch モデルの場合は CUDA メモリも解放される。
        """

        start_time = time.time()

        # PyTorch 推論時
        if self.net_g is not None:
            import torch

            del self.net_g
            self.net_g = None

            # CUDA キャッシュをクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ONNX 推論時
        if self.onnx_session is not None:
            del self.onnx_session
            self.onnx_session = None

        gc.collect()
        logger.info(f"Model unloaded successfully ({time.time() - start_time:.2f}s)")

    def get_style_vector(self, style_id: int, weight: float = 1.0) -> NDArray[Any]:
        """
        スタイルベクトルを取得する。

        Args:
            style_id (int): スタイル ID (0 から始まるインデックス)
            weight (float, optional): スタイルベクトルの重み. Defaults to 1.0.

        Returns:
            NDArray[Any]: スタイルベクトル
        """
        mean = self.style_vectors[0]
        style_vec = self.style_vectors[style_id]
        style_vec = mean + (style_vec - mean) * weight
        return style_vec

    def get_style_vector_from_audio(
        self, audio_path: str, weight: float = 1.0
    ) -> NDArray[Any]:
        """
        音声からスタイルベクトルを推論する。

        Args:
            audio_path (str): 音声ファイルのパス
            weight (float, optional): スタイルベクトルの重み. Defaults to 1.0.
        Returns:
            NDArray[Any]: スタイルベクトル
        """

        if self.style_vector_inference is None:

            # pyannote.audio は scikit-learn などの大量の重量級ライブラリに依存しているため、
            # TTSModel.infer() に reference_audio_path を指定し音声からスタイルベクトルを推論する場合のみ遅延 import する
            try:
                import pyannote.audio
            except ImportError:
                raise ImportError(
                    "pyannote.audio is required to infer style vector from audio"
                )

            # スタイルベクトルを取得するための推論モデルを初期化
            import torch

            from style_bert_vits2.utils import load_pyannote_wespeaker_model

            self.style_vector_inference = pyannote.audio.Inference(
                model=load_pyannote_wespeaker_model(),
                window="whole",
            )
            self.style_vector_inference.to(torch.device(self.device))

        # 音声からスタイルベクトルを推論
        xvec = self.style_vector_inference(audio_path)
        mean = self.style_vectors[0]
        xvec = mean + (xvec - mean) * weight
        return xvec

    @staticmethod
    def convert_to_16_bit_wav(data: NDArray[Any]) -> NDArray[Any]:
        """
        音声データを 16-bit int 形式に変換する。
        gradio.processing_utils.convert_to_16_bit_wav() を移植したもの。

        Args:
            data (NDArray[Any]): 音声データ

        Returns:
            NDArray[Any]: 16-bit int 形式の音声データ
        """

        # Based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
        if data.dtype in [np.float64, np.float32, np.float16]:  # type: ignore
            peak = np.abs(data).max()
            # 完全な無音はゼロ除算で NaN にならないよう、そのまま 0 を返す
            if peak == 0:
                return np.zeros(data.shape, dtype=np.int16)
            data = data / peak
            data = data * 32767
            data = data.astype(np.int16)
        elif data.dtype == np.int32:
            data = data / 65536
            data = data.astype(np.int16)
        elif data.dtype == np.int16:
            pass
        elif data.dtype == np.uint16:
            data = data - 32768
            data = data.astype(np.int16)
        elif data.dtype == np.uint8:
            data = data * 257 - 32768
            data = data.astype(np.int16)
        elif data.dtype == np.int8:
            data = data * 256
            data = data.astype(np.int16)
        else:
            raise ValueError(
                "Audio data cannot be converted automatically from "
                f"{data.dtype} to 16-bit int format."
            )

        return data

    @property
    def trained_language(self) -> Optional[Languages]:
        """
        config.json に記録された学習言語 (学習データで最も多い言語) を返す。記録のない旧モデルでは None。
        """
        languages = self.hyper_parameters.data.languages
        return Languages(languages[0]) if languages else None

    def infer(
        self,
        text: str,
        language: Optional[Languages] = None,
        speaker_id: int = 0,
        reference_audio_path: Optional[str] = None,
        sdp_ratio: float = DEFAULT_SDP_RATIO,
        noise: float = DEFAULT_NOISE,
        noise_w: float = DEFAULT_NOISEW,
        length: float = DEFAULT_LENGTH,
        line_split: bool = DEFAULT_LINE_SPLIT,
        split_interval: float = DEFAULT_SPLIT_INTERVAL,
        assist_text: Optional[str] = None,
        assist_text_weight: float = DEFAULT_ASSIST_TEXT_WEIGHT,
        use_assist_text: bool = False,
        style: str = DEFAULT_STYLE,
        style_weight: float = DEFAULT_STYLE_WEIGHT,
        given_phone: Optional[list[str]] = None,
        given_tone: Optional[list[int]] = None,
        pitch_scale: float = 1.0,
        intonation_scale: float = 1.0,
        null_model_params: Optional[dict[int, NullModelParam]] = None,
        force_reload_model: bool = False,
        *,
        pcm_scale: Literal["legacy_peak", "fixed"] = "legacy_peak",
    ) -> tuple[int, NDArray[Any]]:
        """
        テキストから音声を合成する。

        Args:
            text (str): 読み上げるテキスト
            language (Optional[Languages], optional): 言語。省略時はモデルの学習言語 (記録がなければ JP). Defaults to None.
            speaker_id (int, optional): 話者 ID. Defaults to 0.
            reference_audio_path (Optional[str], optional): 音声スタイルの参照元の音声ファイルのパス. Defaults to None.
            sdp_ratio (float, optional): DP と SDP の混合比。0 で DP のみ、1で SDP のみを使用 (値を大きくするとテンポに緩急がつく). Defaults to DEFAULT_SDP_RATIO.
            noise (float, optional): DP に与えられるノイズ. Defaults to DEFAULT_NOISE.
            noise_w (float, optional): SDP に与えられるノイズ. Defaults to DEFAULT_NOISEW.
            length (float, optional): 生成音声の長さ（話速）のパラメータ。大きいほど生成音声が長くゆっくり、小さいほど短く早くなる。 Defaults to DEFAULT_LENGTH.
            line_split (bool, optional): テキストを改行ごとに分割して生成するかどうか (True の場合 given_phone/given_tone は無視される). Defaults to DEFAULT_LINE_SPLIT.
            split_interval (float, optional): 改行ごとに分割する場合の無音 (秒). Defaults to DEFAULT_SPLIT_INTERVAL.
            assist_text (Optional[str], optional): 感情表現の参照元の補助テキスト. Defaults to None.
            assist_text_weight (float, optional): 感情表現の補助テキストを適用する強さ. Defaults to DEFAULT_ASSIST_TEXT_WEIGHT.
            use_assist_text (bool, optional): 音声合成時に感情表現の補助テキストを使用するかどうか. Defaults to False.
            style (str, optional): 音声スタイル (Neutral, Happy など). Defaults to DEFAULT_STYLE.
            style_weight (float, optional): 音声スタイルを適用する強さ. Defaults to DEFAULT_STYLE_WEIGHT.
            given_phone (Optional[list[int]], optional): 読み上げテキストの読みを表す音素列。指定する場合は given_tone も別途指定が必要. Defaults to None.
            given_tone (Optional[list[int]], optional): アクセントのトーンのリスト. Defaults to None.
            pitch_scale (float, optional): ピッチの高さ (1.0 から変更すると若干音質が低下する). Defaults to 1.0.
            intonation_scale (float, optional): 抑揚の平均からの変化幅 (1.0 から変更すると若干音質が低下する). Defaults to 1.0.
            null_model_params (Optional[dict[int, NullModelParam]], optional): 推論時に使用するヌルモデルの情報。ONNX 推論では無視される。
            force_reload_model (bool, optional): モデルを強制的に再ロードするかどうか. Defaults to False.
            pcm_scale (Literal["legacy_peak", "fixed"], optional): 16bit PCM への変換方式。legacy_peak は従来どおり最大振幅で正規化し、fixed は正規化せず固定スケールで変換する (infer_stream() と同じ音量). Defaults to "legacy_peak".
        Returns:
            tuple[int, NDArray[Any]]: サンプリングレートと音声データ (16bit PCM)
        """

        if pcm_scale not in ("legacy_peak", "fixed"):
            raise ValueError(f"pcm_scale must be 'legacy_peak' or 'fixed': {pcm_scale}")
        if language is None:
            language = self.trained_language or Languages.JP
        logger.info(f"Start generating audio data from text:\n{text}")
        # KO は JP-Extra 系アーキテクチャ (単一 BERT 入力) を共用するため許可する
        if language not in ("JP", "KO") and self.hyper_parameters.version.endswith("JP-Extra"):  # fmt: skip
            raise ValueError(
                "The model is trained with JP-Extra, but the language is not JP or KO"
            )
        if reference_audio_path == "":
            reference_audio_path = None
        if assist_text == "" or not use_assist_text:
            assist_text = None

        # スタイルベクトルを取得
        if reference_audio_path is None:
            style_id = self.style2id[style]
            style_vector = self.get_style_vector(style_id, style_weight)
        else:
            style_vector = self.get_style_vector_from_audio(
                reference_audio_path, style_weight
            )

        # PyTorch 推論時
        start_time = time.time()
        if not self.is_onnx_model:
            import torch

            from style_bert_vits2.models.infer import infer

            if null_model_params is not None:
                self.null_model_params = null_model_params
            else:
                self.null_model_params = None

            # force_reload_model が True のとき、メモリ上に保持されているモデルを破棄する
            if force_reload_model is True:
                self.net_g = None

            # モデルがロードされていない場合はロードする
            if self.net_g is None:
                self.load()
            assert self.net_g is not None

            # 通常のテキストから音声を生成
            if not line_split:
                with torch.no_grad():
                    audio = infer(
                        text=text,
                        sdp_ratio=sdp_ratio,
                        noise_scale=noise,
                        noise_scale_w=noise_w,
                        length_scale=length,
                        sid=speaker_id,
                        language=language,
                        hps=self.hyper_parameters,
                        net_g=self.net_g,
                        device=self.device,
                        assist_text=assist_text,
                        assist_text_weight=assist_text_weight,
                        style_vec=style_vector,
                        given_phone=given_phone,
                        given_tone=given_tone,
                    )

            # 改行ごとに分割して音声を生成
            else:
                texts = [t for t in text.split("\n") if t != ""]
                audios = []
                with torch.no_grad():
                    for i, t in enumerate(texts):
                        audios.append(
                            infer(
                                text=t,
                                sdp_ratio=sdp_ratio,
                                noise_scale=noise,
                                noise_scale_w=noise_w,
                                length_scale=length,
                                sid=speaker_id,
                                language=language,
                                hps=self.hyper_parameters,
                                net_g=self.net_g,
                                device=self.device,
                                assist_text=assist_text,
                                assist_text_weight=assist_text_weight,
                                style_vec=style_vector,
                            )
                        )
                        if i != len(texts) - 1:
                            audios.append(np.zeros(int(44100 * split_interval)))
                    audio = np.concatenate(audios)

        # ONNX 推論時
        else:
            from style_bert_vits2.models.infer_onnx import infer_onnx

            # force_reload_model が True のとき、メモリ上に保持されているモデルを破棄する
            if force_reload_model is True:
                self.onnx_session = None

            # モデルがロードされていない場合はロードする
            if self.onnx_session is None:
                self.load()
            assert self.onnx_session is not None

            # 通常のテキストから音声を生成
            if not line_split:
                audio = infer_onnx(
                    text=text,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise,
                    noise_scale_w=noise_w,
                    length_scale=length,
                    sid=speaker_id,
                    language=language,
                    hps=self.hyper_parameters,
                    onnx_session=self.onnx_session,
                    onnx_providers=self.onnx_providers,
                    assist_text=assist_text,
                    assist_text_weight=assist_text_weight,
                    style_vec=style_vector,
                    given_phone=given_phone,
                    given_tone=given_tone,
                )

            # 改行ごとに分割して音声を生成
            else:
                texts = [t for t in text.split("\n") if t != ""]
                audios = []
                for i, t in enumerate(texts):
                    audios.append(
                        infer_onnx(
                            text=t,
                            sdp_ratio=sdp_ratio,
                            noise_scale=noise,
                            noise_scale_w=noise_w,
                            length_scale=length,
                            sid=speaker_id,
                            language=language,
                            hps=self.hyper_parameters,
                            onnx_session=self.onnx_session,
                            onnx_providers=self.onnx_providers,
                            assist_text=assist_text,
                            assist_text_weight=assist_text_weight,
                            style_vec=style_vector,
                        )
                    )
                    if i != len(texts) - 1:
                        audios.append(np.zeros(int(44100 * split_interval)))
                audio = np.concatenate(audios)

        logger.info(
            f"Audio data generated successfully ({time.time() - start_time:.2f}s)"
        )

        if not (pitch_scale == 1.0 and intonation_scale == 1.0):
            _, audio = adjust_voice(
                fs=self.hyper_parameters.data.sampling_rate,
                wave=audio,
                pitch_scale=pitch_scale,
                intonation_scale=intonation_scale,
            )
        if pcm_scale == "fixed":
            audio = float_to_pcm16_fixed(audio)
        else:
            audio = self.convert_to_16_bit_wav(audio)
        return (self.hyper_parameters.data.sampling_rate, audio)

    def infer_stream(
        self,
        text: str,
        language: Optional[Languages] = None,
        speaker_id: int = 0,
        reference_audio_path: Optional[str] = None,
        sdp_ratio: float = DEFAULT_SDP_RATIO,
        noise: float = DEFAULT_NOISE,
        noise_w: float = DEFAULT_NOISEW,
        length: float = DEFAULT_LENGTH,
        line_split: bool = DEFAULT_LINE_SPLIT,
        split_interval: float = DEFAULT_SPLIT_INTERVAL,
        assist_text: Optional[str] = None,
        assist_text_weight: float = DEFAULT_ASSIST_TEXT_WEIGHT,
        use_assist_text: bool = False,
        style: str = DEFAULT_STYLE,
        style_weight: float = DEFAULT_STYLE_WEIGHT,
        given_phone: Optional[list[str]] = None,
        given_tone: Optional[list[int]] = None,
        null_model_params: Optional[dict[int, NullModelParam]] = None,
        force_reload_model: bool = False,
        *,
        chunk_size: int = 100,
        overlap_size: int = 32,
    ) -> AudioStream:
        """
        テキストから音声をストリーミング合成する。デコーダを chunk_size フレームずつ実行し、生成された順に 16bit PCM のチャンクを返す。
        入力の検証と最初の行の潜在変数の生成はこの呼び出しの時点で完了し、デコードは返り値を反復したときに行われる。
        音量は正規化しない固定スケール (infer(pcm_scale="fixed") と同じ) で、JP-Extra モデル (JP / KO) の PyTorch 推論のみ対応する。
        pitch_scale / intonation_scale は波形全体を必要とするため対応しない。

        Args:
            text 〜 force_reload_model: infer() と同じ
            chunk_size (int, optional): 1 回にデコードするフレーム数. Defaults to 100.
            overlap_size (int, optional): 隣り合うチャンクの重なりフレーム数 (偶数、モデルごとの最小値以上). Defaults to 32.

        Returns:
            AudioStream: サンプリングレート・総サンプル数 (改行分割で複数行の場合は None)・int16 チャンクの反復子
        """

        from style_bert_vits2.models.infer import prepare_latent

        if language is None:
            language = self.trained_language or Languages.JP
        # モデルに触れる前に、すべての入力を検証する
        if self.is_onnx_model:
            raise ValueError("Streaming inference is not supported for ONNX models")
        if not self.hyper_parameters.version.endswith("JP-Extra"):
            raise ValueError("Streaming inference supports JP-Extra models only")
        if language not in ("JP", "KO"):
            raise ValueError("The model is trained with JP-Extra, but the language is not JP or KO")
        geometry = decoder_geometry(self.hyper_parameters.model)
        plan_chunks(1, chunk_size, overlap_size)  # 引数の妥当性だけを先に確認する
        if overlap_size < geometry.min_overlap:
            raise ValueError(f"overlap_size must be at least {geometry.min_overlap} for this model: {overlap_size}")
        if not math.isfinite(split_interval) or split_interval < 0:
            raise ValueError(f"split_interval must be a finite non-negative number: {split_interval}")
        # infer() と同じく、改行分割時は空行だけを除外する
        texts = [t for t in text.split("\n") if t != ""] if line_split else [text]
        if len(texts) == 0 or texts[0] == "":
            raise ValueError("Text to synthesize is empty")

        logger.info(f"Start streaming audio generation from text:\n{text}")
        if reference_audio_path == "":
            reference_audio_path = None
        if assist_text == "" or not use_assist_text:
            assist_text = None
        if reference_audio_path is None:
            style_vector = self.get_style_vector(self.style2id[style], style_weight)
        else:
            style_vector = self.get_style_vector_from_audio(reference_audio_path, style_weight)

        # infer() と同じく、ヌルモデルの設定はロード時にマージされる
        self.null_model_params = null_model_params
        if force_reload_model is True:
            self.net_g = None
        if self.net_g is None:
            self.load()
        assert self.net_g is not None
        # 反復の途中で他の呼び出しがモデルを再ロードしても、このストリームは開始時点のモデルを使い続ける
        net_g = self.net_g

        latent_args: dict[str, Any] = dict(
            style_vec=style_vector, sdp_ratio=sdp_ratio, noise_scale=noise, noise_scale_w=noise_w, length_scale=length,
            sid=speaker_id, language=language, hps=self.hyper_parameters, net_g=net_g, device=self.device,
            assist_text=assist_text, assist_text_weight=assist_text_weight,
        )
        # infer() と同じく、改行分割時は given_phone / given_tone を無視する
        if line_split:
            first = prepare_latent(texts[0], **latent_args)
        else:
            first = prepare_latent(texts[0], **latent_args, given_phone=given_phone, given_tone=given_tone)
        num_samples = first[2] * geometry.upsample_factor if len(texts) == 1 else None
        chunks = self._iter_stream_chunks(texts, first, latent_args, geometry, chunk_size, overlap_size, split_interval)
        return AudioStream(self.hyper_parameters.data.sampling_rate, num_samples, chunks)

    def _iter_stream_chunks(
        self,
        texts: list[str],
        first: tuple[Any, Any, int],
        latent_args: dict[str, Any],
        geometry: DecoderGeometry,
        chunk_size: int,
        overlap_size: int,
        split_interval: float,
    ) -> Generator[NDArray[np.int16], None, None]:
        """infer_stream() の反復本体。行ごとに潜在変数を用意し、デコードしたチャンクを 16bit PCM に変換して返す。"""

        import torch

        from style_bert_vits2.models.infer import decode_chunks, prepare_latent

        start_time = time.time()
        silence = np.zeros(int(self.hyper_parameters.data.sampling_rate * split_interval), dtype=np.int16)
        pending: Optional[tuple[Any, Any, int]] = first
        del first  # 最初の行の潜在変数を、その行のデコードが終わったら解放できるようにする
        try:
            for i, text in enumerate(texts):
                z, g, num_frames = pending if pending is not None else prepare_latent(text, **latent_args)
                pending = None
                plans = plan_chunks(num_frames, chunk_size, overlap_size)
                for audio in decode_chunks(latent_args["net_g"], z, g, plans, geometry.upsample_factor):
                    yield float_to_pcm16_fixed(audio)
                del z, g
                if i != len(texts) - 1 and len(silence) > 0:
                    yield silence
        except GeneratorExit:
            logger.info("Streaming audio generation was closed before completion")
            raise
        except Exception:
            logger.exception("Error occurred during streaming audio generation")
            raise
        logger.info(f"Streaming audio generated successfully ({time.time() - start_time:.2f}s)")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TTSModelInfo(BaseModel):
    name: str
    files: list[str]
    styles: list[str]
    speakers: list[str]


class TTSModelHolder:
    """
    Style-Bert-VITS2 の音声合成モデルを管理するクラス。
    model_holder.models_info から指定されたディレクトリ内にある音声合成モデルの一覧を取得できる。
    """

    def __init__(
        self,
        model_root_dir: Path,
        device: str,
        onnx_providers: Sequence[Union[str, tuple[str, dict[str, Any]]]],
        ignore_onnx: bool = False,
    ) -> None:
        """
        Style-Bert-VITS2 の音声合成モデルを管理するクラスを初期化する。
        音声合成モデルは下記のように配置されていることを前提とする (.safetensors / .onnx のファイル名は自由) 。
        ```
        model_root_dir
        ├── model-name-1
        │   ├── config.json
        │   ├── model-name-1_e160_s14000.safetensors
        │   └── style_vectors.npy
        ├── model-name-2
        │   ├── config.json
        │   ├── model-name-2_e160_s14000.safetensors
        │   └── style_vectors.npy
        └── ...
        ```

        Args:
            model_root_dir (Path): 音声合成モデルが配置されているディレクトリのパス
            device (str): PyTorch 推論での音声合成時に利用するデバイス (cpu, cuda, mps など)
            onnx_providers (list[str]): ONNX 推論で利用する ExecutionProvider (CPUExecutionProvider, CUDAExecutionProvider など)
            ignore_onnx (bool, optional): ONNX モデルを除外するかどうか. Defaults to False.
        """

        self.root_dir: Path = model_root_dir
        self.device: str = device
        self.onnx_providers: Sequence[Union[str, tuple[str, dict[str, Any]]]] = onnx_providers  # fmt: skip
        self.ignore_onnx: bool = ignore_onnx
        self.model_files_dict: dict[str, list[Path]] = {}
        self.current_model: Optional[TTSModel] = None
        self.model_names: list[str] = []
        self.models_info: list[TTSModelInfo] = []
        self.refresh()

    def refresh(self) -> None:
        """
        音声合成モデルの一覧を更新する。
        """

        self.model_files_dict = {}
        self.model_names = []
        self.current_model = None
        self.models_info = []

        model_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        for model_dir in model_dirs:
            if model_dir.name.startswith("."):
                continue
            suffixes = [".pth", ".pt", ".safetensors"]
            if self.ignore_onnx is False:
                suffixes.append(".onnx")
            model_files = sorted(
                [
                    f
                    for f in model_dir.iterdir()
                    # 上記 suffixes にマッチするファイルのみを取得し、. から始まるファイルは除外
                    if f.suffix in suffixes and not f.name.startswith(".")
                ],
                # 更新日時が新しい順にソート
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if len(model_files) == 0:
                logger.warning(f"No model files found in {model_dir}, so skip it")
                continue
            config_path = model_dir / "config.json"
            if not config_path.exists():
                logger.warning(
                    f"Config file {config_path} not found, so skip {model_dir}"
                )
                continue
            self.model_files_dict[model_dir.name] = model_files
            self.model_names.append(model_dir.name)
            hyper_parameters = HyperParameters.load_from_json(config_path)
            style2id: dict[str, int] = hyper_parameters.data.style2id
            styles = list(style2id.keys())
            spk2id: dict[str, int] = hyper_parameters.data.spk2id
            speakers = list(spk2id.keys())
            self.models_info.append(
                TTSModelInfo(
                    name=model_dir.name,
                    files=[str(f) for f in model_files],
                    styles=styles,
                    speakers=speakers,
                )
            )

    def get_model(self, model_name: str, model_path_str: str) -> TTSModel:
        """
        指定された音声合成モデルのインスタンスを取得する。
        この時点ではモデルはロードされていない (明示的にロードしたい場合は model.load() を呼び出す)。

        Args:
            model_name (str): 音声合成モデルの名前
            model_path_str (str): 音声合成モデルのファイルパス (.safetensors)

        Returns:
            TTSModel: 音声合成モデルのインスタンス
        """

        model_path = Path(model_path_str)
        if model_name not in self.model_files_dict:
            raise ValueError(f"Model `{model_name}` is not found")
        if model_path not in self.model_files_dict[model_name]:
            raise ValueError(f"Model file `{model_path}` is not found")
        if self.current_model is None or self.current_model.model_path != model_path:
            self.current_model = TTSModel(
                model_path=model_path,
                config_path=self.root_dir / model_name / "config.json",
                style_vec_path=self.root_dir / model_name / "style_vectors.npy",
                device=self.device,
                onnx_providers=self.onnx_providers,
            )

        return self.current_model

    def get_model_for_gradio(self, model_name: str, model_path_str: str):
        import gradio as gr

        model_path = Path(model_path_str)
        if model_name not in self.model_files_dict:
            raise ValueError(f"Model `{model_name}` is not found")
        if model_path not in self.model_files_dict[model_name]:
            raise ValueError(f"Model file `{model_path}` is not found")
        if (
            self.current_model is not None
            and self.current_model.model_path == model_path
        ):
            # Already loaded
            speakers = list(self.current_model.spk2id.keys())
            styles = list(self.current_model.style2id.keys())
            return (
                gr.Dropdown(choices=styles, value=styles[0]),
                gr.Button(interactive=True, value="음성 합성"),
                gr.Dropdown(choices=speakers, value=speakers[0]),
                self.__language_update_for_gradio(),
            )
        self.current_model = TTSModel(
            model_path=model_path,
            config_path=self.root_dir / model_name / "config.json",
            style_vec_path=self.root_dir / model_name / "style_vectors.npy",
            device=self.device,
            onnx_providers=self.onnx_providers,
        )
        speakers = list(self.current_model.spk2id.keys())
        styles = list(self.current_model.style2id.keys())
        return (
            gr.Dropdown(choices=styles, value=styles[0]),
            gr.Button(interactive=True, value="음성 합성"),
            gr.Dropdown(choices=speakers, value=speakers[0]),
            self.__language_update_for_gradio(),
        )

    def __language_update_for_gradio(self):
        """ロードしたモデルの学習言語を言語欄に反映する (記録のない旧モデルではユーザーの選択をそのまま残す)"""
        import gradio as gr

        assert self.current_model is not None
        language = self.current_model.trained_language
        return gr.update(value=language.value) if language is not None else gr.update()

    def update_model_files_for_gradio(self, model_name: str):
        import gradio as gr

        model_files = [str(f) for f in self.model_files_dict[model_name]]
        return gr.Dropdown(choices=model_files, value=model_files[0])

    def update_model_names_for_gradio(
        self,
    ):
        import gradio as gr

        self.refresh()
        initial_model_name = self.model_names[0]
        initial_model_files = [
            str(f) for f in self.model_files_dict[initial_model_name]
        ]
        return (
            gr.Dropdown(choices=self.model_names, value=initial_model_name),
            gr.Dropdown(choices=initial_model_files, value=initial_model_files[0]),
            gr.Button(interactive=False),  # For tts_button
        )
