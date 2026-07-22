import datetime
import json
from pathlib import Path
from typing import Optional

import gradio as gr

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
    GRADIO_THEME,
    Languages,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp import InvalidToneError
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.g2p_utils import g2kata_tone, kata_tone2phone_tone
from style_bert_vits2.nlp.japanese.normalizer import normalize_text
from style_bert_vits2.tts_model import NullModelParam, TTSModelHolder
from style_bert_vits2.utils import torch_device_to_onnx_providers


# pyopenjtalk_worker を起動
## pyopenjtalk_worker は TCP ソケットサーバーのため、ここで起動する
pyopenjtalk.initialize_worker()

# Web UI での学習時の無駄な GPU VRAM 消費を避けるため、あえてここでは BERT モデルの事前ロードを行わない
# データセットの BERT 特徴量は事前に bert_gen.py により抽出されているため、学習時に BERT モデルをロードしておく必要はない
# BERT モデルの事前ロードは「ロード」ボタン押下時に実行される TTSModelHolder.get_model_for_gradio() 内で行われる
# Web UI での学習時、音声合成タブの「ロード」ボタンを押さなければ、BERT モデルが VRAM にロードされていない状態で学習を開始できる

languages = [lang.value for lang in Languages]

initial_text = "こんにちは、初めまして。あなたの名前はなんていうの？"

examples = [
    [initial_text, "JP"],
    [
        """あなたがそんなこと言うなんて、私はとっても嬉しい。
あなたがそんなこと言うなんて、私はとっても怒ってる。
あなたがそんなこと言うなんて、私はとっても驚いてる。
あなたがそんなこと言うなんて、私はとっても辛い。""",
        "JP",
    ],
    [  # ChatGPTに考えてもらった告白セリフ
        """私、ずっと前からあなたのことを見てきました。あなたの笑顔、優しさ、強さに、心惹かれていたんです。
友達として過ごす中で、あなたのことがだんだんと特別な存在になっていくのがわかりました。
えっと、私、あなたのことが好きです！もしよければ、私と付き合ってくれませんか？""",
        "JP",
    ],
    [  # 夏目漱石『吾輩は猫である』
        """吾輩は猫である。名前はまだ無い。
どこで生れたかとんと見当がつかぬ。なんでも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。
吾輩はここで初めて人間というものを見た。しかもあとで聞くと、それは書生という、人間中で一番獰悪な種族であったそうだ。
この書生というのは時々我々を捕まえて煮て食うという話である。""",
        "JP",
    ],
    [  # 梶井基次郎『桜の樹の下には』
        """桜の樹の下には屍体が埋まっている！これは信じていいことなんだよ。
何故って、桜の花があんなにも見事に咲くなんて信じられないことじゃないか。俺はあの美しさが信じられないので、このにさんにち不安だった。
しかしいま、やっとわかるときが来た。桜の樹の下には屍体が埋まっている。これは信じていいことだ。""",
        "JP",
    ],
    [  # ChatGPTと考えた、感情を表すセリフ
        """やったー！テストで満点取れた！私とっても嬉しいな！
どうして私の意見を無視するの？許せない！ムカつく！あんたなんか死ねばいいのに。
あはははっ！この漫画めっちゃ笑える、見てよこれ、ふふふ、あはは。
あなたがいなくなって、私は一人になっちゃって、泣いちゃいそうなほど悲しい。""",
        "JP",
    ],
    [  # 上の丁寧語バージョン
        """やりました！テストで満点取れましたよ！私とっても嬉しいです！
どうして私の意見を無視するんですか？許せません！ムカつきます！あんたなんか死んでください。
あはははっ！この漫画めっちゃ笑えます、見てくださいこれ、ふふふ、あはは。
あなたがいなくなって、私は一人になっちゃって、泣いちゃいそうなほど悲しいです。""",
        "JP",
    ],
    [  # ChatGPTに考えてもらった音声合成の説明文章
        """音声合成は、機械学習を活用して、テキストから人の声を再現する技術です。この技術は、言語の構造を解析し、それに基づいて音声を生成します。
この分野の最新の研究成果を使うと、より自然で表現豊かな音声の生成が可能である。深層学習の応用により、感情やアクセントを含む声質の微妙な変化も再現することが出来る。""",
        "JP",
    ],
    [
        "Speech synthesis is the artificial production of human speech. A computer system used for this purpose is called a speech synthesizer, and can be implemented in software or hardware products.",
        "EN",
    ],
    [
        "语音合成是人工制造人类语音。用于此目的的计算机系统称为语音合成器，可以通过软件或硬件产品实现。",
        "ZH",
    ],
    [
        "음성 합성은 기계학습을 활용하여 텍스트로부터 사람의 목소리를 재현하는 기술입니다. 자연스러운 발음과 감정 표현이 가능합니다!",
        "KO",
    ],
]

initial_md = """
- Ver 2.5에서 추가된 기본 [`koharune-ami`(코하루네 아미) 모델](https://huggingface.co/litagin/sbv2_koharune_ami)과 [`amitaro`(아미타로) 모델](https://huggingface.co/litagin/sbv2_amitaro)은 [아미타로의 목소리 소재 공방](https://amitaro.net/)에서 공개된 코퍼스 음원·라이브 방송 음성을 사전에 허가를 받아 학습한 모델입니다. 아래의 **이용약관을 반드시 읽은 뒤** 이용해 주세요.

- Ver 2.5 업데이트 후 위 모델을 다운로드하려면 `Initialize.bat`을 더블클릭하거나, 수동으로 다운로드해 `model_assets` 디렉터리에 배치해 주세요.

- Ver 2.3에서 추가된 **에디터판**이 실제 낭독 용도로는 더 쓰기 편할 수 있습니다. `Editor.bat` 또는 `python server_editor.py --inbrowser`로 실행할 수 있습니다.
"""

terms_of_use_md = """
## 부탁 사항과 기본 모델 라이선스

최신 부탁 사항·이용약관은 [여기](https://github.com/litagin02/Style-Bert-VITS2/blob/master/docs/TERMS_OF_USE.md)를 참조해 주세요. 항상 최신 버전이 적용됩니다.

Style-Bert-VITS2를 사용할 때는 아래의 부탁 사항을 지켜 주시면 감사하겠습니다. 다만 모델 이용약관 이전 부분은 어디까지나 「부탁」이며 아무런 강제력이 없고, Style-Bert-VITS2의 이용약관도 아닙니다. 따라서 [리포지토리 라이선스](https://github.com/litagin02/Style-Bert-VITS2#license)와 모순되지 않으며, 리포지토리 이용에 있어서는 항상 리포지토리 라이선스만이 구속력을 가집니다.

### 하지 말아 주었으면 하는 것

다음 목적으로는 Style-Bert-VITS2를 사용하지 말아 주세요.

- 법률을 위반하는 목적
- 정치적인 목적 (본가 Bert-VITS2에서 금지되어 있습니다)
- 타인을 해치는 목적
- 사칭·딥페이크 제작 목적

### 지켜 주었으면 하는 것

- Style-Bert-VITS2를 이용할 때는 사용하는 모델의 이용약관·라이선스를 반드시 확인하고, 존재하는 경우 그에 따라 주세요.
- 또한 소스 코드를 이용할 때는 [리포지토리 라이선스](https://github.com/litagin02/Style-Bert-VITS2#license)를 따라 주세요.

아래는 기본으로 포함된 모델들의 라이선스입니다.

### JVNV 코퍼스 (jvnv-F1-jp, jvnv-F2-jp, jvnv-M1-jp, jvnv-M2-jp)

- [JVNV 코퍼스](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus)의 라이선스는 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.ja)이므로 이를 계승합니다.

### 코하루네 아미 (koharune-ami) / 아미타로 (amitaro)

[아미타로의 목소리 소재 공방 규약](https://amitaro.net/voice/voice_rule/)과 [아미타로 라이브 방송 음성 이용약관](https://amitaro.net/voice/livevoice/#index_id6)을 모두 지켜야 합니다. 특히 다음 사항을 준수해 주세요 (규약을 지키면 상업·비상업을 불문하고 이용할 수 있습니다):

#### 금지 사항

- 연령 제한이 있는 작품·용도에 사용
- 신흥 종교·정치·다단계 등에 깊이 관련된 작품·용도
- 특정 단체·개인·국가를 비방 중상하는 작품·용도
- 생성된 음성을 아미타로 본인의 목소리로 취급하는 것
- 생성된 음성을 아미타로 이외의 사람의 목소리로 취급하는 것

#### 크레딧 표기

생성 음성을 공개할 때는 (매체 불문) 반드시 알아보기 쉬운 곳에 `あみたろの声素材工房 (https://amitaro.net/)`의 목소리를 바탕으로 한 음성 모델을 사용했음을 알 수 있는 크레딧 표기를 기재해 주세요.

크레딧 표기 예:
- `Style-BertVITS2モデル: 小春音アミ、あみたろの声素材工房 (https://amitaro.net/)`
- `Style-BertVITS2モデル: あみたろ、あみたろの声素材工房 (https://amitaro.net/)`

#### 모델 머지

모델 머지에 관해서는 [아미타로의 목소리 소재 공방 FAQ 답변](https://amitaro.net/voice/faq/#index_id17)을 준수해 주세요:
- 본 모델을 다른 모델과 머지할 수 있는 것은, 그 다른 모델 제작 시 학습에 사용된 목소리의 권리자가 허락한 경우에 한함
- 아미타로 목소리의 특징이 남아 있는 경우 (머지 비율이 25% 이상인 경우), 그 이용은 [아미타로의 목소리 소재 공방 규약](https://amitaro.net/voice/voice_rule/) 범위 내로 한정되며, 해당 모델에도 이 규약이 적용됨
"""

how_to_md = """
아래와 같이 `model_assets` 디렉터리 안에 모델 파일들을 배치해 주세요.
```
model_assets
├── your_model
│   ├── config.json
│   ├── your_model_file1.safetensors
│   ├── your_model_file2.safetensors
│   ├── ...
│   └── style_vectors.npy
└── another_model
    ├── ...
```
각 모델에는 다음 파일들이 필요합니다:
- `config.json`: 학습 시 설정 파일
- `*.safetensors`: 학습된 모델 파일 (1개 이상 필요, 여러 개 가능)
- `style_vectors.npy`: 스타일 벡터 파일

위 2개는 `Train.bat` 학습으로 자동으로 올바른 위치에 저장됩니다. `style_vectors.npy`는 `Style.bat`을 실행해 지시에 따라 생성해 주세요.
"""

style_md = f"""
- 프리셋 또는 음성 파일로 낭독의 음색·감정·스타일 같은 것을 제어할 수 있습니다.
- 기본값인 {DEFAULT_STYLE}로도 읽는 문장에 맞는 감정으로 충분히 풍부하게 낭독됩니다. 이 스타일 제어는 그것을 가중치를 두고 덮어쓰는 느낌입니다.
- 강도를 너무 크게 하면 발음이 이상해지거나 목소리가 되지 않는 등 무너질 수 있습니다.
- 어느 정도 강도가 좋은지는 모델·스타일에 따라 다른 것 같습니다.
- 음성 파일을 입력하는 경우, 학습 데이터와 비슷한 음색의 화자 (특히 같은 성별)가 아니면 좋은 효과가 나지 않을 수 있습니다.
"""
voice_keys = ["dec"]
voice_pitch_keys = ["flow"]
speech_style_keys = ["enc_p"]
tempo_keys = ["sdp", "dp"]


def make_interactive():
    return gr.update(interactive=True, value="음성 합성")


def make_non_interactive():
    return gr.update(interactive=False, value="음성 합성 (모델을 로드해 주세요)")


def gr_util(item):
    if item == "프리셋에서 선택":
        return (gr.update(visible=True), gr.Audio(visible=False, value=None))
    else:
        return (gr.update(visible=False), gr.update(visible=True))


null_models_frame = 0


def change_null_model_row(
    null_model_index: int,
    null_model_name: str,
    null_model_path: str,
    null_voice_weights: float,
    null_voice_pitch_weights: float,
    null_speech_style_weights: float,
    null_tempo_weights: float,
    null_models: dict[int, NullModelParam],
):
    null_models[null_model_index] = NullModelParam(
        name=null_model_name,
        path=Path(null_model_path),
        weight=null_voice_weights,
        pitch=null_voice_pitch_weights,
        style=null_speech_style_weights,
        tempo=null_tempo_weights,
    )
    if len(null_models) > null_models_frame:
        keys_to_keep = list(range(null_models_frame))
        result = {k: null_models[k] for k in keys_to_keep}
    else:
        result = null_models
    return result, True


def create_inference_app(model_holder: TTSModelHolder) -> gr.Blocks:
    def tts_fn(
        model_name,
        model_path,
        text,
        language,
        reference_audio_path,
        sdp_ratio,
        noise_scale,
        noise_scale_w,
        length_scale,
        line_split,
        split_interval,
        assist_text,
        assist_text_weight,
        use_assist_text,
        style,
        style_weight,
        kata_tone_json_str,
        use_tone,
        speaker,
        pitch_scale,
        intonation_scale,
        null_models: dict[int, NullModelParam],
        force_reload_model: bool,
    ):
        model_holder.get_model(model_name, model_path)
        assert model_holder.current_model is not None
        logger.debug(f"Null models setting: {null_models}")

        wrong_tone_message = ""
        kata_tone: Optional[list[tuple[str, int]]] = None
        if use_tone and kata_tone_json_str != "":
            if language != "JP":
                logger.warning("Only Japanese is supported for tone generation.")
                wrong_tone_message = "악센트 지정은 현재 일본어만 지원합니다."
            if line_split:
                logger.warning("Tone generation is not supported for line split.")
                wrong_tone_message = (
                    "악센트 지정은 줄바꿈 단위 생성을 사용하지 않는 경우에만 지원됩니다."
                )
            try:
                kata_tone = []
                json_data = json.loads(kata_tone_json_str)
                # tupleを使うように変換
                for kana, tone in json_data:
                    assert isinstance(kana, str) and tone in (0, 1), f"{kana}, {tone}"
                    kata_tone.append((kana, tone))
            except Exception as e:
                logger.warning(f"Error occurred when parsing kana_tone_json: {e}")
                wrong_tone_message = f"악센트 지정이 잘못되었습니다: {e}"
                kata_tone = None

        # toneは実際に音声合成に代入される際のみnot Noneになる
        tone: Optional[list[int]] = None
        if kata_tone is not None:
            phone_tone = kata_tone2phone_tone(kata_tone)
            tone = [t for _, t in phone_tone]

        speaker_id = model_holder.current_model.spk2id[speaker]

        start_time = datetime.datetime.now()

        try:
            sr, audio = model_holder.current_model.infer(
                text=text,
                language=language,
                reference_audio_path=reference_audio_path,
                sdp_ratio=sdp_ratio,
                noise=noise_scale,
                noise_w=noise_scale_w,
                length=length_scale,
                line_split=line_split,
                split_interval=split_interval,
                assist_text=assist_text,
                assist_text_weight=assist_text_weight,
                use_assist_text=use_assist_text,
                style=style,
                style_weight=style_weight,
                given_tone=tone,
                speaker_id=speaker_id,
                pitch_scale=pitch_scale,
                intonation_scale=intonation_scale,
                null_model_params=null_models,
                force_reload_model=force_reload_model,
            )
        except InvalidToneError as e:
            logger.error(f"Tone error: {e}")
            return f"Error: 악센트 지정이 잘못되었습니다:\n{e}", None, kata_tone_json_str
        except ValueError as e:
            logger.error(f"Value error: {e}")
            return f"Error: {e}", None, kata_tone_json_str

        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()

        if tone is None and language == "JP":
            # アクセント指定に使えるようにアクセント情報を返す
            norm_text = normalize_text(text)
            kata_tone = g2kata_tone(norm_text)
            kata_tone_json_str = json.dumps(kata_tone, ensure_ascii=False)
        elif tone is None:
            kata_tone_json_str = ""
        message = f"Success, time: {duration} seconds."
        if wrong_tone_message != "":
            message = wrong_tone_message + "\n" + message
        return message, (sr, audio), kata_tone_json_str, False

    def get_model_files(model_name: str):
        return [str(f) for f in model_holder.model_files_dict[model_name]]

    model_names = model_holder.model_names
    if len(model_names) == 0:
        logger.error(
            f"モデルが見つかりませんでした。{model_holder.root_dir}にモデルを置いてください。"
        )
        with gr.Blocks() as app:
            gr.Markdown(
                f"Error: 모델을 찾을 수 없습니다. {model_holder.root_dir}에 모델을 배치해 주세요."
            )
        return app
    initial_id = 0
    initial_pth_files = get_model_files(model_names[initial_id])

    with gr.Blocks(theme=GRADIO_THEME) as app:
        gr.Markdown(initial_md)
        gr.Markdown(terms_of_use_md)
        null_models = gr.State({})
        force_reload_model = gr.State(False)
        with gr.Accordion(label="사용법", open=False):
            gr.Markdown(how_to_md)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=3):
                        model_name = gr.Dropdown(
                            label="모델 목록",
                            choices=model_names,
                            value=model_names[initial_id],
                        )
                        model_path = gr.Dropdown(
                            label="모델 파일",
                            choices=initial_pth_files,
                            value=initial_pth_files[0],
                        )
                    refresh_button = gr.Button("새로고침", scale=1, visible=True)
                    load_button = gr.Button("로드", scale=1, variant="primary")
                text_input = gr.TextArea(label="텍스트", value=initial_text)
                pitch_scale = gr.Slider(
                    minimum=0.8,
                    maximum=1.5,
                    value=1,
                    step=0.05,
                    label="음높이 (1 이외에는 음질 저하)",
                )
                intonation_scale = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=1,
                    step=0.1,
                    label="억양 (1 이외에는 음질 저하)",
                )

                line_split = gr.Checkbox(
                    label="줄바꿈 단위로 나눠서 생성 (나누는 쪽이 감정이 더 잘 실립니다)",
                    value=DEFAULT_LINE_SPLIT,
                )
                split_interval = gr.Slider(
                    minimum=0.0,
                    maximum=2,
                    value=DEFAULT_SPLIT_INTERVAL,
                    step=0.1,
                    label="줄바꿈마다 넣는 무음 길이 (초)",
                )
                line_split.change(
                    lambda x: (gr.Slider(visible=x)),
                    inputs=[line_split],
                    outputs=[split_interval],
                )
                tone = gr.Textbox(
                    label="악센트 조정 (숫자는 0=낮음, 1=높음만, 일본어 전용)",
                    info="줄바꿈으로 나누지 않는 경우에만 사용할 수 있습니다. 만능은 아닙니다.",
                )
                use_tone = gr.Checkbox(label="악센트 조정 사용", value=False)
                use_tone.change(
                    lambda x: (gr.Checkbox(value=False) if x else gr.Checkbox()),
                    inputs=[use_tone],
                    outputs=[line_split],
                )
                language = gr.Dropdown(choices=languages, value="JP", label="언어")
                speaker = gr.Dropdown(label="화자")
                with gr.Accordion(label="상세 설정", open=False):
                    sdp_ratio = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=DEFAULT_SDP_RATIO,
                        step=0.1,
                        label="SDP Ratio",
                    )
                    noise_scale = gr.Slider(
                        minimum=0.1,
                        maximum=2,
                        value=DEFAULT_NOISE,
                        step=0.1,
                        label="Noise",
                    )
                    noise_scale_w = gr.Slider(
                        minimum=0.1,
                        maximum=2,
                        value=DEFAULT_NOISEW,
                        step=0.1,
                        label="Noise_W",
                    )
                    length_scale = gr.Slider(
                        minimum=0.1,
                        maximum=2,
                        value=DEFAULT_LENGTH,
                        step=0.1,
                        label="Length",
                    )
                    use_assist_text = gr.Checkbox(
                        label="Assist text 사용", value=False
                    )
                    assist_text = gr.Textbox(
                        label="Assist text",
                        placeholder="어째서 내 의견을 무시하는 거야? 용서 못 해, 짜증 나! 죽어버렸으면 좋겠어.",
                        info="이 텍스트를 읽었을 때와 비슷한 음색·감정이 되기 쉬워집니다. 대신 억양·템포 등이 희생되는 경향이 있습니다.",
                        visible=False,
                    )
                    assist_text_weight = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=DEFAULT_ASSIST_TEXT_WEIGHT,
                        step=0.1,
                        label="Assist text 강도",
                        visible=False,
                    )
                    use_assist_text.change(
                        lambda x: (gr.Textbox(visible=x), gr.Slider(visible=x)),
                        inputs=[use_assist_text],
                        outputs=[assist_text, assist_text_weight],
                    )
                with gr.Accordion(label="널 모델", open=False):
                    with gr.Row():
                        null_models_count = gr.Number(
                            label="널 모델 수", value=0, step=1
                        )
                    with gr.Column(variant="panel"):

                        @gr.render(inputs=[null_models_count])
                        def render_null_models(
                            null_models_count: int,
                        ):
                            global null_models_frame
                            null_models_frame = null_models_count
                            for i in range(null_models_count):
                                with gr.Row():
                                    null_model_index = gr.Number(
                                        value=i,
                                        key=f"null_model_index_{i}",
                                        visible=False,
                                    )
                                    null_model_name = gr.Dropdown(
                                        label="모델 목록",
                                        choices=model_names,
                                        key=f"null_model_name_{i}",
                                        value=model_names[initial_id],
                                    )
                                    null_model_path = gr.Dropdown(
                                        label="모델 파일",
                                        key=f"null_model_path_{i}",
                                        # FIXME: 再レンダー時に選択肢が消えるのでどうにかしたい
                                        # 現在は再レンダーでvalueは保存されるが選択肢は保存されないので選択肢が空になる
                                        # そのときに選択肢にない値となるので、それを許す
                                        allow_custom_value=True,
                                    )
                                    null_voice_weights = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        value=1,
                                        step=0.1,
                                        key=f"null_voice_weights_{i}",
                                        label="음색",
                                    )
                                    null_voice_pitch_weights = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        value=1,
                                        step=0.1,
                                        key=f"null_voice_pitch_weights_{i}",
                                        label="목소리 높이",
                                    )
                                    null_speech_style_weights = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        value=1,
                                        step=0.1,
                                        key=f"null_speech_style_weights_{i}",
                                        label="말투",
                                    )
                                    null_tempo_weights = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        value=1,
                                        step=0.1,
                                        key=f"null_tempo_weights_{i}",
                                        label="템포",
                                    )

                                    null_model_name.change(
                                        model_holder.update_model_files_for_gradio,
                                        inputs=[null_model_name],
                                        outputs=[null_model_path],
                                    )
                                    null_model_path.change(
                                        make_non_interactive, outputs=[tts_button]
                                    )
                                    # 愚直すぎるのでもう少しなんとかしたい
                                    null_model_path.change(
                                        change_null_model_row,
                                        inputs=[
                                            null_model_index,
                                            null_model_name,
                                            null_model_path,
                                            null_voice_weights,
                                            null_voice_pitch_weights,
                                            null_speech_style_weights,
                                            null_tempo_weights,
                                            null_models,
                                        ],
                                        outputs=[null_models, force_reload_model],
                                    )
                                    null_voice_weights.change(
                                        change_null_model_row,
                                        inputs=[
                                            null_model_index,
                                            null_model_name,
                                            null_model_path,
                                            null_voice_weights,
                                            null_voice_pitch_weights,
                                            null_speech_style_weights,
                                            null_tempo_weights,
                                            null_models,
                                        ],
                                        outputs=[null_models, force_reload_model],
                                    )
                                    null_voice_pitch_weights.change(
                                        change_null_model_row,
                                        inputs=[
                                            null_model_index,
                                            null_model_name,
                                            null_model_path,
                                            null_voice_weights,
                                            null_voice_pitch_weights,
                                            null_speech_style_weights,
                                            null_tempo_weights,
                                            null_models,
                                        ],
                                        outputs=[null_models, force_reload_model],
                                    )
                                    null_speech_style_weights.change(
                                        change_null_model_row,
                                        inputs=[
                                            null_model_index,
                                            null_model_name,
                                            null_model_path,
                                            null_voice_weights,
                                            null_voice_pitch_weights,
                                            null_speech_style_weights,
                                            null_tempo_weights,
                                            null_models,
                                        ],
                                        outputs=[null_models, force_reload_model],
                                    )
                                    null_tempo_weights.change(
                                        change_null_model_row,
                                        inputs=[
                                            null_model_index,
                                            null_model_name,
                                            null_model_path,
                                            null_voice_weights,
                                            null_voice_pitch_weights,
                                            null_speech_style_weights,
                                            null_tempo_weights,
                                            null_models,
                                        ],
                                        outputs=[null_models, force_reload_model],
                                    )

                    add_btn = gr.Button("널 모델 추가")
                    del_btn = gr.Button("널 모델 제거")
                    add_btn.click(
                        lambda x: x + 1,
                        inputs=[null_models_count],
                        outputs=[null_models_count],
                    )
                    del_btn.click(
                        lambda x: x - 1 if x > 0 else 0,
                        inputs=[null_models_count],
                        outputs=[null_models_count],
                    )

            with gr.Column():
                with gr.Accordion("스타일 상세 설명", open=False):
                    gr.Markdown(style_md)
                style_mode = gr.Radio(
                    ["프리셋에서 선택", "음성 파일 입력"],
                    label="스타일 지정 방법",
                    value="프리셋에서 선택",
                )
                style = gr.Dropdown(
                    label=f"스타일 ({DEFAULT_STYLE}이 평균 스타일)",
                    choices=["모델을 로드해 주세요"],
                    value="모델을 로드해 주세요",
                )
                style_weight = gr.Slider(
                    minimum=0,
                    maximum=20,
                    value=DEFAULT_STYLE_WEIGHT,
                    step=0.1,
                    label="스타일 강도 (목소리가 무너지면 줄여 주세요)",
                )
                ref_audio_path = gr.Audio(
                    label="참조 음성", type="filepath", visible=False
                )
                tts_button = gr.Button(
                    "음성 합성 (모델을 로드해 주세요)",
                    variant="primary",
                    interactive=False,
                )
                text_output = gr.Textbox(label="정보")
                audio_output = gr.Audio(label="결과")
                with gr.Accordion("텍스트 예시", open=False):
                    gr.Examples(examples, inputs=[text_input, language])

        tts_button.click(
            tts_fn,
            inputs=[
                model_name,
                model_path,
                text_input,
                language,
                ref_audio_path,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                line_split,
                split_interval,
                assist_text,
                assist_text_weight,
                use_assist_text,
                style,
                style_weight,
                tone,
                use_tone,
                speaker,
                pitch_scale,
                intonation_scale,
                null_models,
                force_reload_model,
            ],
            outputs=[text_output, audio_output, tone, force_reload_model],
        )

        model_name.change(
            model_holder.update_model_files_for_gradio,
            inputs=[model_name],
            outputs=[model_path],
        )

        model_path.change(make_non_interactive, outputs=[tts_button])

        refresh_button.click(
            model_holder.update_model_names_for_gradio,
            outputs=[model_name, model_path, tts_button],
        )

        load_button.click(
            model_holder.get_model_for_gradio,
            inputs=[model_name, model_path],
            outputs=[style, tts_button, speaker],
        )

        style_mode.change(
            gr_util,
            inputs=[style_mode],
            outputs=[style, ref_audio_path],
        )

    return app


if __name__ == "__main__":
    import torch

    from config import get_path_config

    path_config = get_path_config()
    assets_root = path_config.assets_root
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_holder = TTSModelHolder(
        assets_root, device, torch_device_to_onnx_providers(device)
    )
    app = create_inference_app(model_holder)
    app.launch(inbrowser=True)
