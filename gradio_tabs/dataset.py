import gradio as gr

from style_bert_vits2.constants import GRADIO_THEME
from style_bert_vits2.logging import logger
from style_bert_vits2.utils.subprocess import run_script_with_log


def do_slice(
    model_name: str,
    min_sec: float,
    max_sec: float,
    min_silence_dur_ms: int,
    time_suffix: bool,
    input_dir: str,
):
    if model_name == "":
        return "Error: 모델명을 입력해 주세요."
    logger.info("Start slicing...")
    cmd = [
        "slice.py",
        "--model_name",
        model_name,
        "--min_sec",
        str(min_sec),
        "--max_sec",
        str(max_sec),
        "--min_silence_dur_ms",
        str(min_silence_dur_ms),
    ]
    if time_suffix:
        cmd.append("--time_suffix")
    if input_dir != "":
        cmd += ["--input_dir", input_dir]
    # onnxの警告が出るので無視する
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        return f"Error: {message}"
    return "음성 슬라이스가 완료되었습니다."


def do_transcribe(
    model_name,
    whisper_model,
    compute_type,
    language,
    initial_prompt,
    use_hf_whisper,
    batch_size,
    num_beams,
    hf_repo_id,
):
    if model_name == "":
        return "Error: 모델명을 입력해 주세요."
    if hf_repo_id == "litagin/anime-whisper":
        logger.info(
            "Since litagin/anime-whisper does not support initial prompt, it will be ignored."
        )
        initial_prompt = ""

    cmd = [
        "transcribe.py",
        "--model_name",
        model_name,
        "--model",
        whisper_model,
        "--compute_type",
        compute_type,
        "--language",
        language,
        "--initial_prompt",
        f'"{initial_prompt}"',
        "--num_beams",
        str(num_beams),
    ]
    if use_hf_whisper:
        cmd.append("--use_hf_whisper")
        cmd.extend(["--batch_size", str(batch_size)])
        if hf_repo_id != "openai/whisper":
            cmd.extend(["--hf_repo_id", hf_repo_id])
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        return f"Error: {message}. 에러 메시지가 비어 있는 경우 아무 문제가 없을 수도 있으니, 전사 파일을 확인해 문제없으면 무시해 주세요."
    return "음성 전사(문자 변환)가 완료되었습니다."


how_to_md = """
Style-Bert-VITS2 학습용 데이터셋을 만들기 위한 도구입니다. 다음 2가지로 구성됩니다.

- 주어진 음성에서 적당한 길이의 발화 구간을 잘라내는 슬라이스
- 음성에 대한 전사 (문자 변환)

둘 다 사용해도 되고, 슬라이스할 필요가 없는 경우 후자만 사용해도 됩니다. **코퍼스 음원 등 이미 적당한 길이의 음성 파일이 있는 경우 슬라이스는 불필요**합니다.

## 필요한 것

학습하고 싶은 음성이 담긴 음성 파일 몇 개 (형식은 wav 외에도 mp3 등 일반적인 음성 파일 형식이면 가능).
합계 시간이 어느 정도는 있는 편이 좋을 수 있으며, 10분 정도로도 괜찮았다는 보고가 있습니다. 단일 파일이든 여러 파일이든 상관없습니다.

## 슬라이스 사용법
1. `inputs` 폴더에 음성 파일을 모두 넣는다 (스타일을 나누고 싶은 경우, 하위 폴더에 스타일별로 음성을 나눠서 넣는다)
2. `모델명`을 입력하고, 필요하면 설정을 조정한 뒤 `슬라이스 실행` 버튼을 누른다
3. 완성된 음성 파일들은 `Data/{모델명}/raw`에 저장된다

## 전사 사용법

1. `Data/{모델명}/raw`에 음성 파일이 들어 있는지 확인 (바로 아래가 아니어도 됨)
2. 필요하면 설정을 조정한 뒤 버튼을 누른다
3. 전사 파일은 `Data/{모델명}/esd.list`에 저장된다

## 주의

- ~~너무 긴 (12-15초 이상?) wav 파일은 학습에 사용되지 않는 것 같습니다. 너무 짧아도 좋지 않을 수 있습니다.~~ 이 제한은 Ver 2.5에서 학습 시 「커스텀 배치 샘플러를 사용하지 않음」을 선택하면 없어졌습니다. 다만 너무 긴 음성이 있으면 VRAM 소비량이 늘거나 불안정해질 수 있으므로, 적당한 길이로 슬라이스하는 것을 권장합니다.
- 전사 결과를 어느 정도 수정해야 하는지는 데이터셋에 따라 다릅니다.
"""


def create_dataset_app() -> gr.Blocks:
    with gr.Blocks(theme=GRADIO_THEME) as app:
        gr.Markdown(
            "**이미 1파일 2-12초 정도의 음성 파일 모음과 그 전사 데이터가 있는 경우, 이 탭을 사용하지 않고 학습할 수 있습니다.**"
        )
        with gr.Accordion("사용법", open=False):
            gr.Markdown(how_to_md)
        model_name = gr.Textbox(
            label="모델명을 입력해 주세요 (화자명으로도 사용됩니다)."
        )
        with gr.Accordion("음성 슬라이스"):
            gr.Markdown(
                "**이미 적당한 길이의 음성 파일로 이루어진 데이터가 있는 경우, 그 음성을 Data/{모델명}/raw에 넣으면 이 단계는 불필요합니다.**"
            )
            with gr.Row():
                with gr.Column():
                    input_dir = gr.Textbox(
                        label="원본 음성이 들어 있는 폴더 경로",
                        value="inputs",
                        info="아래 폴더에 wav나 mp3 등의 파일을 넣어 두세요",
                    )
                    min_sec = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=2,
                        step=0.5,
                        label="이 초수 미만은 버림",
                    )
                    max_sec = gr.Slider(
                        minimum=0,
                        maximum=15,
                        value=12,
                        step=0.5,
                        label="이 초수 이상은 버림",
                    )
                    min_silence_dur_ms = gr.Slider(
                        minimum=0,
                        maximum=2000,
                        value=700,
                        step=100,
                        label="무음으로 간주해 구분하는 최소 무음 길이 (ms)",
                    )
                    time_suffix = gr.Checkbox(
                        value=False,
                        label="WAV 파일명 끝에 원본 파일의 시간 범위를 붙임",
                    )
                    slice_button = gr.Button("슬라이스 실행")
                result1 = gr.Textbox(label="결과")
        with gr.Row():
            with gr.Column():
                use_hf_whisper = gr.Checkbox(
                    label="HuggingFace의 Whisper 사용 (속도가 빠르지만 VRAM을 많이 사용)",
                    value=False,
                )
                whisper_model = gr.Dropdown(
                    [
                        "large",
                        "large-v2",
                        "large-v3",
                    ],
                    label="Whisper 모델",
                    value="large-v3",
                    visible=True,
                )
                hf_repo_id = gr.Dropdown(
                    [
                        "openai/whisper-large-v3-turbo",
                        "openai/whisper-large-v3",
                        "openai/whisper-large-v2",
                        "kotoba-tech/kotoba-whisper-v2.1",
                        "litagin/anime-whisper",
                    ],
                    label="HuggingFace의 Whisper repo_id",
                    value="openai/whisper-large-v3-turbo",
                    visible=False,
                )
                compute_type = gr.Dropdown(
                    [
                        "int8",
                        "int8_float32",
                        "int8_float16",
                        "int8_bfloat16",
                        "int16",
                        "float16",
                        "bfloat16",
                        "float32",
                    ],
                    label="계산 정밀도",
                    value="bfloat16",
                    visible=True,
                )
                batch_size = gr.Slider(
                    minimum=1,
                    maximum=128,
                    value=16,
                    step=1,
                    label="배치 크기",
                    info="크게 하면 속도가 빨라지지만 VRAM을 많이 사용",
                    visible=False,
                )
                language = gr.Dropdown(["ja", "en", "zh"], value="ja", label="언어")
                initial_prompt = gr.Textbox(
                    label="초기 프롬프트",
                    value="こんにちは。元気、ですかー？ふふっ、私は……ちゃんと元気だよ！",
                    info="이렇게 전사해 달라는 예문 (문장부호 사용법·웃음 표기·고유명사 등. 언어에 맞는 예문을 입력하세요)",
                )
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    label="빔 서치의 빔 수",
                    info="작을수록 속도가 빨라짐 (이전에는 5)",
                )
            transcribe_button = gr.Button("음성 전사")
            result2 = gr.Textbox(label="결과")
        slice_button.click(
            do_slice,
            inputs=[
                model_name,
                min_sec,
                max_sec,
                min_silence_dur_ms,
                time_suffix,
                input_dir,
            ],
            outputs=[result1],
        )
        transcribe_button.click(
            do_transcribe,
            inputs=[
                model_name,
                whisper_model,
                compute_type,
                language,
                initial_prompt,
                use_hf_whisper,
                batch_size,
                num_beams,
                hf_repo_id,
            ],
            outputs=[result2],
        )
        use_hf_whisper.change(
            lambda x: (
                gr.update(visible=not x),
                gr.update(visible=x),
                gr.update(visible=x),
                gr.update(visible=not x),
            ),
            inputs=[use_hf_whisper],
            outputs=[whisper_model, hf_repo_id, batch_size, compute_type],
        )

    return app


if __name__ == "__main__":
    app = create_dataset_app()
    app.launch(inbrowser=True)
