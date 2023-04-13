import os

from contextlib import nullcontext
import gradio as gr
import torch
from torch import autocast
from diffusers import DiffusionPipeline
from transformers import (
    pipeline,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
)

import utils

device = "cuda" if torch.cuda.is_available() else "cpu"
device_dict = {"cuda": 0, "cpu": -1}
context = autocast if device == "cuda" else nullcontext
dtype = torch.float16 if device == "cuda" else torch.float32

# Detect if code is running in Colab
is_colab = utils.is_google_colab()
colab_instruction = "" if is_colab else """
<p>You can skip the queue using Colab: 
<a href="https://colab.research.google.com/drive/1nhXyddThldnxPfIYO2my_bYinlMUW30R?usp=sharing">
<img data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a></p>"""
device_print = "GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"

# Add language detection pipeline
language_detection_model_ckpt = "papluca/xlm-roberta-base-language-detection"
language_detection_pipeline = pipeline("text-classification",
                                       model=language_detection_model_ckpt,
                                       device=device_dict[device])

# Add model for language translation
trans_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
trans_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-one-mmt").to(device)

model_id = "CompVis/stable-diffusion-v1-4"

if is_colab:
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        custom_pipeline="multilingual_stable_diffusion",
        detection_pipeline=language_detection_pipeline,
        translation_model=trans_model,
        translation_tokenizer=trans_tokenizer,
        revision="fp16",
        torch_dtype=dtype,
    )
else:
    import streamlit as st
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        custom_pipeline="multilingual_stable_diffusion",
        use_auth_token=os.environ["USER_TOKEN"],
        detection_pipeline=language_detection_pipeline,
        translation_model=trans_model,
        translation_tokenizer=trans_tokenizer,
        revision="fp16",
        torch_dtype=dtype,
    )

pipe = pipe.to(device)

#torch.backends.cudnn.benchmark = True
num_samples = 2

def infer(prompt, steps, scale):
    
    with context("cuda"):
        print(prompt)
        images = pipe(num_samples*[prompt],
                      guidance_scale=scale,
                      num_inference_steps=int(steps)).images

    return images

css = """
        a {
            color: inherit;
            text-decoration: underline;
        }
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: #0000FF;
            background: #0000FF;
        }
        input[type='range'] {
            accent-color: #0000FF;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }
        #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        #advanced-btn {
            font-size: .7rem !important;
            line-height: 19px;
            margin-top: 12px;
            margin-bottom: 12px;
            padding: 2px 8px;
            border-radius: 14px !important;
        }
        #advanced-options {
            margin-bottom: 20px;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
        #container-advanced-btns{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        #share-btn-container {
            display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
        }
        #share-btn {
            all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;
        }
        #share-btn * {
            all: unset;
        }
        .gr-form{
            flex: 1 1 50%; border-top-right-radius: 0; border-bottom-right-radius: 0;
        }
        #prompt-container{
            gap: 0;
        }
        #generated_id{
            min-height: 700px
        }
        .logo-img {
            float: left;
        }
        nav {
            float: right;
        }
"""
block = gr.Blocks(css=css)

examples = [
    [
        'נמר לבן הולך על חוף הים, שקיעה, צבעים חזקים, צלליות, רזלוציה גבוהה, מאוד מפורט ומדוייק, ריאליסטי',
        50,
        7.5,
    ],
    [
        '一隻狗在天堂',
        45,
        7.5,
    ],
    [
        'Una casa en la playa en un atardecer lluvioso',
        45,
        7.5,
    ],
    [
        'Ein Hund, der Orange isst',
        45,
        7.5,
    ],
    [
        "Photo d'un restaurant parisien",
        45,
        7.5,
    ],
    [
        "Franču restorāna fotogrāfija",
        45,
        7.5,
    ],
    [
        "పారిసియన్ రెస్టారెంట్ యొక్క ఫోటో",
        45,
        7.5,
    ],
    [
        "صورة لمطعم باريسي",
        45,
        7.5,
    ],
]

with block as demo:
    gr.HTML(
        f"""
            <div style="text-align: center; max-width: 650px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <img class="logo-img" src="vinai.png" ALT="align box" ALIGN=CENTER>
                <h1 style="font-weight: 900; margin-bottom: 7px;">
                  Multilingual Stable Diffusion
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
                Stable Diffusion Pipeline that supports Vietnamese.
              </p>
              <p style="margin-bottom: 10px; font-size: 94%">
                {colab_instruction}
                Running on <b>{device_print}</b>{(" in a <b>Google Colab</b>." if is_colab else "")}
              </p>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():
            with gr.Row().style(mobile_collapse=False, equal_height=True):

                text = gr.Textbox(
                    label="Enter your prompt", show_label=False, max_lines=1
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )
                btn = gr.Button("Run").style(
                    margin=False,
                    rounded=(False, True, True, False),
                )
               
        gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery").style(
            grid=[2], height="auto"
        )
        
        with gr.Row(elem_id="advanced-options"):
            steps = gr.Slider(label="Steps", minimum=5, maximum=50, value=45, step=5)
            scale = gr.Slider(
                label="Guidance Scale", minimum=0, maximum=50, value=7.5, step=0.1
            )
        
        ex = gr.Examples(examples=examples, fn=infer, inputs=[text, steps, scale], outputs=gallery, cache_examples=False)
        ex.dataset.headers = [""]
        
        text.submit(infer, inputs=[text, steps, scale], outputs=gallery)
        btn.click(infer, inputs=[text, steps, scale], outputs=gallery)

    gr.HTML(
            """
                <div class="footer">
                    <p>Stable Diffusion model that supports multiple languages by <a href="https://huggingface.co/juancopi81" style="text-decoration: underline;" target="_blank">juancopi81</a>
                    </p>
                </div>
                <div class="acknowledgments">
                    <p><h4>LICENSE</h4>
The model is licensed with a <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" style="text-decoration: underline;" target="_blank">CreativeML Open RAIL-M</a> license. The authors claim no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in this license. The license forbids you from sharing any content that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation and target vulnerable groups. For the full list of restrictions please <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" target="_blank" style="text-decoration: underline;" target="_blank">read the license</a></p>
                    <p><h4>Biases and content acknowledgment</h4>
Despite how impressive being able to turn text into image is, beware to the fact that this model may output content that reinforces or exacerbates societal biases, as well as realistic faces, pornography and violence. The model was trained on the <a href="https://laion.ai/blog/laion-5b/" style="text-decoration: underline;" target="_blank">LAION-5B dataset</a>, which scraped non-curated image-text-pairs from the internet (the exception being the removal of illegal content) and is meant for research purposes. You can read more in the <a href="https://huggingface.co/CompVis/stable-diffusion-v1-4" style="text-decoration: underline;" target="_blank">model card</a></p>
               </div>
           """
        )
    gr.Markdown('''
      [![Twitter Follow](https://img.shields.io/twitter/follow/juancopi81?style=social)](https://twitter.com/juancopi81)
      ![visitors](https://visitor-badge.glitch.me/badge?page_id=Juancopi81.MultilingualStableDiffusion)
    ''')

if not is_colab:
    demo.queue(concurrency_count=1)
demo.launch(debug=is_colab, share=is_colab)
