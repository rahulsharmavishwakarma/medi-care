import argparse
from collections import defaultdict
import datetime
import json
import os
import time

import gradio as gr
import requests

from llava.serve.lang_chain_try import langchain_lokha
from llava.conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)
from llava.constants import LOGDIR
from llava.utils import (build_logger, server_error_msg,
    violates_moderation, moderation_msg)
# from llava.serve.gradio_patch import Chatbot as grChatbot
from llava.serve.gradio_css import code_highlight_css
import hashlib


logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "LLaVA Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown(
                value=model, visible=True)

    state = default_conversation.copy()
    return (state,
            dropdown_update,
            gr.Chatbot(visible=True),
            gr.Textbox(visible=True),
            gr.Button(visible=True),
            gr.Row(visible=True),
            gr.Accordion(visible=True))


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    return (state, gr.Dropdown(
               choices=models,
               value=models[0] if len(models) > 0 else ""),
            gr.Chatbot(visible=True),
            gr.Textbox(visible=True),
            gr.Button(visible=True),
            gr.Row(visible=True),
            gr.Accordion(visible=True))


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, image_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def add_text(state, text, image, image_process_mode, request: gr.Request):
    state.user_query = text
    if image is not None:
        state.user_image = image
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg, None) + (
                no_change_btn,) * 5

    text = text[:1536]  # Hard cut-off
    if image is not None:
        multimodal_msg = None
        text = text[:1200]  # Hard cut-off for images
        if '<image>' not in text:
            text = text + '\n<image>'

        if multimodal_msg is not None:
            return (state, state.to_gradio_chatbot(), multimodal_msg, None,) + (
                no_change_btn,) * 5
        text = (text, image, image_process_mode)
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code


def http_bot(state, model_selector, temperature, max_new_tokens, image_process_mode, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector
    logger.info(f"Text is {state.user_query}")
    logger.info(f"Image is {state.user_image}")
    logger.info(f"Image Processing Mode is {image_process_mode}")
    # if len(state.messages[-2][1]) > 1:
    #     user_query = state.messages[-2][1][0].replace('<image>', '')
    # else:
    #     user_query =  state.messages[-2][1].replace('<image>', '')
    # logger.info(f"Text Box is {user_query}")
    # logger.info(f"state Messages is {state.messages[-2]}")
    if '@med' in state.user_query:
        # all_images = state.get_images(return_pil=True)
        if state.user_image is  None:
            respond = "Please provide an image for the question."
            state.messages[-1][-1] = ''
            respond_char = ''
            for char in respond:
                respond_char += char
                state.messages[-1][-1] = respond_char + "▌"
                yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
            return
        else:
            output = langchain_lokha(state, state.user_query, state.user_image, image_process_mode)
            state.messages[-1][-1] = ''
            output_char = ''
            for character in output:
                output_char += character
                state.messages[-1][-1] = output_char + "▌"
                time.sleep(0.05)
                yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

        state.messages[-1][-1] = state.messages[-1][-1][:-1]
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5 
    else:
        if state.skip_next:
            # This generate call is skipped due to invalid inputs
            yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
            return

        if len(state.messages) == state.offset + 2:
            # First round of conversation
            if "llava" in model_name.lower():
                if "v1" in model_name:
                    template_name = "llava_v1"
                else:
                    template_name = "multimodal"
            elif "koala" in model_name: # Hardcode the condition
                template_name = "bair_v1"
            elif "v1" in model_name:    # vicuna v1_1/v1_2
                template_name = "vicuna_v1_1"
            else:
                template_name = "v1"
            template_name = "multimodal" # FIXME: overwrite
            new_state = conv_templates[template_name].copy()
            new_state.append_message(new_state.roles[0], state.messages[-2][1])
            new_state.append_message(new_state.roles[1], None)
            state = new_state

        # Query worker address
        controller_url = args.controller_url
        ret = requests.post(controller_url + "/get_worker_address",
                json={"model": model_name})
        worker_addr = ret.json()["address"]
        logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

        # No available worker
        if worker_addr == "":
            state.messages[-1][-1] = server_error_msg
            yield (state, state.to_gradio_chatbot(), disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
            return

        # Construct prompt
        prompt = state.get_prompt()

        all_images = state.get_images(return_pil=True)
        # logger.info(f"==== Image Details ====\n{all_images}")
        # try:
        #     if len(all_images) == 0:
        #       logger.info("----------------- The Image is not Present -------------")
        # except:
        #     print('The error occured')

        all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
        for image, hash in zip(all_images, all_image_hash):
            t = datetime.datetime.now()
            filename = os.path.join(LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg")
            if not os.path.isfile(filename):
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                image.save(filename)

        # Make requests
        pload = {
            "model": model_name,
            "prompt": prompt,
            "temperature": float(temperature),
            "max_new_tokens": min(int(max_new_tokens), 1536),
            "stop": state.sep if state.sep_style == SeparatorStyle.SINGLE else state.sep2,
            "images": f'List of {len(state.get_images())} images: {all_image_hash}',
        }
        logger.info(f"==== request ====\n{pload}")

        pload['images'] = state.get_images()

        state.messages[-1][-1] = "▌"
        yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

        try:
            # Stream output
            response = requests.post(worker_addr + "/worker_generate_stream",
                headers=headers, json=pload, stream=True, timeout=500000)
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    data = json.loads(chunk.decode())
                    if data["error_code"] == 0:
                        output = data["text"][len(prompt):].strip()
                        output = post_process_code(output)
                        logger.info(f"==== output ====\n{output}")
                        state.messages[-1][-1] = output + "▌"
                        logger.info(f'==== state.messages[-1][-1] ====\n{state.messages[-1][-1]}')
                        yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                    else:
                        output = data["text"] + f" (error_code: {data['error_code']})"
                        state.messages[-1][-1] = output
                        yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
                        return
                    time.sleep(0.03)
        except requests.exceptions.RequestException as e:
            state.messages[-1][-1] = server_error_msg
            yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
            return

        state.messages[-1][-1] = state.messages[-1][-1][:-1]
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

        finish_tstamp = time.time()
        logger.info(f"{output}")

        with open(get_conv_log_filename(), "a") as fout:
            data = {
                "tstamp": round(finish_tstamp, 4),
                "type": "chat",
                "model": model_name,
                "start": round(start_tstamp, 4),
                "finish": round(start_tstamp, 4),
                "state": state.dict(),
                "images": all_image_hash,
                "ip": request.client.host,
            }
            fout.write(json.dumps(data) + "\n")

title_markdown = ("""
# 🌋 Beyond Imagery: AI-Enhanced Diagnostic Assistant for Cancer and Tumor Diagnosis using Radiology Imaging

[[Paper]](https://www.ijets.in/Downloads/Published/E0202403007.pdf)
""")

tos_markdown = ("""
### Objective
The project aims to improve decision-making and patient outcomes in medical imaging through Visual Question Answering (VQA).
""")


learn_more_markdown = ("""
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
""")


css = code_highlight_css + """
pre {
    white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
"""

def dvilokha(state, textbox, imagebox, image_process_mode, request: gr.Request):
    # user_query = state.message[-2][1][0].replace('<image>', '')
    if '@med' in textbox:
        # all_images = state.get_images(return_pil=True)
        if imagebox is  None:
            respond = "Please provide an image for the question."
            state.messages[-1][-1] += respond + "▌"
            return (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
        else:
            output = langchain_lokha(textbox, imagebox, image_process_mode)
            for character in output:
                state.messages[-1][-1] += character+ "▌"
                time.sleep(0.05)
                yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

        state.messages[-1][-1] = state.messages[-1][-1][:-1]
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5 

def build_demo(embed_mode):
    textbox = gr.Textbox(show_label=False,
        placeholder="Enter text and press ENTER", visible=True)
    with gr.Blocks(title="LLaVA-Med", theme=gr.themes.Base(), css=css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False)

                imagebox = gr.Image(type="pil")
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad"],
                    value="Crop",
                    label="Preprocess for non-square image")

                cur_dir = os.path.dirname(os.path.abspath(__file__))
                gr.Examples(examples=[
                    
                    [f"{cur_dir}/examples/med_img_1.png", "Can you describe the image in details?"],   
                    [f"{cur_dir}/examples/synpic42202.jpg", "Is there evidence of an aortic aneurysm? Please choose from the following two options: [yes, no]?"], # answer" yes 
                    [f"{cur_dir}/examples/synpic32933.jpg", "What is the abnormality by the right hemidiaphragm?"],      # answer: free air                             
                    
                ], inputs=[imagebox, textbox])

                with gr.Accordion("Parameters", open=False, visible=True) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature",)
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)

            with gr.Column(scale=6):
                chatbot = gr.Chatbot(elem_id="chatbot", label="LLaVA-Med Chatbot", visible=True)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=60):
                        submit_btn = gr.Button(value="Submit", visible=True)
                with gr.Row(visible=True) as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    #stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

        if not embed_mode:
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=True)

        # Text and Image
        user_text = textbox
        user_image = imagebox

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        upvote_btn.click(upvote_last_response,
            [state, model_selector], [textbox, upvote_btn, downvote_btn, flag_btn])
        downvote_btn.click(downvote_last_response,
            [state, model_selector], [textbox, upvote_btn, downvote_btn, flag_btn])
        flag_btn.click(flag_last_response,
            [state, model_selector], [textbox, upvote_btn, downvote_btn, flag_btn])
        regenerate_btn.click(regenerate, [state, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list).then(
            http_bot, [state, model_selector, temperature, max_output_tokens, image_process_mode],
            [state, chatbot] + btn_list)
        clear_btn.click(clear_history, None, [state, chatbot, textbox, imagebox] + btn_list)

        textbox.submit(add_text, [state, textbox, imagebox, image_process_mode], [state, chatbot, textbox, imagebox] + btn_list
                ).then(http_bot, [state, model_selector, temperature, max_output_tokens, image_process_mode],
                    [state, chatbot] + btn_list)
        
        submit_btn.click(add_text, [state, textbox, imagebox, image_process_mode], [state, chatbot, textbox, imagebox] + btn_list
                ).then(http_bot, [state, model_selector, temperature, max_output_tokens, image_process_mode],
                    [state, chatbot] + btn_list)

        if args.model_list_mode == "once":
            demo.load(load_demo, [url_params], [state, model_selector,
                chatbot, textbox, submit_btn, button_row, parameter_row],
                js=get_window_url_params)
        elif args.model_list_mode == "reload":
            demo.load(load_demo_refresh_model_list, None, [state, model_selector,
                chatbot, textbox, submit_btn, button_row, parameter_row])
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=8)
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()

    logger.info(args)
    demo = build_demo(args.embed)
    demo.queue(default_concurrency_limit=args.concurrency_count, status_update_rate=10,
               api_open=False).launch(
        server_name=args.host, server_port=args.port, share=args.share)