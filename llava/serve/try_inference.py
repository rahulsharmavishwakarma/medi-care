import argparse
from collections import defaultdict
import datetime
import json
import os
import time
from PIL import Image

# import gradio as gr
import requests

from llava.conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)
from llava.constants import LOGDIR
from llava.utils import (build_logger, server_error_msg,
    violates_moderation, moderation_msg)
# from llava.serve.gradio_patch import Chatbot as grChatbot
# from llava.serve.gradio_css import code_highlight_css
import hashlib


headers = {"User-Agent": "LLaVA Client"}


def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code


def add_text(state, text, image, image_process_mode):
    # print(text)
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return state
    
    
    # if image is not None:
    #     print("ITs not None")

    text = text[:1536]  # Hard cut-off
    if image is not None:
        multimodal_msg = None
        text = text[:1200]  # Hard cut-off for images
        if '<image>' not in text:
            text = text + '\n<image>'

        if multimodal_msg is not None:
            return state
            
        text = (text, image, image_process_mode)
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return state



def http_bot(state, temperature, max_new_tokens):
    model_name = 'LLaVA-Med_weights'   # model_selector  -----> Change to Name of the Model

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        return state

    if len(state.messages) == state.offset + 2:
        template_name = "multimodal" # FIXME: overwrite
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state
    
    # User Query Extraction
    # user_query = state.messages[-2][1][0].replace('<image>', '')
    # print("-------------------")
    # print(user_query)
    # print("-------------------")
    

    # Query worker address
    controller_url = 'http://localhost:10029' # -----> Replace it with Controller URL
    ret = requests.post(controller_url + "/get_worker_address",
            json={"model": model_name})
    worker_addr = ret.json()["address"]

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        return state

    # Construct prompt
    prompt = state.get_prompt()

    all_images = state.get_images(return_pil=True)
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

    pload['images'] = state.get_images()

    state.messages[-1][-1] = "▌"
    message = ''
    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
            headers=headers, json=pload, stream=True, timeout=5000)
        message = []
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    output = post_process_code(output)
                    state.messages[-1][-1] = output + "▌" 
                    message.append(output)
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    message.append(output)

    except requests.exceptions.RequestException as e:
        print(server_error_msg)

    # print(message[-1])
    # print("----- STate MEssage -----")
    # print(state.messages)
    return message[-1]


def llava_med(state, text, image_path, image_process_mode):
    state = default_conversation.copy()
    # image = Image.open(image_path)
    # image = image.convert('RGB')
    # image_process_mode = 'Crop'
    # state = add_text(state, text, image, image_process_mode)
    output = http_bot(state, 0.5, 1024)
    return output


if __name__ == '__main__':
    user_prompt = "is this a radiology image?"
    image_path = '/content/LLaVA-Med/llava/serve/examples/med_img_1.png'
    result = llava_med(user_prompt, image_path)
    print(result)
  
