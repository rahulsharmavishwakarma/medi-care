from llava.serve.try_inference import llava_med

import os

os.environ["OPENAI_API_KEY"] = "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"
os.environ["GOOGLE_API_KEY"] = "AIzaSyB3IhQmdb6FS48yODc64lcLmF5Us30WgY4"
os.environ["GOOGLE_CSE_ID"] = "526a61a0320a34080"

import os
from langchain.agents import Tool
from langchain.agents import load_tools, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent
import gradio as gr
from langchain.prompts import PromptTemplate

from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

'------------------------------------------------------------------'

image = '/content/LLaVA-Med/llava/serve/examples/synpic42202.jpg'
image_process_mode = 'Crop'
user_query = ' can i take sleep medicine '

questions = [
        "Are there any abnormalities present in the lung parenchyma?",
        "Are there any masses, nodules, or infiltrates detected in the lungs?",
        "Are there any abnormalities in the pleura or mediastinum?",
        "What is the nature of any detected masses or nodules (e.g., solid, ground-glass, spiculated)?",
        "Are any abnormalities suggestive of infection, inflammation, or malignancy?",
        "Is there evidence of focal or diffuse disease involvement?",
        "Are there signs of disease progression or regression compared to previous scans?",
        "What is the size, shape, and location of any detected lesions?",
        "Are there any calcifications or cavitations present within the lesions?",
        "Is there evidence of pulmonary fibrosis, emphysema, or other parenchymal lung diseases?",
        "Are there any signs of ground-glass opacities, consolidation, or air trapping?",
        "Are there any abnormalities in the pulmonary vasculature, such as emboli or thrombosis?",
        "Is there evidence of pulmonary hypertension or vascular congestion?",
        "Are there any bronchial wall thickening, stenosis, or dilatation observed?",
        "Are there signs of bronchiectasis or bronchial obstruction?",
        "Are there any enlarged lymph nodes or masses in the mediastinum?",
        "Is there evidence of mediastinal shift or compression of adjacent structures?",
        "Is there evidence of pleural effusion, thickening, or calcification?",
        "Are there signs of pneumothorax or pleural-based lesions?",
        "How do the imaging findings correlate with the patient's clinical history and symptoms?",
        "What further diagnostic or therapeutic steps are recommended based on the imaging findings?"
    ]


ques_ans = {}

for question in questions:
    answer = llava_med(text=question, image_path=image, image_process_mode=image_process_mode)
    ques_ans[question] = answer

complete = ''

i = 1
for question in ques_ans.keys():
    complete += f'\n{i}.{question} \nA. {ques_ans[question]}\n' 
    i += 1
i = 1

print(complete)
'----------------------------------------------------------------'
model_name = "gpt-3.5-turbo-0301"

    # Initialize the chat object.

chat = ChatOpenAI(model_name=model_name, temperature=0.4)
prompt_template = PromptTemplate.from_template(
        """

    {q_and_a}

    Based on this Diagnosis Q&A, Create a Diagnosis Report Description without missing any information.

    """
    )
prompt = prompt_template.format(q_and_a = complete)
# print(prompt)
messages = []
human_msg = HumanMessage(content=prompt)
messages.append(human_msg)
diagnosis_report = chat(messages)
print(diagnosis_report.content)

'----------------------------------------------------------------'
search = GoogleSearchAPIWrapper()
tools = [
        Tool(
            name="Search",
            func = search.run,
            description = "Useful for retrieving information from the provided context and conducting web searches when necessary."
        ),
        ]

llm = ChatOpenAI(temperature=0.3)
agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                verbose=True)

template = """

    Questions : {user_query}

    The Diagnosis Description: {diagnosis_report}


    You are a Great Healthcare Professional. Considering the Diagnosis Description, provide an helpful answers to the Question.
    If there is any Caution or the Dosage of the Medication, then provide complete details about it.
    Don't requests or search the web for more than 2 times.
    """

prompt = PromptTemplate(
        input_variables=['user_query', 'diagnosis_report'], 
        template=template,
    )

final_prompt = prompt.format(user_query = user_query,
                            diagnosis_report = diagnosis_report.content)

output = agent_chain.run(final_prompt)

print(output)
'----------------------------------------------------------------'


