import logging
import os
import traceback
from abc import ABC, abstractmethod

import torch
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass


class GPT3TurboSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-3.5-turbo"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            traceback.print_exc()
            return e


class GPT3SummarizationModel(BaseSummarizationModel):
    def __init__(self, model="text-davinci-003"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            traceback.print_exc()
            return e


class Phi3SummarizationModel(BaseSummarizationModel):
    def __init__(self, model='microsoft/Phi-3-mini-4k-instruct',
                 tokenizer='microsoft/Phi-3-mini-4k-instruct'):
        device_map = 'cuda:3' if torch.cuda.is_available() else 'cpu'

        self.model = AutoModelForCausalLM.from_pretrained(model,
                                                          device_map=device_map,
                                                          torch_dtype='auto',
                                                          trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        # self.tokenizer.to(device_map)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"Write a summary of the following, including as many key details as "
                               f"possible: {context}:",
                }
            ]

            inputs = (self.tokenizer
                      .apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt"))
            # inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt")

            # TODO: implement stopping_criteria
            outputs = self.model.generate(inputs, max_new_tokens=max_tokens,
                                          temperature=0,
                                          top_p=1,
                                          repetition_penalty=1.0,
                                          diversity_penalty=0.0,
                                          )

            text = self.tokenizer.batch_decode(outputs)[0].strip()
            print('------------summary--------------')
            print('\tcontext: ', context, '\n\tsummary: ', text)
            text = text[text.index(':<|end|><|assistant|>') + len(':<|end|><|assistant|>'):].strip()
            print('summary trimmed: ', text)
            print('---------------------------')

            return text
        except Exception as e:
            print(e)
            traceback.print_exc()
            return ""
