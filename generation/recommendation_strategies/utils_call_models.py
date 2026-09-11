from openai import OpenAI
from pydantic import BaseModel
from models import *

N_TOKENS = 5000
TEMP = 0.3

# Functions
def extract_answer(model_answer: str):
    start_strings = ["assistantfinal", "final answer", "let's craft"]
    for s in start_strings:
        index = model_answer.find(s)
        if index > -1:
            answer = model_answer[index + len(s):]
            return answer
    print("No start string found!")
    return model_answer

def query_model(client: OpenAI, prompt: str):
    completion = client.completions.create(model=models["llm"]["model_name"],
                                      prompt=f"{prompt}",
                                      max_tokens=N_TOKENS,
                                      temperature=TEMP)
    model_answer = completion.choices[0].text
    return extract_answer(model_answer)

def query_model_structured_message(client: OpenAI, sys_prompt: str, user_prompt: str):
    completion = client.chat.completions.create(model=models["llm"]["model_name"],
                                      messages=[{"role":"system", "content":f"{sys_prompt}"}, {"role":"user", "content":f"{user_prompt}"}],
                                      max_tokens=N_TOKENS,
                                      temperature=TEMP)
    # print(completion)
    model_answer = completion.choices[0].message.content
    # return extract_answer(model_answer)
    return model_answer

def query_model_structured_output(client: OpenAI, sys_prompt: str, user_prompt: str, structured_output_fomat: BaseModel):
    completion = client.beta.chat.completions.parse(
                                    messages = [
                                                {"role":"system", "content":f"{sys_prompt}"},
                                                {"role":"user", "content":f"{user_prompt}"}
                                                ],
                                    model=models["llm"]["model_name"],
                                    max_completion_tokens=N_TOKENS,
                                    temperature=TEMP,
                                    response_format=structured_output_fomat)
    str_answer = completion.model_dump()
    #print("Generated output:", str_answer['choices'][0]['message']['content'])
    return str_answer['choices'][0]['message']['content']