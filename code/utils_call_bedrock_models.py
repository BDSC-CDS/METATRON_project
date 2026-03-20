import json
import ast
import os
import boto3
import botocore
from pydantic import BaseModel
import instructor
import pandas as pd
import _io
from typing import Any

# settings 
cred_path = "bedrick_cred/"

MAX_TOKENS = 1000

# AWS usage
def generate_structured_output(str_client: instructor.Instructor, model_id: str, sys_prompt: str, user_prompt: str, 
                               OutputStructure: BaseModel) -> str:
    try:
        user = str_client.chat.completions.create(
            modelId= model_id,
            messages=[
                {"role": "system", "content": f"{sys_prompt}"},
                {"role": "user", "content": f"{user_prompt}"},
            ],
            # instructor require a response model to parse the output
            response_model=OutputStructure,
            max_tokens=MAX_TOKENS
        )
        return user.model_dump()
    except Exception as e:
        print(f"JSON Parsing Error: {e}")
        return None
    
# Elaborate evaluation output
def query_models(client: instructor.Instructor, sys_prompt: str, user_prompt: str, model: str, output_structure: BaseModel) -> Any:
    # print(f"Sys prompt: {sys_prompt}")
    # print(f"User prompt: {prompt}")
    answer = generate_structured_output(client, sys_prompt=sys_prompt, user_prompt=user_prompt, model_id=model, 
                                        OutputStructure=output_structure)
    return answer