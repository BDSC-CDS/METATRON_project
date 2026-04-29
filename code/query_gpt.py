from openai import OpenAI


def query_gpt(system_prompt, prompt, model_name, client, OutputFormat):

    chat_completion = client.beta.chat.completions.parse(
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },  
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=model_name,
        temperature=0.8,
        response_format=OutputFormat
    )
    answer_str =chat_completion.choices[0].message.content
    try:
        answer = OutputFormat.parse_raw(answer_str)
        return answer.dict()
    except Exception as e:
        print(f"Error parsing answer: {e}")
        print(f"Answer string: {answer_str}")
        return None
