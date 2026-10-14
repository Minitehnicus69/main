from llm.agent import API_KEY, BASE_URL, MODEL_NAME
from openai import OpenAI

c = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=30)
print([m.id for m in c.models.list()][:12])
r = c.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": "Spune doar: ok"}],
                              max_tokens=300, extra_body={"reasoning_effort": "low"})
print(MODEL_NAME, "->", r.choices[0].message.content, "|", r.choices[0].finish_reason)