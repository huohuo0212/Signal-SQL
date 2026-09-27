import json.decoder
import time
import os
import httpx
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from openai import OpenAI
from utils.enums import LLM

# 模块级变量，存储初始化后的 client
_client = None
_model = None
_api_key = None

def init_chatgpt(OPENAI_API_KEY, OPENAI_GROUP_ID, model):
    """初始化 OpenAI/OpenRouter 客户端 (openai v1.x API)"""
    global _client, _model, _api_key
    _model = model
    _api_key = OPENAI_API_KEY
    
    api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    
    _client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=api_base,
        timeout=httpx.Timeout(120.0, connect=30.0),
        http_client=httpx.Client(verify=False)
    )
    print(f"[ChatGPT] Initialized API client for model: {model} (base={api_base}, timeout=120s)")


def ask_chat(model, messages: list, temperature, n):
    """使用 openai v1.x Chat Completions API"""
    import requests
    response_clean = []
    total_prompt = 0
    total_comp = 0
    
    api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    url = f"{api_base.rstrip('/')}/chat/completions"
    
    global _api_key
    key_to_use = _api_key if _api_key else os.environ.get("OPENAI_API_KEY", "")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key_to_use}"
    }
    
    for _ in range(n):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4096,
            "n": 1
        }
        res = requests.post(url, headers=headers, json=payload, timeout=60, verify=False)
        res.raise_for_status()
        data = res.json()
        
        response_clean.append(data["choices"][0]["message"]["content"])
        if "usage" in data:
            total_prompt += data["usage"].get("prompt_tokens", 0)
            total_comp += data["usage"].get("completion_tokens", 0)
    
    if n == 1:
        response_clean = response_clean[0]
    
    return dict(
        response=response_clean,
        prompt_tokens=total_prompt,
        completion_tokens=total_comp,
        total_tokens=total_prompt + total_comp
    )


def ask_llm(model: str, batch: list, temperature: float, n: int):
    n_repeat = 0
    MAX_RETRIES = 15
    response = {}

    while True:
        try:
            # batch size must be 1 for chat models
            assert len(batch) == 1, "batch must be 1 in chat mode"
            messages = [{"role": "user", "content": batch[0]}]
            response = ask_chat(model, messages, temperature, n)
            
            # 兼容性处理
            # 对于 Chat 接口，由于 batch size=1，我们需要返回 2D 列表 (或者保持外层长度=batch_size)
            # 即 `response['response']` 应该是 `[ [ans1, ans2, ... ans_n] ]`
            if isinstance(response, dict) and 'response' in response:
                ans_list = response['response']
                if not isinstance(ans_list, list):
                    ans_list = [ans_list]
                
                # 由于这是1个 query，如果是 n>1，由于 n 对应的答案是一个整体，我们把它包一层
                # 如果 n=1, ask_chat 返回单字符串，上面变成了 ['ans']，还得再包装一层变成 [['ans']]
                # 但 ask_llm.py 的 n=1 的逻辑直接取 responses = [...] 然后遍历。所以如果 n=1, [ans] 是可以的。
                # 但为了统一逻辑，对于 batch_size=1 的 query，其所有候选应该被视为一条记录的多个预测。
                # 其实原来的 ask_llm.py 代码 n=1: for sql in responses (也就是 [ans] 遍历出 ans)
                # 若 n>1: zip(responses, cur_db_ids) -> responses 需要有 batch_size 长度
                
                # 安全起见，无论 n 是多少，强制包裹以保证 responses[0] 是该 question 的预测列表（n=1 时由于 ask_llm 特殊判断不需要）
                if n > 1:
                    response['response'] = [ans_list]
                else:
                    response['response'] = ans_list
            
            break  # 成功跳出
            
        except Exception as e:
            n_repeat += 1
            wait_time = min(2 ** n_repeat, 30)  # 指数退避, 最多等30秒
            print(f"\n[Error Attempt {n_repeat}/∞] {type(e).__name__}: {e}")
            print(f"  Retrying in {wait_time}s...")
            time.sleep(wait_time)
            continue

    return response


