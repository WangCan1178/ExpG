import json

import tqdm
from utils import QwenOpenAIWrapper
from api_call_extraction import get_api_call, parse_api_call
from tool_manager import ToolManager

# 与 lv1/lv2 一致的提示，可按需替换为 evaluator.py 中的完整 prompt
api_call_prompt = '''
Based on the given API description and the existing conversation history 1..t, please generate the API request that the AI should call in step t+1 and output it in the format of [ApiName(key1='value1', key2='value2', ...)], replace the ApiName with the actual API name, and replace the key and value with the actual parameters. 
Your output should start with a square bracket "[" and end with a square bracket "]". Do not output any other explanation or prompt or the result of the API call in your output. 
This year is 2023.
Input: 
User: [User's utterence]
AI: [AI's utterence]

Expected output:
[ApiName(key1='value1', key2='value2', ...)]

API descriptions:
'''

response_prompt = '''
Based on the given API description and the existing conversation history 1..t, please generate the next dialog that the AI should response after the API call t.
This year is 2023.
Input: 
User: [User's utterence]
AI: [AI's utterence]
[ApiName(key1='value1', key2='value2', …)]

Expected output:
AI: [AI's utterence]

API descriptions:
'''

model = QwenOpenAIWrapper(model="qwen3-8b")
tm = ToolManager('./lv3_apis')
gts = json.load(open("/Users/xiaocan/Toolmem/experiment/api-bank/API-Bank/test-data/level-3.json"))

def chat(prompt, msgs):
    msgs = [{'role':'system','content': prompt}] + msgs
    r = model.call(msgs)
    return r['choices'][0]['message']['content']

api_preds = []
dialog_preds = []

for sid, sample in tqdm.tqdm(enumerate(gts)):
    # 多轮对话起始：用户需求
    history = [{'role':'user','content': sample["requirement"]}]

    for api_id, api_step in enumerate(sample["apis"]):
        api_desc = tm.get_api_description(api_step["api_name"])
        out = chat(api_call_prompt + api_desc, history)
        api_call = get_api_call(out) or out.strip()

        # 解析并真实调用工具，供上下文使用（失败时记异常）
        try:
            pred_name, pred_params = parse_api_call(api_call)
            call_result = tm.api_call(pred_name, **pred_params)
        except Exception as e:
            call_result = {"exception": str(e)}

        api_preds.append({
            "sample_id": sid,
            "api_id": api_id,
            "pred": api_call
        })

        # 将模型回复与执行结果加入多轮上下文
        history.append({'role':'assistant','content': out})
        history.append({'role':'system',
                        'content': f"{api_call} Response: {call_result}"})

    # 所有 API 完成后生成最终对话回复
    reply = chat(response_prompt, history)
    dialog_preds.append({
        "sample_id": sid,
        "output": sample["response"],  # evaluator 需要参考回复
        "pred": reply
    })

# 写出预测文件
with open("pred_api.jsonl","w",encoding="utf-8") as fa:
    for r in api_preds:
        fa.write(json.dumps(r, ensure_ascii=False) + "\n")
print("pred_api.jsonl 已生成（if_api=True 使用）")

with open("pred_dialog.jsonl","w",encoding="utf-8") as fd:
    for r in dialog_preds:
        fd.write(json.dumps(r, ensure_ascii=False) + "\n")
print("pred_dialog.jsonl 已生成（if_api=False 使用）")