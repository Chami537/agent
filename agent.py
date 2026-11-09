import json 
import os
import subprocess
import sys

from openai import OpenAI

client=OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ.get("BASE_URL", "https://api.deepseek.com"),
)

MODEL=os.environ.get("MODEL", "deepseek-flash")

MAX_STEPS=20

SYSTEM=(
    "You are an agent that can execute commands on the local machine. "
    "You will be given a task and you should execute commands to accomplish the task. "
)

def read_file(path:str)->str:
    with open(path,encoding="utf-8") as f:
        return f.read()

def write_file(path:str, content:str)->str:
    with open(path,"w",encoding="utf-8") as f:
        f.write(content)
    return f"已写入{path}({len(content)}字符)"

def run_shell(command:str)->str:
    if input(f"\n[需要确认]执行命令:{command}\n允许?(y/n)").strip().lower()!="y":
        return "用户拒绝执行命令"
    try:
        p=subprocess.run(command, shell=True, capture_output=True, text=True,timeout=60)
    except subprocess.TimeoutExpired:
        return "命令执行超时"
    return (p.stdout+p.stderr)[-4000:]or "命令执行无输出"

TOOL_FUNCS={
    "read_file": read_file,
    "write_file": write_file,
    "run_shell": run_shell,
}

def _tool(name, desc, props):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
                "required": list(props),
            },
        },
    }


TOOLS = [
    _tool("read_file", "读取文件内容", {"path": "文件路径"}),
    _tool("write_file", "写入文件（覆盖）", {"path": "文件路径", "content": "完整文件内容"}),
    _tool("run_shell", "执行 shell 命令并返回输出", {"command": "要执行的命令"}),
]


def run(task:str)->None:
    messages=[
        {"role":"system","content":SYSTEM},
        {"role":"user","content":task}
    ]
    for step in range(1,MAX_STEPS+1):
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            print(f"\n[完成,共{step}步]没有工具调用,输出:\n{msg.content}")
            return 

        for call in msg.tool_calls:
            name = call.function.name#获取模型要求调用的工具名称
            print(f"[第 {step} 步] {name}({call.function.arguments[:80]})")
            try:
                args = json.loads(call.function.arguments)
                result = TOOL_FUNCS[name](**args)
            except Exception as e:  # 工具出错也要告诉模型，让它自己修正
                result = f"工具执行出错：{type(e).__name__}: {e}"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    print(f"\n[达到 {MAX_STEPS} 步上限，已停止]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('用法：python agent.py "你的任务"')
    run(sys.argv[1])