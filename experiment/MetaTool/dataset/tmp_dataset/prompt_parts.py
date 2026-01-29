import argparse
import json
from pathlib import Path


USER_QUERY_HEADER = "User's Query:"
TOOLS_HEADER = "List of Tools with Names and Descriptions:"
TOOLS_END_MARK = "[List of Tools with Names and Descriptions End]"


def extract_relevant_prompt(text: str) -> str:
    """
    从完整的 action_prompt 中截取：
    - User's Query 段
    - List of Tools with Names and Descriptions 段
    并拼成新的 prompt 返回。
    """
    if not text:
        return text

    # 找到 "User's Query:" 起点
    uq_idx = text.find(USER_QUERY_HEADER)
    if uq_idx == -1:
        print(text)
        # 没找到就直接返回原文，避免破坏数据
        return text

    # 从 "User's Query:" 之后继续找 tools 列表头
    tools_header_idx = text.find(TOOLS_HEADER, uq_idx)
    if tools_header_idx == -1:
        print(text)
        # 只找到用户问题，就截取从 User's Query: 到文本末尾
        return text[uq_idx:].strip()

    # 找 tools 列表结尾标记
    tools_end_idx = text.find(TOOLS_END_MARK, tools_header_idx)
    if tools_end_idx == -1:
        # 没有明确的结束标记，就从 User's Query: 截到文本末尾
        print(text)
        return text[uq_idx:].strip()

    tools_end_idx += len(TOOLS_END_MARK)

    new_prompt = text[uq_idx:tools_end_idx]
    return new_prompt.strip()


def process_file(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 既兼容 list[dict] 也兼容单个 dict
    if isinstance(data, dict):
        items = [data]
        is_single = True
    else:
        items = data
        is_single = False

    for item in items:
        if not isinstance(item, dict):
            continue
        action_prompt = item.get("action_prompt")
        if not isinstance(action_prompt, str):
            continue
        prefix = "You are a helpful AI assistant. Your current task is to choose the appropriate tool to solve the user’s query based on their question. I will provide you with the user’s question and information about the tools. If there is a tool in the list that is applicable to this query, please return the name of the tool (you can choose two tools at most, and return in the form of [tool1name, tool2name] without any other information or format). If there isn’t, please return 'None.'" if str(item.get("source_subtask", "")).lower().find("subtask4") != -1 else "You are a helpful AI assistant. Your current task is to choose the appropriate tool to solve the user's query based on their question. I will provide you with the user's question and information about the tools.\nIf there is a tool in the list that is applicable to this query, please return the name of the tool (you can only choose one tool, and you should only return the name of the tool without any other information or format). If there isn't, please return 'None'."
        new_prompt = extract_relevant_prompt(action_prompt)
        item["action_prompt"] = f"{prefix}{new_prompt}"

    out_data = items[0] if is_single else items
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="从 action_prompt 中提取 User's Query 和 List of Tools with Names and Descriptions 两段，并覆盖写回 action_prompt 字段后输出到新文件。"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="输入 JSON 文件绝对路径（包含 action_prompt 字段）。",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="输出 JSON 文件绝对路径。",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    process_file(input_path, output_path)


if __name__ == "__main__":
    main()


