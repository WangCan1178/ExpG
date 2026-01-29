import os
import sys
import json
import argparse

# 不需要这个脚本
# 保证可以从项目根目录导入 src 下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.prompt.prompt_construction import PromptConstructor


TOP_50_TOOLS = [
    "FinanceTool",
    "NewsTool",
    "PDF&URLTool",
    "TripTool",
    "CourseTool",
    "WeatherTool",
    "JobTool",
    "ResearchFinder",
    "ProductSearch",
    "ResearchHelper",
    "Discount",
    "MusicTool",
    "TripAdviceTool",
    "HousePurchasingTool",
    "RepoTool",
    "NotesTool",
    "CreditYelp",
    "noteable",
    "ABCmouse",
    "PolishTool",
    "MemoryTool",
    "ChartTool",
    "RestaurantBookingTool",
    "tira",
    "search",
    "hackit_web_scanner",
    "MixerBox_Translate_AI_language_tutor",
    "storybird_stories",
    "placid",
    "universal",
    "ArtCollection",
    "ImageSearch",
    "webhooks",
    "LarkBaseImporter",
    "fundsdbsearch",
    "dart",
    "find_agency",
    "AutoInfra1",
    "SSH",
    "IndoorPlants",
    "recipe_retrieval",
    "AI2sql",
    "speak",
    "ExchangeTool",
    "CharityTool",
    "MapTool",
    "DataRetrievalTool",
    "LawTool",
    "CompanyInfoTool",
    "MediaModifyTool",
]


def build_tool_block(tool_name: str, prompt_constructor: PromptConstructor) -> str:
    """
    按指定格式拼接工具名称和描述：
    List of Tools with Names and Descriptions:
    [List of Tools with Names and Descriptions Start]
    1. tool name: xxx, tool description: [...]
    """
    tool_des = prompt_constructor.get_tool_description(tool_name)
    block = (
        "List of Tools with Names and Descriptions:\n"
        "[List of Tools with Names and Descriptions Start]\n"
        f"1. tool name: {tool_name}, tool description: {tool_des}\n"
    )
    return block


def get_tool_for_negative_example(query: str) -> str | None:
    """
    之前这里是通过 Milvus + 向量检索找到最相似的 1 个工具。
    现在你不再调用 OpenAI / embedding 接口，因此该函数不再实际被使用，
    仅保留占位以避免外部引用出错。
    """
    return None


def get_tool_for_positive_example(item) -> str | None:
    """
    对于 positive 样本，直接使用其 tool 字段。
    如果 tool 为列表，则取第一个。
    """
    tool = item.get("tool")
    if tool is None:
        return None
    if isinstance(tool, list):
        return tool[0] if tool else None
    return tool


def append_tools_to_thought_prompt(input_path: str, output_path: str) -> None:
    """
    对于 JSON 中的每个样本：
    - 如果 label == 'negative'：
        直接使用预定义的 50 个最常用工具
    - 如果 label == 'positive'：
        使用其 tool 字段
    然后按指定格式，把工具信息拼接到 thought_prompt 结尾，并保存到新的 JSON 文件。
    """
    prompt_constructor = PromptConstructor()

    with open(input_path, "r") as f:
        data = json.load(f)

    for idx, item in enumerate(data):
        label = item.get("label")
        thought_prompt = item.get("thought_prompt")

        # 没有 thought_prompt 的样本直接跳过
        if not thought_prompt:
            continue

        if label == "negative":
            # negative 样本：使用预定义的 Top 50 工具
            tool_names = TOP_50_TOOLS
        else:
            # positive 样本：仍然使用其 tool 字段
            tool_name = get_tool_for_positive_example(item)
            if not tool_name:
                print(f"[WARN] Sample {idx} has no valid tool, skipped.")
                continue
            tool_names = [tool_name]

        # 按与 positive 相同的列表格式拼接工具信息：
        # List of Tools with Names and Descriptions:
        # [List of Tools with Names and Descriptions Start]
        # 1. tool name: xxx, tool description: [...]
        lines = [
            "List of Tools with Names and Descriptions:",
            "[List of Tools with Names and Descriptions Start]",
        ]
        for i, t_name in enumerate(tool_names, start=1):
            tool_des = prompt_constructor.get_tool_description(t_name)
            lines.append(f"{i}. tool name: {t_name}, tool description: {tool_des}")
        tool_block = "\n".join(lines) + "\n"

        # 在 thought_prompt 末尾追加工具列表
        item["thought_prompt"] = thought_prompt.rstrip() + "\n" + tool_block

    with open(output_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Processed file saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="根据样本的 label 和 query / tool 字段，将单个工具名称和描述拼接到 thought_prompt 之后。"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="输入 JSON 文件路径，例如 dataset/tmp_dataset/Task1.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出 JSON 文件路径，例如 dataset/tmp_dataset/Task1_with_tools.json",
    )

    args = parser.parse_args()
    append_tools_to_thought_prompt(args.input, args.output)


if __name__ == "__main__":
    main()


