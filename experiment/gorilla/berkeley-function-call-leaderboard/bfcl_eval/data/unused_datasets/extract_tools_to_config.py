#!/usr/bin/env python3
"""
提取 data 文件夹和 multi_turn_func_doc 文件夹中所有 JSON 文件的工具信息，
并按照 config.json 的格式保存。
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Set

def extract_tools_from_jsonl(file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    从 JSONL 文件中提取工具信息。
    返回字典：{工具名称: {description: ..., parameters: ...}}
    """
    tools = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # 处理 data 文件夹中的文件格式（包含 "function" 字段）
                    if "function" in data:
                        functions = data["function"]
                        if isinstance(functions, list):
                            for func in functions:
                                if isinstance(func, dict) and "name" in func:
                                    name = func["name"].lower()  # 统一转换为小写
                                    description = func.get("description", "")
                                    parameters = func.get("parameters", {})
                                    
                                    # 如果工具已存在，保留更完整的描述和参数
                                    if name not in tools:
                                        tools[name] = {
                                            "description": description,
                                            "parameters": parameters
                                        }
                                    else:
                                        # 如果新描述更长或参数更完整，则更新
                                        if len(description) > len(tools[name]["description"]):
                                            tools[name]["description"] = description
                                        if parameters and not tools[name]["parameters"]:
                                            tools[name]["parameters"] = parameters
                    
                    # 处理 multi_turn_func_doc 文件夹中的文件格式（直接包含 "name" 和 "description"）
                    elif "name" in data:
                        name = data["name"].lower()  # 统一转换为小写
                        description = data.get("description", "")
                        parameters = data.get("parameters", {})
                        
                        # 如果工具已存在，保留更完整的描述和参数
                        if name not in tools:
                            tools[name] = {
                                "description": description,
                                "parameters": parameters
                            }
                        else:
                            # 如果新描述更长或参数更完整，则更新
                            if len(description) > len(tools[name]["description"]):
                                tools[name]["description"] = description
                            if parameters and not tools[name]["parameters"]:
                                tools[name]["parameters"] = parameters
                
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num} in {file_path}: {e}")
                    continue
    
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    
    return tools

def extract_all_tools(data_dir: str, multi_turn_func_doc_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    从所有 JSON 文件中提取工具信息。
    """
    all_tools = {}
    
    # 处理 data 文件夹中的 JSON 文件
    data_path = Path(data_dir)
    excluded_files = {"config.json", "multi_config.json"}
    
    print(f"Processing files in {data_dir}...")
    for json_file in data_path.glob("*.json"):
        if json_file.name in excluded_files:
            print(f"Skipping {json_file.name}")
            continue
        
        print(f"  Processing {json_file.name}...")
        tools = extract_tools_from_jsonl(str(json_file))
        print(f"    Found {len(tools)} tools")
        
        # 合并工具信息
        for name, info in tools.items():
            if name not in all_tools:
                all_tools[name] = info
            else:
                # 保留更完整的描述和参数
                if len(info["description"]) > len(all_tools[name]["description"]):
                    all_tools[name]["description"] = info["description"]
                if info["parameters"] and not all_tools[name]["parameters"]:
                    all_tools[name]["parameters"] = info["parameters"]
    
    # 处理 multi_turn_func_doc 文件夹中的 JSON 文件
    multi_turn_path = Path(multi_turn_func_doc_dir)
    if multi_turn_path.exists():
        print(f"\nProcessing files in {multi_turn_func_doc_dir}...")
        for json_file in multi_turn_path.glob("*.json"):
            print(f"  Processing {json_file.name}...")
            tools = extract_tools_from_jsonl(str(json_file))
            print(f"    Found {len(tools)} tools")
            
            # 合并工具信息
            for name, info in tools.items():
                if name not in all_tools:
                    all_tools[name] = info
                else:
                    # 保留更完整的描述和参数
                    if len(info["description"]) > len(all_tools[name]["description"]):
                        all_tools[name]["description"] = info["description"]
                    if info["parameters"] and not all_tools[name]["parameters"]:
                        all_tools[name]["parameters"] = info["parameters"]
    else:
        print(f"Warning: {multi_turn_func_doc_dir} does not exist")
    
    return all_tools

def main():
    # 设置路径
    base_dir = Path(__file__).parent.parent / "data"
    data_dir = str(base_dir)
    multi_turn_func_doc_dir = str(base_dir / "multi_turn_func_doc")
    output_file = str(base_dir / "config.json")
    
    print("=" * 60)
    print("Extracting tools from JSON files...")
    print("=" * 60)
    
    # 提取所有工具信息
    all_tools = extract_all_tools(data_dir, multi_turn_func_doc_dir)
    
    print(f"\nTotal unique tools found: {len(all_tools)}")
    
    # 保存到 config.json
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_tools, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully saved {len(all_tools)} tools to {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
