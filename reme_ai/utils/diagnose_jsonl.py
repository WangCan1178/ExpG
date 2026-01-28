"""
诊断工具：检查 JSONL 文件中的问题

用于详细检查 JSONL 文件中每一行的 JSON 有效性，并报告具体的问题位置。
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict
from loguru import logger


def diagnose_jsonl_file(file_path: Path) -> Dict:
    """
    诊断 JSONL 文件，找出所有有问题的行
    
    Args:
        file_path: JSONL 文件路径
        
    Returns:
        dict: 诊断结果
        {
            "total_lines": int,
            "valid_lines": int,
            "invalid_lines": int,
            "empty_lines": int,
            "errors": [
                {
                    "line_num": int,
                    "error_type": str,
                    "error_msg": str,
                    "error_pos": dict,
                    "line_preview": str
                }
            ]
        }
    """
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    result = {
        "total_lines": 0,
        "valid_lines": 0,
        "invalid_lines": 0,
        "empty_lines": 0,
        "errors": []
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            result["total_lines"] += 1
            stripped = line.strip()
            
            if not stripped:
                result["empty_lines"] += 1
                continue
            
            try:
                json.loads(stripped)
                result["valid_lines"] += 1
            except json.JSONDecodeError as e:
                result["invalid_lines"] += 1
                error_info = {
                    "line_num": line_num,
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                    "error_pos": {
                        "lineno": getattr(e, 'lineno', None),
                        "colno": getattr(e, 'colno', None),
                        "pos": getattr(e, 'pos', None)
                    },
                    "line_preview": stripped[:200] + "..." if len(stripped) > 200 else stripped,
                    "line_length": len(stripped)
                }
                result["errors"].append(error_info)
    
    return result


def print_diagnosis(file_path: Path):
    """打印诊断结果"""
    print(f"\n{'='*80}")
    print(f"诊断文件: {file_path}")
    print(f"{'='*80}\n")
    
    result = diagnose_jsonl_file(file_path)
    
    print(f"总行数: {result['total_lines']}")
    print(f"有效行: {result['valid_lines']}")
    print(f"无效行: {result['invalid_lines']}")
    print(f"空行: {result['empty_lines']}")
    
    if result['errors']:
        print(f"\n发现 {len(result['errors'])} 个错误:\n")
        for i, error in enumerate(result['errors'], 1):
            print(f"错误 #{i}:")
            print(f"  行号: {error['line_num']}")
            print(f"  错误类型: {error['error_type']}")
            print(f"  错误信息: {error['error_msg']}")
            if error['error_pos']['pos'] is not None:
                print(f"  问题位置: 第 {error['error_pos']['lineno']} 行，"
                      f"第 {error['error_pos']['colno']} 列 "
                      f"(字符位置 {error['error_pos']['pos']})")
            print(f"  行长度: {error['line_length']} 字符")
            print(f"  行内容预览:")
            print(f"    {error['line_preview']}")
            print()
    else:
        print("\n✓ 未发现错误，文件格式正确！")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python diagnose_jsonl.py <文件路径>")
        print("示例:")
        print("  python diagnose_jsonl.py ./local_vector_store/tool_workspace.jsonl")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    print_diagnosis(file_path)

