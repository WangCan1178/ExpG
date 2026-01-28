"""
工具函数：清理和修复损坏的 JSONL 文件

当向量存储的 JSONL 文件因为并发写入或其他原因导致包含无效 JSON 行时，
可以使用此工具清理文件，移除无效行并保留有效数据。
"""

import json
import shutil
from pathlib import Path
from typing import Tuple
from loguru import logger


def clean_jsonl_file(file_path: Path, backup: bool = True) -> Tuple[int, int]:
    """
    清理 JSONL 文件，移除无效的 JSON 行
    
    Args:
        file_path: JSONL 文件路径
        backup: 是否在清理前创建备份（默认 True）
        
    Returns:
        Tuple[int, int]: (有效行数, 移除的无效行数)
        
    Example:
        >>> from pathlib import Path
        >>> valid_count, invalid_count = clean_jsonl_file(Path("data.jsonl"))
        >>> print(f"保留 {valid_count} 行，移除 {invalid_count} 行")
    """
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 创建备份
    if backup:
        backup_path = file_path.with_suffix(file_path.suffix + '.backup')
        logger.info(f"创建备份文件: {backup_path}")
        shutil.copy2(file_path, backup_path)
    
    # 读取并清理
    valid_lines = []
    invalid_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped:
                    # 跳过空行
                    continue
                
                try:
                    # 尝试解析 JSON
                    json.loads(stripped)
                    valid_lines.append(line.rstrip('\n\r'))  # 保留原始行，但统一换行符
                except json.JSONDecodeError as e:
                    invalid_count += 1
                    # 显示更详细的错误信息，包括问题位置和行内容预览
                    preview = stripped[:100] + "..." if len(stripped) > 100 else stripped
                    logger.warning(f"第 {line_num} 行无效 JSON，已跳过: {e}")
                    logger.warning(f"  问题位置: 第 {e.lineno} 行，第 {e.colno} 列 (字符位置 {e.pos})")
                    logger.warning(f"  行内容预览: {preview}")
        
        # 写回清理后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in valid_lines:
                f.write(line + '\n')
        
        valid_count = len(valid_lines)
        logger.info(f"清理完成: {file_path} - 保留 {valid_count} 行，移除 {invalid_count} 行")
        return valid_count, invalid_count
        
    except Exception as e:
        logger.error(f"清理文件时出错: {file_path}, 错误: {e}")
        # 如果出错且有备份，尝试恢复
        if backup and backup_path.exists():
            logger.warning(f"尝试从备份恢复: {backup_path}")
            shutil.copy2(backup_path, file_path)
        raise


def clean_jsonl_directory(directory: Path, pattern: str = "*.jsonl", backup: bool = True) -> dict:
    """
    清理目录中所有匹配的 JSONL 文件
    
    Args:
        directory: 目录路径
        pattern: 文件匹配模式（默认 "*.jsonl"）
        backup: 是否创建备份
        
    Returns:
        dict: 清理结果统计
        {
            "files_processed": int,
            "total_valid_lines": int,
            "total_invalid_lines": int,
            "file_results": {file_path: (valid_count, invalid_count)}
        }
        
    Example:
        >>> from pathlib import Path
        >>> results = clean_jsonl_directory(Path("./local_vector_store"))
        >>> print(f"处理了 {results['files_processed']} 个文件")
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    
    files = list(directory.glob(pattern))
    if not files:
        logger.warning(f"在 {directory} 中未找到匹配 {pattern} 的文件")
        return {
            "files_processed": 0,
            "total_valid_lines": 0,
            "total_invalid_lines": 0,
            "file_results": {}
        }
    
    results = {
        "files_processed": 0,
        "total_valid_lines": 0,
        "total_invalid_lines": 0,
        "file_results": {}
    }
    
    for file_path in files:
        try:
            valid_count, invalid_count = clean_jsonl_file(file_path, backup=backup)
            results["files_processed"] += 1
            results["total_valid_lines"] += valid_count
            results["total_invalid_lines"] += invalid_count
            results["file_results"][str(file_path)] = (valid_count, invalid_count)
        except Exception as e:
            logger.error(f"处理文件失败: {file_path}, 错误: {e}")
    
    logger.info(f"目录清理完成: {directory}")
    logger.info(f"处理了 {results['files_processed']} 个文件")
    logger.info(f"总计保留 {results['total_valid_lines']} 行，移除 {results['total_invalid_lines']} 行")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python clean_jsonl.py <文件或目录路径> [--no-backup]")
        print("示例:")
        print("  python clean_jsonl.py ./local_vector_store/tool_workspace.jsonl")
        print("  python clean_jsonl.py ./local_vector_store")
        print("  python clean_jsonl.py ./local_vector_store --no-backup")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    backup = "--no-backup" not in sys.argv
    
    if path.is_file():
        clean_jsonl_file(path, backup=backup)
    elif path.is_dir():
        clean_jsonl_directory(path, backup=backup)
    else:
        print(f"错误: {path} 不是有效的文件或目录")
        sys.exit(1)

