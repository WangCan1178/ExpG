"""
专门查找第3440个字符位置的问题
"""
import json
from pathlib import Path


def check_file_at_position(file_path: Path, target_pos: int = 3439):
    """检查文件中第target_pos个字符位置的问题"""
    if not file_path.exists():
        return None
    
    print(f"\n{'='*80}")
    print(f"检查文件: {file_path}")
    print(f"{'='*80}\n")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        current_char_pos = 0
        
        for line_num, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            
            line_start_pos = current_char_pos
            line_end_pos = current_char_pos + len(stripped)
            
            # 检查这一行是否包含目标位置
            if line_start_pos <= target_pos < line_end_pos:
                print(f"✓ 找到包含位置 {target_pos} 的行: 第 {line_num} 行")
                print(f"  行在文件中的字符位置: {line_start_pos} 到 {line_end_pos}")
                print(f"  行长度: {len(stripped)} 字符")
                
                # 计算在行内的相对位置
                relative_pos = target_pos - line_start_pos
                print(f"  在行内的相对位置: {relative_pos}")
                
                # 尝试解析JSON
                try:
                    json.loads(stripped)
                    print(f"  ✓ JSON 格式正确")
                except json.JSONDecodeError as e:
                    error_pos = getattr(e, 'pos', None)
                    error_colno = getattr(e, 'colno', None)
                    
                    print(f"  ❌ JSON 解析错误:")
                    print(f"     错误信息: {e}")
                    print(f"     错误在行内位置: {error_pos}")
                    print(f"     错误列号: {error_colno}")
                    
                    if error_pos is not None:
                        # 显示错误位置附近的内容
                        start = max(0, error_pos - 100)
                        end = min(len(stripped), error_pos + 100)
                        context = stripped[start:end]
                        marker_pos = error_pos - start
                        
                        print(f"\n  错误位置附近的内容:")
                        print(f"  {context}")
                        print(f"  {' ' * marker_pos}^ (位置 {error_pos})")
                        
                        # 显示问题字符
                        if error_pos < len(stripped):
                            problem_char = stripped[error_pos]
                            print(f"\n  问题字符: '{problem_char}' (ASCII: {ord(problem_char)})")
                            
                            # 检查前后字符
                            if error_pos > 0:
                                prev_char = stripped[error_pos - 1]
                                print(f"  前一个字符: '{prev_char}' (ASCII: {ord(prev_char)})")
                            if error_pos + 1 < len(stripped):
                                next_char = stripped[error_pos + 1]
                                print(f"  后一个字符: '{next_char}' (ASCII: {ord(next_char)})")
                        
                        # 检查是否是未转义的引号问题
                        if error_pos < len(stripped):
                            # 向前查找最近的引号
                            quote_start = stripped.rfind('"', 0, error_pos)
                            if quote_start != -1:
                                # 检查引号是否被转义
                                escape_count = 0
                                i = quote_start - 1
                                while i >= 0 and stripped[i] == '\\':
                                    escape_count += 1
                                    i -= 1
                                
                                if escape_count % 2 == 0:  # 偶数个反斜杠，引号未转义
                                    print(f"\n  ⚠️  可能的问题: 在位置 {quote_start} 有一个未转义的引号")
                                    print(f"     这可能导致字符串提前结束")
                
                # 检查目标位置的具体内容
                if relative_pos < len(stripped):
                    char_at_target = stripped[relative_pos]
                    print(f"\n  位置 {target_pos} (行内位置 {relative_pos}) 的字符: '{char_at_target}' (ASCII: {ord(char_at_target)})")
                    
                    # 显示目标位置附近的内容
                    start = max(0, relative_pos - 50)
                    end = min(len(stripped), relative_pos + 50)
                    context = stripped[start:end]
                    marker_pos = relative_pos - start
                    
                    print(f"\n  目标位置附近的内容:")
                    print(f"  {context}")
                    print(f"  {' ' * marker_pos}^ (位置 {target_pos})")
                
                return {
                    'file': str(file_path),
                    'line_num': line_num,
                    'line_length': len(stripped),
                    'target_pos': target_pos,
                    'relative_pos': relative_pos,
                    'has_error': True
                }
            
            current_char_pos = line_end_pos + 1  # +1 for newline
    
    print(f"  未找到包含位置 {target_pos} 的行")
    return None


if __name__ == "__main__":
    import sys
    
    # 检查所有可能的文件
    base_dir = Path("/Users/xiaocan/Toolmem/local_vector_store")
    target_pos = 3439  # 字符位置从0开始，所以3439是第3440个字符
    
    files_to_check = [
        base_dir / "bfclqwen3-235b-a22b.jsonl",
        base_dir / "qwen3-235b" / "bfclqwen3-235b-a22b.jsonl",
    ]
    
    # 如果提供了文件路径参数
    if len(sys.argv) > 1:
        files_to_check = [Path(sys.argv[1])]
    
    # 如果提供了位置参数
    if len(sys.argv) > 2:
        target_pos = int(sys.argv[2])
    
    found_issue = False
    for file_path in files_to_check:
        result = check_file_at_position(file_path, target_pos)
        if result:
            found_issue = True
    
    if not found_issue:
        print("\n未在指定文件中找到问题。")
        print("可能的原因:")
        print("1. 问题文件路径不同")
        print("2. 文件在运行时动态生成")
        print("3. workspace_id 对应的文件路径不同")

