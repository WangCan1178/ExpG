import re
import ast

def fn(**kwargs):
    return kwargs

def get_api_call(model_output):
    api_call_pattern = r"\[(\w+)\((.*)\)\]"
    api_call_pattern = re.compile(api_call_pattern)
    match = api_call_pattern.search(model_output)
    if match:
        return match.group(0)
    else:
        return None

def parse_api_call(text):
    """
    解析API调用字符串，提取API名称和参数字典。
    支持列表和字典类型的参数，确保它们被正确解析为Python对象而不是字符串。
    """
    pattern = r"\[(\w+)\((.*)\)\]"
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)

    if not match:
        return None, {}
    
    api_name = match.group(1)
    params = match.group(2).strip()
    
    if not params:
        return api_name, {}
    
    param_dict = {}
    
    # 使用状态机方式解析参数，处理嵌套引号、列表、字典等复杂情况
    i = 0
    while i < len(params):
        # 跳过空白和逗号
        while i < len(params) and params[i] in ' ,':
            i += 1
        if i >= len(params):
            break
        
        # 找到 key
        key_start = i
        while i < len(params) and params[i] not in ' =':
            i += 1
        key = params[key_start:i].strip()
        
        if not key:
            break
        
        # 跳过等号和空白
        while i < len(params) and params[i] in ' =':
            i += 1
        if i >= len(params):
            break
        
        # 解析 value
        if params[i] == "'":
            # 单引号字符串
            i += 1
            value_start = i
            while i < len(params) and params[i] != "'":
                if params[i] == '\\' and i + 1 < len(params):
                    i += 2  # 跳过转义字符
                else:
                    i += 1
            value = params[value_start:i]
            i += 1  # 跳过结束引号
        elif params[i] == '"':
            # 双引号字符串
            i += 1
            value_start = i
            while i < len(params) and params[i] != '"':
                if params[i] == '\\' and i + 1 < len(params):
                    i += 2  # 跳过转义字符
                else:
                    i += 1
            value = params[value_start:i]
            i += 1  # 跳过结束引号
        elif params[i] == '[' or params[i] == '{':
            # 列表或字典字面量，需要找到匹配的结束括号
            bracket_start = i
            bracket_type = params[i]
            bracket_end = ']' if bracket_type == '[' else '}'
            i += 1
            depth = 1
            
            # 找到匹配的结束括号
            while i < len(params) and depth > 0:
                if params[i] == '\\' and i + 1 < len(params):
                    i += 2  # 跳过转义字符
                    continue
                if params[i] == bracket_type:
                    depth += 1
                elif params[i] == bracket_end:
                    depth -= 1
                elif params[i] in ('"', "'"):
                    # 跳过字符串内容
                    quote_char = params[i]
                    i += 1
                    while i < len(params) and params[i] != quote_char:
                        if params[i] == '\\' and i + 1 < len(params):
                            i += 2
                        else:
                            i += 1
                i += 1
            
            # 提取完整的列表/字典字符串
            value_str = params[bracket_start:i]
            
            # 尝试解析为Python对象
            try:
                value = ast.literal_eval(value_str)
            except (ValueError, SyntaxError):
                # 解析失败，保持为字符串
                value = value_str
        elif params[i].isdigit() or (params[i] == '-' and i + 1 < len(params) and params[i+1].isdigit()):
            # 数字（包括负数）
            value_start = i
            if params[i] == '-':
                i += 1
            while i < len(params) and (params[i].isdigit() or params[i] == '.'):
                i += 1
            value_str = params[value_start:i]
            try:
                if '.' in value_str:
                    value = float(value_str)
                else:
                    value = int(value_str)
            except ValueError:
                value = value_str
        elif i + 4 <= len(params) and params[i:i+4].lower() == 'true':
            # 布尔值 True
            value = True
            i += 4
        elif i + 5 <= len(params) and params[i:i+5].lower() == 'false':
            # 布尔值 False
            value = False
            i += 5
        elif i + 4 <= len(params) and params[i:i+4].lower() == 'none':
            # None值
            value = None
            i += 4
        else:
            # 无引号值，取到下一个逗号或结束
            value_start = i
            while i < len(params) and params[i] != ',':
                i += 1
            value = params[value_start:i].strip()
            
            # 尝试解析为Python字面量（列表、字典、数字等）
            value_stripped = value.strip()
            if (value_stripped.startswith('[') and value_stripped.endswith(']')) or \
               (value_stripped.startswith('{') and value_stripped.endswith('}')):
                try:
                    value = ast.literal_eval(value_stripped)
                except (ValueError, SyntaxError):
                    pass
        
        param_dict[key] = value
    
    return api_name, param_dict

