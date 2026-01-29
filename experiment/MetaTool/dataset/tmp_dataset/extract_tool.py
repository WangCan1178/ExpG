import json, pathlib, collections, random
base = pathlib.Path('/Users/xiaocan/Toolmem/experiment/MetaTool/dataset/tmp_dataset')
files = [
    # base/'Task1.json',
    base/'Task2-Subtask1.json',
    base/'Task2-Subtask2.json',
    base/'Task2-Subtask3.json',
    base/'Task2-Subtask4.json',
]

# 读取每个 subtask 的数据
per_file_data = {}
per_file_tool_counts = {}
# 统计所有文件的全局工具使用总数
global_tool_counts = collections.Counter()

for fp in files:
    with fp.open() as f:
        data = json.load(f)
    per_file_data[fp.name] = data
    c = collections.Counter()
    for item in data:
        tool_field = item.get('tool')
        if tool_field is None:
            continue
        tools = tool_field if isinstance(tool_field, list) else [tool_field]
        for tool in tools:
            c[tool] += 1
            global_tool_counts[tool] += 1
    per_file_tool_counts[fp.name] = c

# 统计每个 subtask 用到的不同工具数目
print("每个 subtask 用到的不同工具数目：")
for fp in files:
    fname = fp.name
    unique_tools = len(per_file_tool_counts[fname])
    print(f'{fname}: {unique_tools} 个不同的工具')
print()

# 打印每个 subtask 文件中每个不同工具的调用次数（按从多到少排序）
print("每个 subtask 文件中每个工具的调用次数（按从多到少排序）：")
for fp in files:
    fname = fp.name
    print(f'\n{fname}:')
    for tool, count in per_file_tool_counts[fname].most_common():
        print(f'  {tool}: {count} 次')
print()

# 打印4个文件的不同工具的总调用次数（按从多到少排序）
print("4个文件的不同工具的总调用次数（按从多到少排序）：")
for tool, count in global_tool_counts.most_common():
    print(f'  {tool}: {count} 次')
print()

# 基于全局统计，选出使用次数最多的 50 个工具
top50_tools_global = set([t for t, _ in global_tool_counts.most_common(100)])
# 基于全局统计，选出使用次数最少的 50 个工具
least50_tools_global = set([t for t, _ in sorted(global_tool_counts.items(), key=lambda x: x[1])[:50]])

# 对每个 subtask：从前 50 个最常调用的工具中，
# 对每个工具随机抽取最多 5 条样本（若不足 5 条则全选），
# 然后合并到 task2.json 中，并带上 id 和来源 subtask。
selected_indices_per_file = {}
selected_counts_per_file = {}
all_selected = []

for fp in files:
    fname = fp.name
    data = per_file_data[fname]
    
    # 统计该 subtask 中属于前 50 个最常调用工具涉及到的样本索引
    tool_to_indices = collections.defaultdict(list)
    for idx, item in enumerate(data):
        tool_field = item.get('tool')
        if tool_field is None:
            continue
        tools = tool_field if isinstance(tool_field, list) else [tool_field]
        for tool in tools:
            if tool in top50_tools_global:
                tool_to_indices[tool].append(idx)

    selected_indices = set()
    for tool in top50_tools_global:
        indices = tool_to_indices.get(tool, [])
        if not indices:
            continue
        if len(indices) > 10:
            chosen = random.sample(indices, 10)
        else:
            chosen = indices
        selected_indices.update(chosen)

    selected_indices_per_file[fname] = selected_indices
    selected_counts_per_file[fname] = len(selected_indices)

    # 将选中的样本加入总集合，并增加 id / source_subtask 字段
    for idx in sorted(selected_indices):
        item = data[idx]
        new_item = dict(item)
        new_item['id'] = f'{fname}-{idx}'
        new_item['source_subtask'] = fname
        all_selected.append(new_item)

# 打印每个 subtask 选出的样本数
total_selected = 0
for fp in files:
    fname = fp.name
    cnt = selected_counts_per_file.get(fname, 0)
    total_selected += cnt
    print(f'{fname}: {cnt} samples selected')
print(f'Total selected samples: {total_selected}')

# 保存到 task2.json
out_path = base / 'task2test.json'
with out_path.open('w') as f:
    json.dump(all_selected, f, ensure_ascii=False, indent=2)

print(f'\n文件已保存到 task2.json，共 {len(all_selected)} 个样本')