import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # 为 3D 绘图注册投影
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib import font_manager as fm
import json
from pathlib import Path
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Use local Consolas font for all text (legends, axes, etc.)
font_path = '/Users/xiaocan/Toolmem/consola.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Consolas'
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# Global font size for legends and axis labels
FONT_SIZE = 9  # slightly smaller than Matplotlib default (~10)

# -----------------------
# Utils: distributions & metrics
# -----------------------
def scores_to_classes(scores: np.ndarray):
    """Map each 10-dim binary vector to an equivalence class id via tuple."""
    return [tuple(row.tolist()) for row in scores]

def empirical_dist(class_ids, class_to_idx=None, num_classes=None):
    """Return probability vector over a fixed class index space."""
    if class_to_idx is None:
        uniq = sorted(set(class_ids))
        class_to_idx = {c: i for i, c in enumerate(uniq)}
    if num_classes is None:
        num_classes = len(class_to_idx)

    counts = np.zeros(num_classes, dtype=np.float64)
    for c in class_ids:
        counts[class_to_idx[c]] += 1.0
    p = counts / max(counts.sum(), 1.0)
    return p, class_to_idx

def kl_divergence(p, q, eps=1e-12):
    """KL(p||q) with epsilon smoothing."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))

def negative_kl_similarity(p, q, eps=1e-12):
    """负KL散度：-KL(p||q)，越高表示分布越接近"""
    return -kl_divergence(p, q, eps=eps)

def entropy(p, eps=1e-12):
    """计算熵：H(X) = -Σ p(x) * log(p(x))"""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > eps]  # 只考虑非零概率
    if len(p) == 0:
        return 0.0
    p = p / p.sum()  # 归一化
    return float(-np.sum(p * np.log(p)))

def diversity_entropy(selected_class_ids, class_to_idx=None, num_classes=None):
    """多样性指标：基于类别分布的熵（entropy）"""
    p, _ = empirical_dist(selected_class_ids, class_to_idx=class_to_idx, num_classes=num_classes)
    return entropy(p)


def get_truncated_cmap(name, minval=0.15, maxval=1.0, n=256):
    """截断默认 colormap，避免浅色段变成纯白。"""
    base = plt.get_cmap(name)
    colors = base(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list(f"{name}_trunc", colors)

# -----------------------
# Distribution loading from jsonl files
# -----------------------
def extract_success_rates_from_jsonl(jsonl_path):
    """从 jsonl 文件提取所有成功率"""
    success_rates = []
    result_word_counts = []  # 记录每个 result 转换为字符串后的单词数量
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # 从 metadata 中获取 tool_call_results
                metadata = data.get('metadata', {})
                tool_call_results = metadata.get('tool_call_results', [])
                
                # 如果没有在 metadata 中，尝试直接在顶层
                if not tool_call_results:
                    tool_call_results = data.get('tool_call_results', [])
                
                for result in tool_call_results:
                    # 只统计特定字段的单词数量：evaluation, score, success, time_cost, token_cost, input, output
                    fields_to_count = []
                    if 'evaluation' in result:
                        fields_to_count.append(str(result['evaluation']))
                    if 'score' in result:
                        fields_to_count.append(str(result['score']))
                    if 'success' in result:
                        fields_to_count.append(str(result['success']))
                    if 'time_cost' in result:
                        fields_to_count.append(str(result['time_cost']))
                    if 'token_cost' in result:
                        fields_to_count.append(str(result['token_cost']))
                    if 'input' in result:
                        fields_to_count.append(str(result['input']))
                    if 'output' in result:
                        fields_to_count.append(str(result['output']))
                    
                    # 合并所有字段并统计单词数量
                    combined_str = ' '.join(fields_to_count)
                    word_count = len(combined_str.split())
                    result_word_counts.append(word_count)
                    
                    if 'scores' in result:
                        scores_array = result['scores']
                        if isinstance(scores_array, list) and len(scores_array) > 0:
                            scores = np.array(scores_array)
                            # 检查是否是二进制
                            unique_vals = np.unique(scores)
                            if len(unique_vals) <= 2 and all(v in [0, 1] for v in unique_vals):
                                success_rate = float(np.sum(scores) / len(scores))
                                success_rates.append(success_rate)
                            
            except json.JSONDecodeError:
                continue
            except Exception:
                continue
    
    # 计算每个 result 的平均单词数量
    avg_word_count = float(np.mean(result_word_counts)) if result_word_counts else 0.0
    print(f"文件 {jsonl_path}: 每个 result 的平均单词数量 = {avg_word_count:.2f} (共 {len(result_word_counts)} 个 result)")
    
    return np.array(success_rates)


def extract_scores_arrays_from_jsonl(jsonl_path):
    """从 jsonl 文件提取所有 scores 数组"""
    all_scores = []
    result_word_counts = []  # 记录每个 result 转换为字符串后的单词数量
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # 从 metadata 中获取 tool_call_results
                metadata = data.get('metadata', {})
                tool_call_results = metadata.get('tool_call_results', [])
                
                # 如果没有在 metadata 中，尝试直接在顶层
                if not tool_call_results:
                    tool_call_results = data.get('tool_call_results', [])
                
                for result in tool_call_results:
                    # 只统计特定字段的单词数量：evaluation, score, success, time_cost, token_cost, input, output
                    fields_to_count = []
                    if 'evaluation' in result:
                        fields_to_count.append(str(result['evaluation']))
                    if 'score' in result:
                        fields_to_count.append(str(result['score']))
                    if 'success' in result:
                        fields_to_count.append(str(result['success']))
                    if 'time_cost' in result:
                        fields_to_count.append(str(result['time_cost']))
                    if 'token_cost' in result:
                        fields_to_count.append(str(result['token_cost']))
                    if 'input' in result:
                        fields_to_count.append(str(result['input']))
                    if 'output' in result:
                        fields_to_count.append(str(result['output']))
                    
                    # 合并所有字段并统计单词数量
                    combined_str = ' '.join(fields_to_count)
                    word_count = len(combined_str.split())
                    result_word_counts.append(word_count)
                    
                    if 'scores' in result:
                        scores_array = result['scores']
                        if isinstance(scores_array, list) and len(scores_array) > 0:
                            scores = np.array(scores_array, dtype=int)
                            # 检查是否是二进制
                            unique_vals = np.unique(scores)
                            if len(unique_vals) <= 2 and all(v in [0, 1] for v in unique_vals):
                                all_scores.append(scores)
                            
            except json.JSONDecodeError:
                continue
            except Exception:
                continue
    
    # 计算每个 result 的平均单词数量
    avg_word_count = float(np.mean(result_word_counts)) if result_word_counts else 0.0
    print(f"文件 {jsonl_path}: 每个 result 的平均单词数量 = {avg_word_count:.2f} (共 {len(result_word_counts)} 个 result)")
    
    return all_scores


def estimate_mixed_distribution_params(success_rates):
    """估计混合分布参数（完全成功 + Beta分布）"""
    n_total = len(success_rates)
    n_perfect = np.sum(success_rates == 1.0)
    partial_rates = success_rates[(success_rates > 0) & (success_rates < 1)]
    
    # 完全成功的概率
    p_success = n_perfect / n_total if n_total > 0 else 0.0
    
    # Beta 分布参数（只使用部分成功的数据）
    if len(partial_rates) > 0 and HAS_SCIPY:
        try:
            alpha, beta, _, _ = stats.beta.fit(partial_rates, floc=0, fscale=1)
        except:
            # 如果拟合失败，使用矩估计
            mean_partial = np.mean(partial_rates)
            var_partial = np.var(partial_rates)
            if var_partial > 0:
                alpha = mean_partial * (mean_partial * (1 - mean_partial) / var_partial - 1)
                beta = (1 - mean_partial) * (mean_partial * (1 - mean_partial) / var_partial - 1)
                alpha = max(0.1, alpha)
                beta = max(0.1, beta)
            else:
                alpha, beta = 2.0, 1.2
    else:
        # 简化的矩估计
        if len(partial_rates) > 0:
            mean_partial = np.mean(partial_rates)
            var_partial = np.var(partial_rates)
            if var_partial > 0:
                alpha = mean_partial * (mean_partial * (1 - mean_partial) / var_partial - 1)
                beta = (1 - mean_partial) * (mean_partial * (1 - mean_partial) / var_partial - 1)
                alpha = max(0.1, alpha)
                beta = max(0.1, beta)
            else:
                alpha, beta = 2.0, 1.2
        else:
            alpha, beta = 2.0, 1.2
    
    return {
        'p_success': float(p_success),
        'alpha': float(alpha),
        'beta': float(beta),
    }


def load_distribution_params_from_jsonls(jsonl_paths, weights=None):
    """
    从多个 jsonl 文件加载分布参数并合并
    学习每个维度是 0 或 1 的概率
    
    Args:
        jsonl_paths: jsonl 文件路径列表，例如 ['toole.jsonl', 'apibank-8b.jsonl', 'bfcl-qwen3-8b.jsonl']
        weights: 每个文件的权重（可选），如果不提供则按样本数加权
    
    Returns:
        分布参数字典，包含 'dim_probs'（每个维度的概率列表）
    """
    all_scores_arrays = []
    file_scores = []
    
    for jsonl_path in jsonl_paths:
        path = Path(jsonl_path)
        if not path.exists():
            print(f"警告: 文件不存在，跳过: {jsonl_path}")
            continue
        scores_arrays = extract_scores_arrays_from_jsonl(path)
        if len(scores_arrays) > 0:
            all_scores_arrays.extend(scores_arrays)
            file_scores.append(scores_arrays)
            print(f"从 {jsonl_path} 加载了 {len(scores_arrays)} 个 scores 数组")
    
    if len(all_scores_arrays) == 0:
        print("警告: 未找到任何数据，使用默认参数")
        return {'dim_probs': None}  # 返回 None 表示使用默认方法
    
    # 合并所有 scores 数组
    if weights is not None:
        # 按权重加权：对每个文件的数据按权重采样
        all_scores_arrays = []
        for i, scores_list in enumerate(file_scores):
            weight = weights[i] if i < len(weights) else 1.0
            n_samples = int(len(scores_list) * weight)
            if n_samples > 0:
                indices = np.random.choice(len(scores_list), size=min(n_samples, len(scores_list)), replace=False)
                all_scores_arrays.extend([scores_list[j] for j in indices])
    
    # 找到最大维度
    max_dim = max(len(scores) for scores in all_scores_arrays)
    print(f"最大维度: {max_dim}")
    
    # 按维度分组统计（剔除噪声维度，例如 d=6）
    dim_to_scores = {}  # {维度: [该维度的所有 scores 数组]}
    for scores in all_scores_arrays:
        d = len(scores)
        # 6 维被认为是噪声，不参与统计
        if d == 6:
            continue
        if d not in dim_to_scores:
            dim_to_scores[d] = []
        dim_to_scores[d].append(scores)
    
    # 对每个维度，计算每个位置是 1 的概率
    dim_probs_dict = {}  # {维度: [p_0, p_1, ..., p_{d-1}]}
    for d, scores_list in dim_to_scores.items():
        # 将所有 scores 数组合并成一个矩阵
        scores_matrix = np.array(scores_list)  # shape: (n_samples, d)
        # 计算每个维度（列）的平均值，即该维度是 1 的概率
        probs = np.mean(scores_matrix, axis=0)  # shape: (d,)
        dim_probs_dict[d] = probs.tolist()
        print(f"维度 {d}: 各位置为1的概率 = {[f'{p:.3f}' for p in probs]}")

    # 进一步汇总：把所有维度/所有位置的概率取平均，得到一个全局 Bernoulli(p)
    # 需求：不管目标维度 d 是多少，都用同一个 p 来独立采样每个位置。
    all_pos_probs = []
    for probs in dim_probs_dict.values():
        all_pos_probs.extend(probs)
    global_bernoulli_p = float(np.mean(all_pos_probs)) if len(all_pos_probs) > 0 else 0.5
    print(f"全局平均 Bernoulli(p) = {global_bernoulli_p:.6f}（用于任意维度的 i.i.d. 采样）")

    return {'dim_probs': dim_probs_dict, 'global_bernoulli_p': global_bernoulli_p-0.1}


def generate_scores_from_mixed_distribution(n_samples, d, dist_params, rng):
    """
    使用每个维度独立的概率分布生成 scores 数组
    
    Args:
        n_samples: 生成的样本数
        d: scores 的维度（每个样本是一个 d 维的二进制向量）
        dist_params: 分布参数字典，包含 'dim_probs'（每个维度的概率字典）
        rng: 随机数生成器
    
    Returns:
        scores 数组，形状为 (n_samples, d)
    """
    dim_probs = dist_params.get('dim_probs')
    global_p = dist_params.get('global_bernoulli_p')

    # 新需求：使用全局平均 Bernoulli(p)，不区分维度/位置
    if global_p is not None:
        p = float(global_p)
        p = min(1.0, max(0.0, p))
        scores = (rng.random((n_samples, d)) < p).astype(int)
        return scores

    # 兼容旧逻辑：如果提供了维度概率，使用“每个位置不同概率”的伯努利分布
    if dim_probs is not None and d in dim_probs:
        probs = np.array(dim_probs[d])
        scores = (rng.random((n_samples, d)) < probs).astype(int)
        return scores
    elif dim_probs is not None:
        # 如果该维度没有数据，尝试使用最接近的维度
        available_dims = sorted(dim_probs.keys())
        if available_dims:
            closest_dim = min(available_dims, key=lambda x: abs(x - d))
            print(f"警告: 维度 {d} 没有数据，使用维度 {closest_dim} 的概率")
            probs = np.array(dim_probs[closest_dim])
            if d > len(probs):
                mean_prob = np.mean(probs)
                probs = np.concatenate([probs, np.full(d - len(probs), mean_prob)])
            else:
                probs = probs[:d]
            scores = (rng.random((n_samples, d)) < probs).astype(int)
            return scores
    
    # 如果没有提供维度概率，回退到旧的混合分布方法（向后兼容）
    if 'p_success' in dist_params:
        p_success = dist_params['p_success']
        alpha = dist_params['alpha']
        beta = dist_params['beta']
        
        scores_list = []
        for _ in range(n_samples):
            # 使用混合分布生成成功率
            if rng.random() < p_success:
                success_rate = 1.0
            else:
                success_rate = rng.beta(alpha, beta)
            
            # 根据成功率生成二进制 scores
            n_ones = int(round(success_rate * d))
            n_ones = max(0, min(d, n_ones))  # 确保在 [0, d] 范围内
            
            # 生成一个 d 维的二进制向量，其中 n_ones 个为 1
            score_vec = np.zeros(d, dtype=int)
            if n_ones > 0:
                indices = rng.choice(d, size=n_ones, replace=False)
                score_vec[indices] = 1
            
            scores_list.append(score_vec)
        
        return np.array(scores_list)
    
    # 如果都没有，使用默认的均匀概率
    print(f"警告: 未找到维度 {d} 的概率，使用默认均匀概率 0.5")
    scores = (rng.random((n_samples, d)) < 0.5).astype(int)
    return scores

# -----------------------
# Baselines / Methods
# -----------------------
def diversity_sampling(class_ids, Q, rng):
    """
    近似“最大多样性”的采样：
    - 尽量让每个类别被选中的次数接近（类别层面接近均匀分布）
    - 在每个类别内部均匀随机选样本
    """
    # 按类别分组索引（统一转成 tuple，保证可哈希且与其它地方一致）
    class_to_indices = {}
    for i, c in enumerate(class_ids):
        key = c if isinstance(c, tuple) else tuple(c)
        class_to_indices.setdefault(key, []).append(i)

    classes = list(class_to_indices.keys())
    # 打乱每个类别内部的样本顺序
    for c in classes:
        rng.shuffle(class_to_indices[c])

    # 记录每个类别已选数量和当前指针
    counts = {c: 0 for c in classes}
    ptr = {c: 0 for c in classes}

    selected = []

    while len(selected) < Q:
        # 还有剩余样本的类别
        available_classes = [c for c in classes if ptr[c] < len(class_to_indices[c])]
        if not available_classes:
            break  # 理论上只有在 Q > N 才会触发

        # 从“当前已选数量最少”的类别中随机挑选一个
        min_count = min(counts[c] for c in available_classes)
        candidate_classes = [c for c in available_classes if counts[c] == min_count]
        # numpy 的 choice 可能返回 numpy 对象，这里统一转成 tuple 作为 key
        chosen = rng.choice(candidate_classes)
        c = chosen if isinstance(chosen, tuple) else tuple(chosen)

        idx = class_to_indices[c][ptr[c]]
        ptr[c] += 1
        counts[c] += 1
        selected.append(idx)

    return np.array(selected, dtype=int)

def proportional_sampling(class_ids, Q, rng):
    """
    Instance-balanced sampling: 每个样本的概率相同（1/N）
    直接从所有 N 个样本中均匀随机无放回采样，不考虑类别。
    """
    N = len(class_ids)
    # 直接从所有样本索引中均匀随机无放回采样
    selected = rng.choice(N, size=min(Q, N), replace=False)
    return np.array(selected, dtype=int)

def hybrid_sampling(class_ids, Q, rng):
    """
    兼顾“多样性”和“按原始比例”的采样：
    - 先保证每个非空类别至少采 1 条（如果 Q >= 类别数）；
    - 剩余名额按类别在全集中的占比近似按比例分配；
    - 同时考虑每个类别的上限（不能超过该类总样本数）。
    这是对 _sample_calls_by_score_distribution 的“类别版”简化移植。
    """
    N_total = len(class_ids)
    if Q <= 0 or N_total == 0:
        return np.array([], dtype=int)
    if Q >= N_total:
        return np.arange(N_total, dtype=int)

    # 1) 按类别分组索引
    class_to_indices = {}
    for i, c in enumerate(class_ids):
        key = c if isinstance(c, tuple) else tuple(c)
        class_to_indices.setdefault(key, []).append(i)

    classes = list(class_to_indices.keys())
    C = len(classes)

    # 每类容量
    counts = [len(class_to_indices[c]) for c in classes]  # n_k

    # Q 小于类别数：只能选 Q 个类别，各 1 条，类别选择按 n_k 近似按比例
# Q 小于类别数：确定性选择 size 最大的 Q 个类别，每类选 1 条
    if Q <= C:
        # 按类别容量 counts 从大到小排序，取前 Q 个
        order = np.argsort(-np.array(counts))  # 降序
        top_class_indices = order[:Q]

        selected = []
        for ci in top_class_indices:
            c = classes[int(ci)]
            idxs = class_to_indices[c]
            # 类内仍然随机选 1 条（与你算法一致）
            selected.append(int(rng.choice(idxs)))

        return np.array(selected, dtype=int)


    # Q >= C：每类至少 1 条，剩余 R = Q - C 按比例分配
    R = Q - C

    total_n = sum(counts)
    if total_n == 0:
        return np.array([], dtype=int)
    p_list = [cnt / total_n for cnt in counts]  # p_k

    # 3) 连续理想值 raw_k = R * p_k
    raw = [R * p for p in p_list]
    # 整数部分 extra_k = floor(raw_k)
    extra = [int(np.floor(x)) for x in raw]
    used = sum(extra)
    remain = R - used

    # 4) 将剩余的 remain 个名额按小数部分从大到小分配
    if remain > 0:
        frac = [raw[i] - extra[i] for i in range(C)]
        indices = list(range(C))
        indices.sort(key=lambda i: frac[i], reverse=True)
        for i in indices[:remain]:
            extra[i] += 1

    # 5) 得到每类初始配额：quota_k = 1 + extra_k
    quota = [1 + e for e in extra]  # 对应 classes 的顺序

    # 6) 考虑每类的容量上限：quota_k <= count_k，多余的名额放入 pool 重新分配
    # pool = 0
    # for i in range(C):
    #     cap = counts[i]
    #     if quota[i] > cap:
    #         overflow = quota[i] - cap
    #         quota[i] = cap
    #         pool += overflow
    pool = Q-sum(quota)
    if pool > 0:
        # 还有空间的类别索引
        available_indices = [i for i in range(C) if quota[i] < counts[i]]
        if available_indices:
            # 按 p_k 从大到小排序，优先给大类
            available_indices.sort(key=lambda i: p_list[i], reverse=True)
            idx_cycle = 0
            while pool > 0 and available_indices:
                i = available_indices[idx_cycle]
                if quota[i] < counts[i]:
                    quota[i] += 1
                    pool -= 1
                idx_cycle = (idx_cycle + 1) % len(available_indices)
                if all(quota[j] >= counts[j] for j in available_indices):
                    break
        # pool 剩余说明所有类都满了，此时总采样数 < Q，但不会超过 Q

    # 7) 在每个类别内部按 quota_k 采样
    selected = []
    for i, c in enumerate(classes):
        k_quota = quota[i]
        if k_quota <= 0:
            continue
        idxs = class_to_indices[c]
        if len(idxs) <= k_quota:
            selected.extend(idxs)
        else:
            # 类内无放回随机采样
            chosen = rng.choice(len(idxs), size=k_quota, replace=False)
            selected.extend([idxs[j] for j in chosen])

    # 理论上 len(selected) <= Q
    if len(selected) > Q:
        # 极小概率的边界情况，裁剪到 Q
        selected = list(rng.choice(selected, size=Q, replace=False))

    return np.array(selected, dtype=int)

def our_method_selection(class_ids, Q, rng):
    """
    Placeholder for YOUR method (DExTM equivalence-class selection algorithm).
    Implement later.

    Expected:
      input: class_ids (len N), Q
      output: np.ndarray of selected indices (len Q), without replacement
    """
    raise NotImplementedError("TODO: implement your method here.")

# -----------------------
# Experiment
# -----------------------
def run_experiment(seed=42, N=100, d=10, num_runs=1, plot=True, dist_params=None):
    """
    运行实验，可以运行多次取平均
    
    Args:
        seed: 随机种子
        N: 数据集大小
        d: 向量维度
        num_runs: 运行次数（多次运行取平均）
        plot: 是否绘图
        dist_params: 分布参数字典，如果为 None 则使用默认的长尾分布
    """
    if num_runs == 1:
        return run_once(seed, N, d, plot, dist_params=dist_params)
    
    # 多次运行取平均
    all_similarity_div = []
    all_similarity_prop = []
    all_similarity_hybrid = []
    all_div_div = []
    all_div_prop = []
    all_div_hybrid = []
    Q_list = None
    
    for run_idx in range(num_runs):
        result = run_once(seed + run_idx, N, d, plot=False, verbose=False, dist_params=dist_params)
        if Q_list is None:
            Q_list = result['Q_list']
            all_similarity_div = [[] for _ in Q_list]
            all_similarity_prop = [[] for _ in Q_list]
            all_similarity_hybrid = [[] for _ in Q_list]
            all_div_div = [[] for _ in Q_list]
            all_div_prop = [[] for _ in Q_list]
            all_div_hybrid = [[] for _ in Q_list]
        
        for i, q in enumerate(Q_list):
            all_similarity_div[i].append(result['similarity_div'][i])
            all_similarity_prop[i].append(result['similarity_prop'][i])
            all_similarity_hybrid[i].append(result['similarity_hybrid'][i])
            all_div_div[i].append(result['div_div'][i])
            all_div_prop[i].append(result['div_prop'][i])
            all_div_hybrid[i].append(result['div_hybrid'][i])
    
    # 计算平均值
    similarity_div = [np.mean(vals) for vals in all_similarity_div]
    similarity_prop = [np.mean(vals) for vals in all_similarity_prop]
    similarity_hybrid = [np.mean(vals) for vals in all_similarity_hybrid]
    div_div = [np.mean(vals) for vals in all_div_div]
    div_prop = [np.mean(vals) for vals in all_div_prop]
    div_hybrid = [np.mean(vals) for vals in all_div_hybrid]
    
    # 打印结果
    print(f"\n=== {num_runs}次运行的平均结果 ===")
    print("Q\tmethod\t相似度(-KL)\tdiversity(entropy)")
    for i, Q in enumerate(Q_list):
        print(f"{Q}\tdiversity\t{similarity_div[i]:.6f}\t\t{div_div[i]:.6f}")
        print(f"{Q}\tprop\t{similarity_prop[i]:.6f}\t\t{div_prop[i]:.6f}")
        print(f"{Q}\thybrid\t{similarity_hybrid[i]:.6f}\t\t{div_hybrid[i]:.6f}")
    
    # 绘图
    if plot:
        plot_results(Q_list, similarity_div, similarity_prop, similarity_hybrid,
                     div_div, div_prop, div_hybrid)
    
    return {
        'Q_list': Q_list,
        'similarity_div': similarity_div,
        'similarity_prop': similarity_prop,
        'similarity_hybrid': similarity_hybrid,
        'div_div': div_div,
        'div_prop': div_prop,
        'div_hybrid': div_hybrid,
    }


def evaluate_single_setting(seed, N, d, num_runs=5, q_ratio=0.1, dist_params=None):
    """
    在固定的 (N, d) 下，多次运行，返回：
    - 两种方法在一个固定 Q 下的平均相似度(-KL) 和 多样性(熵)
    这里 Q = max(10, int(q_ratio * N))，并且保证 Q < N。
    
    Args:
        dist_params: 分布参数字典，如果为 None 则使用默认的长尾分布
    """
    all_sim_div = []
    all_sim_prop = []
    all_sim_hybrid = []
    all_div_div = []
    all_div_prop = []
    all_div_hybrid = []

    for run_idx in range(num_runs):
        rng = np.random.default_rng(seed + run_idx)

        # 使用新的分布生成方式（如果提供了 dist_params）或旧的指数衰减分布
        if dist_params is not None:
            # 使用混合分布生成 scores
            scores = generate_scores_from_mixed_distribution(N, d, dist_params, rng)
            class_ids = scores_to_classes(scores)
        else:
            # 旧的指数衰减分布（向后兼容）
            K = 2 ** d
            raw = np.exp(-0.7 * np.arange(K, dtype=np.float64))
            p_true = raw / raw.sum()
            all_classes = [tuple(int(b) for b in np.binary_repr(i, width=d)) for i in range(K)]
            sampled_class_indices = rng.choice(K, size=N, p=p_true)
            class_ids = [all_classes[i] for i in sampled_class_indices]

        # 全量分布
        p_full, class_to_idx = empirical_dist(class_ids)
        num_classes = len(class_to_idx)

        # 选一个固定 Q
        Q = max(10, int(q_ratio * N))
        Q = min(Q, N - 1)  # 保证小于 N

        # Diversity
        idx_d = diversity_sampling(class_ids, Q, rng)
        cls_d = [class_ids[i] for i in idx_d]
        p_d, _ = empirical_dist(cls_d, class_to_idx=class_to_idx, num_classes=num_classes)
        sim_d = negative_kl_similarity(p_d, p_full)
        div_d = diversity_entropy(cls_d, class_to_idx=class_to_idx, num_classes=num_classes)

        # Proportional
        idx_p = proportional_sampling(class_ids, Q, rng)
        cls_p = [class_ids[i] for i in idx_p]
        p_p, _ = empirical_dist(cls_p, class_to_idx=class_to_idx, num_classes=num_classes)
        sim_p = negative_kl_similarity(p_p, p_full)
        div_p = diversity_entropy(cls_p, class_to_idx=class_to_idx, num_classes=num_classes)

        # Hybrid
        idx_h = hybrid_sampling(class_ids, Q, rng)
        cls_h = [class_ids[i] for i in idx_h]
        p_h, _ = empirical_dist(cls_h, class_to_idx=class_to_idx, num_classes=num_classes)
        sim_h = negative_kl_similarity(p_h, p_full)
        div_h = diversity_entropy(cls_h, class_to_idx=class_to_idx, num_classes=num_classes)

        all_sim_div.append(sim_d)
        all_div_div.append(div_d)
        all_sim_prop.append(sim_p)
        all_div_prop.append(div_p)
        all_sim_hybrid.append(sim_h)
        all_div_hybrid.append(div_h)

    return (
        float(np.mean(all_sim_div)),
        float(np.mean(all_div_div)),
        float(np.mean(all_sim_prop)),
        float(np.mean(all_div_prop)),
        float(np.mean(all_sim_hybrid)),
        float(np.mean(all_div_hybrid)),
    )


def run_3d_experiment(
    seed=42,
    N_list=tuple(range(50, 301, 25)),  # 50, 75, ..., 300 更细的 N 网格
    d_list=tuple(range(3, 21, 1)),     # 3, 4, ..., 20 更细的 d 网格
    num_runs=3,                        # 为了运行时间，减少单点重复次数
    q_ratio=0.1,
    plot=True,
    dist_params=None,
):
    """
    运行 3D 实验：
    - x 轴：d（向量维度）
    - y 轴：N（数据集大小）
    - z 轴：相似度(-KL) 或 多样性(熵)
    分别对两种方法（Diversity / Proportional）画两张 3D 曲面图。
    
    Args:
        dist_params: 分布参数字典，如果为 None 则使用默认的长尾分布
    """
    N_list = list(N_list)
    d_list = list(d_list)

    sim_div_mat = np.zeros((len(d_list), len(N_list)), dtype=np.float64)
    div_div_mat = np.zeros_like(sim_div_mat)
    sim_prop_mat = np.zeros_like(sim_div_mat)
    div_prop_mat = np.zeros_like(sim_div_mat)
    sim_hybrid_mat = np.zeros_like(sim_div_mat)
    div_hybrid_mat = np.zeros_like(sim_div_mat)

    for i, d in enumerate(d_list):
        for j, N in enumerate(N_list):
            (
                s_div,
                dv_div,
                s_prop,
                dv_prop,
                s_hybrid,
                dv_hybrid,
            ) = evaluate_single_setting(seed, N, d, num_runs=num_runs, q_ratio=q_ratio, dist_params=dist_params)
            sim_div_mat[i, j] = s_div
            div_div_mat[i, j] = dv_div
            sim_prop_mat[i, j] = s_prop
            div_prop_mat[i, j] = dv_prop
            sim_hybrid_mat[i, j] = s_hybrid
            div_hybrid_mat[i, j] = dv_hybrid
            print(f"d={d}, N={N} -> diversity(sim={s_div:.4f}, H={dv_div:.4f}), "
                  f"prop(sim={s_prop:.4f}, H={dv_prop:.4f}), "
                  f"hybrid(sim={s_hybrid:.4f}, H={dv_hybrid:.4f})")

    if plot:
        plot_3d_results(d_list, N_list,
                        sim_div_mat, sim_prop_mat, sim_hybrid_mat,
                        div_div_mat, div_prop_mat, div_hybrid_mat)

    return {
        "d_list": d_list,
        "N_list": N_list,
        "sim_div": sim_div_mat,
        "sim_prop": sim_prop_mat,
        "sim_hybrid": sim_hybrid_mat,
        "div_div": div_div_mat,
        "div_prop": div_prop_mat,
        "div_hybrid": div_hybrid_mat,
    }


def evaluate_setting_d_q(seed, N, d, Q, num_runs=5, dist_params=None):
    """
    在固定的 (N, d, Q) 下，多次运行，返回两种方法的
    - 平均相似度(-KL)
    - 平均多样性(熵)
    
    Args:
        dist_params: 分布参数字典，如果为 None 则使用默认的长尾分布
    """
    all_sim_div = []
    all_sim_prop = []
    all_sim_hybrid = []
    all_div_div = []
    all_div_prop = []
    all_div_hybrid = []

    for run_idx in range(num_runs):
        rng = np.random.default_rng(seed + run_idx)

        # 使用新的分布生成方式（如果提供了 dist_params）或旧的指数衰减分布
        if dist_params is not None:
            # 使用混合分布生成 scores
            scores = generate_scores_from_mixed_distribution(N, d, dist_params, rng)
            class_ids = scores_to_classes(scores)
        else:
            # 旧的指数衰减分布（向后兼容）
            K = 2 ** d
            raw = np.exp(-0.7 * np.arange(K, dtype=np.float64))
            p_true = raw / raw.sum()
            all_classes = [tuple(int(b) for b in np.binary_repr(i, width=d)) for i in range(K)]
            sampled_class_indices = rng.choice(K, size=N, p=p_true)
            class_ids = [all_classes[i] for i in sampled_class_indices]

        # 全量分布
        p_full, class_to_idx = empirical_dist(class_ids)
        num_classes = len(class_to_idx)

        # 安全保护：保证 1 < Q < N
        Q_eff = max(2, min(Q, N - 1))

        # Diversity
        idx_d = diversity_sampling(class_ids, Q_eff, rng)
        cls_d = [class_ids[i] for i in idx_d]
        p_d, _ = empirical_dist(cls_d, class_to_idx=class_to_idx, num_classes=num_classes)
        sim_d = negative_kl_similarity(p_d, p_full)
        div_d = diversity_entropy(cls_d, class_to_idx=class_to_idx, num_classes=num_classes)

        # Proportional
        idx_p = proportional_sampling(class_ids, Q_eff, rng)
        cls_p = [class_ids[i] for i in idx_p]
        p_p, _ = empirical_dist(cls_p, class_to_idx=class_to_idx, num_classes=num_classes)
        sim_p = negative_kl_similarity(p_p, p_full)
        div_p = diversity_entropy(cls_p, class_to_idx=class_to_idx, num_classes=num_classes)

        # Hybrid
        idx_h = hybrid_sampling(class_ids, Q_eff, rng)
        cls_h = [class_ids[i] for i in idx_h]
        p_h, _ = empirical_dist(cls_h, class_to_idx=class_to_idx, num_classes=num_classes)
        sim_h = negative_kl_similarity(p_h, p_full)
        div_h = diversity_entropy(cls_h, class_to_idx=class_to_idx, num_classes=num_classes)

        all_sim_div.append(sim_d)
        all_div_div.append(div_d)
        all_sim_prop.append(sim_p)
        all_div_prop.append(div_p)
        all_sim_hybrid.append(sim_h)
        all_div_hybrid.append(div_h)

    return (
        float(np.mean(all_sim_div)),
        float(np.mean(all_div_div)),
        float(np.mean(all_sim_prop)),
        float(np.mean(all_div_prop)),
        float(np.mean(all_sim_hybrid)),
        float(np.mean(all_div_hybrid)),
    )


def run_3d_experiment_d_q(
    seed=42,
    N=1000,
    Q_list=tuple(range(10, 210, 10)),   # 10, 20, ..., 200
    d_list=tuple(range(3, 21, 3)),      # 3, 6, ..., 18
    num_runs=5,
    plot=True,
    dist_params=None,
):
    """
    3D 实验（你现在想要的版本）：
    - 固定 N（样本总数）
    - x 轴：采样比例 Q/N （在绘图时转换）
    - y 轴：d（维度）
    - z 轴：相似度(-KL) 或 多样性(熵)
    
    Args:
        dist_params: 分布参数字典，如果为 None 则使用默认的长尾分布
    """
    Q_list = list(Q_list)
    d_list = list(d_list)

    sim_div_mat = np.zeros((len(d_list), len(Q_list)), dtype=np.float64)
    div_div_mat = np.zeros_like(sim_div_mat)
    sim_prop_mat = np.zeros_like(sim_div_mat)
    div_prop_mat = np.zeros_like(sim_div_mat)
    sim_hybrid_mat = np.zeros_like(sim_div_mat)
    div_hybrid_mat = np.zeros_like(sim_div_mat)

    for i, d in enumerate(d_list):
        for j, Q in enumerate(Q_list):
            (
                s_div,
                dv_div,
                s_prop,
                dv_prop,
                s_hybrid,
                dv_hybrid,
            ) = evaluate_setting_d_q(seed, N, d, Q, num_runs=num_runs, dist_params=dist_params)
            sim_div_mat[i, j] = s_div
            div_div_mat[i, j] = dv_div
            sim_prop_mat[i, j] = s_prop
            div_prop_mat[i, j] = dv_prop
            print(f"d={d}, Q={Q}, N={N} -> "
                  f"diversity(sim={s_div:.4f}, H={dv_div:.4f}), "
                  f"prop(sim={s_prop:.4f}, H={dv_prop:.4f}), "
                  f"hybrid(sim={s_hybrid:.4f}, H={dv_hybrid:.4f})")
            sim_hybrid_mat[i, j] = s_hybrid
            div_hybrid_mat[i, j] = dv_hybrid

    if plot:
        plot_3d_results_d_q(d_list, Q_list, N,
                            sim_div_mat, sim_prop_mat, sim_hybrid_mat,
                            div_div_mat, div_prop_mat, div_hybrid_mat)

    return {
        "d_list": d_list,
        "Q_list": Q_list,
        "N": N,
        "sim_div": sim_div_mat,
        "sim_prop": sim_prop_mat,
        "sim_hybrid": sim_hybrid_mat,
        "div_div": div_div_mat,
        "div_prop": div_prop_mat,
        "div_hybrid": div_hybrid_mat,
    }

def run_once(seed=42, N=100, d=10, plot=True, verbose=True, dist_params=None):
    rng = np.random.default_rng(seed)

    # 使用新的分布生成方式（如果提供了 dist_params）或旧的指数衰减分布
    if dist_params is not None:
        # 使用混合分布生成 scores
        scores = generate_scores_from_mixed_distribution(N, d, dist_params, rng)
        class_ids = scores_to_classes(scores)
    else:
        # 旧的指数衰减分布（向后兼容）
        # 1) 生成具有明显"长尾分布"的类别，而不是接近均匀的分布
        #    这样：
        #    - Proportional 方法：更好地匹配该长尾分布，-KL 更高
        #    - Diversity 方法：更趋近于均匀覆盖类别，熵更高，但会偏离原分布
        K = 2 ** d  # 理论类别数
        # 构造一个显著非均匀的"真实"类别分布（例如指数衰减）
        raw = np.exp(-0.7 * np.arange(K, dtype=np.float64))
        p_true = raw / raw.sum()

        # 用这 K 个类别构造具体的 {0,1}^d 向量（只是为了保持直观）
        all_classes = [tuple(int(b) for b in np.binary_repr(i, width=d)) for i in range(K)]

        # 按 p_true 抽样得到 N 个样本的类别
        sampled_class_indices = rng.choice(K, size=N, p=p_true)
        class_ids = [all_classes[i] for i in sampled_class_indices]

    # 2) Full (original) distribution over equivalence classes
    p_full, class_to_idx = empirical_dist(class_ids)
    num_classes = len(class_to_idx)

    # 3) Evaluate for Q = 1,10,20,...,90
    Q_list = list(range(10, 100, 10))

    # 存储结果数据
    similarity_div = []
    similarity_prop = []
    similarity_hybrid = []
    div_div = []
    div_prop = []
    div_hybrid = []

    if verbose:
        print(f"总数据集唯一类别数: {num_classes}")
        print("Q\tmethod\t相似度(-KL)\tdiversity(entropy)")
    for Q in Q_list:
        # Diversity-oriented baseline
        idx_d = diversity_sampling(class_ids, Q, rng)
        cls_d = [class_ids[i] for i in idx_d]
        p_d, _ = empirical_dist(cls_d, class_to_idx=class_to_idx, num_classes=num_classes)
        sim_d = negative_kl_similarity(p_d, p_full)
        div_d = diversity_entropy(cls_d, class_to_idx=class_to_idx, num_classes=num_classes)
        similarity_div.append(sim_d)
        div_div.append(div_d)
        if verbose:
            print(f"{Q}\tdiversity\t{sim_d:.6f}\t\t{div_d:.6f}")

        # Proportional baseline (true proportional)
        idx_p = proportional_sampling(class_ids, Q, rng)
        cls_p = [class_ids[i] for i in idx_p]
        p_p, _ = empirical_dist(cls_p, class_to_idx=class_to_idx, num_classes=num_classes)
        sim_p = negative_kl_similarity(p_p, p_full)
        div_p = diversity_entropy(cls_p, class_to_idx=class_to_idx, num_classes=num_classes)
        similarity_prop.append(sim_p)
        div_prop.append(div_p)
        if verbose:
            print(f"{Q}\tprop\t{sim_p:.6f}\t\t{div_p:.6f}")

        # Hybrid method
        idx_h = hybrid_sampling(class_ids, Q, rng)
        cls_h = [class_ids[i] for i in idx_h]
        p_h, _ = empirical_dist(cls_h, class_to_idx=class_to_idx, num_classes=num_classes)
        sim_h = negative_kl_similarity(p_h, p_full)
        div_h = diversity_entropy(cls_h, class_to_idx=class_to_idx, num_classes=num_classes)
        similarity_hybrid.append(sim_h)
        div_hybrid.append(div_h)
        if verbose:
            print(f"{Q}\thybrid\t{sim_h:.6f}\t\t{div_h:.6f}")

        # Our method (left empty)
        # idx_o = our_method_selection(class_ids, Q, rng)
        # cls_o = [class_ids[i] for i in idx_o]
        # p_o, _ = empirical_dist(cls_o, class_to_idx=class_to_idx, num_classes=len(class_to_idx))
        # print(f"{Q}\tours\t{kl_divergence(p_o, p_full):.6f}\t\t{diversity_unique_classes(cls_o)}")

    # 绘图
    if plot:
        plot_results(Q_list, similarity_div, similarity_prop, similarity_hybrid,
                     div_div, div_prop, div_hybrid)
    
    return {
        'Q_list': Q_list,
        'similarity_div': similarity_div,
        'similarity_prop': similarity_prop,
        'similarity_hybrid': similarity_hybrid,
        'div_div': div_div,
        'div_prop': div_prop,
        'div_hybrid': div_hybrid,
    }

def plot_3d_results_d_q(d_list, Q_list, N,
                        sim_div, sim_prop, sim_hybrid,
                        div_div, div_prop, div_hybrid):
    """
    3D 曲面图（固定 N，x=d, y=Q）：
    - 图 1：z = 相似度(-KL)
    - 图 2：z = 多样性(熵)
    """
    # 保持矩阵形状与 sim_* 相同：行对应 d，列对应 Q
    d_arr = np.array(d_list, dtype=np.float64)
    Q_arr = np.array(Q_list, dtype=np.float64)
    D, Q = np.meshgrid(d_arr, Q_arr, indexing="ij")  # D,Q 形状与 sim_* 相同 (len(d), len(Q))
    # 将 Q 换算成采样比例 r = Q / N 作为 x 轴
    R = Q.astype(np.float64) / float(N)

    # 增加画布宽度，为z轴标签预留更多空间
    fig = plt.figure(figsize=(8, 4))

    # 三种方法各自的 colormap，截断避免浅色为纯白，并整体偏亮一些
    cmap_div = get_truncated_cmap('Blues', minval=0.3, maxval=0.9)
    cmap_prop = get_truncated_cmap('YlOrBr', minval=0.25, maxval=0.95)   # 黄色系
    cmap_hybrid = get_truncated_cmap('Greens', minval=0.3, maxval=0.9)

    # Distribution alignment 3D
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # 注意：这里用 Q 作为 x 轴，D 作为 y 轴
    surf1 = ax1.plot_surface(
        R, D, sim_div,
        cmap=cmap_div,
        rstride=1, cstride=1,
        linewidth=0,
        antialiased=True,
        alpha=0.4,
    )
    surf2 = ax1.plot_surface(
        R, D, sim_prop,
        cmap=cmap_prop,
        rstride=1, cstride=1,
        linewidth=0,
        antialiased=True,
        alpha=0.4,
    )
    # 第三种：Hybrid，用绿色系
    surf3 = ax1.plot_surface(
        R, D, sim_hybrid,
        cmap=cmap_hybrid,
        rstride=1, cstride=1,
        linewidth=0,
        antialiased=True,
        alpha=0.4,
    )
    ax1.set_xlabel('Selection ratio', fontsize=FONT_SIZE)
    ax1.set_ylabel('Dimension', fontsize=FONT_SIZE)
    ax1.zaxis.set_rotate_label(True)
    ax1.set_zlabel('Distribution alignment (negative KL)', rotation=90)
    # ax1.set_title('3D surface of similarity with fixed N')
    r_min = min(Q_list) / float(N)
    r_max = max(Q_list) / float(N)
    ax1.set_xlim(r_min, r_max)
    ax1.set_ylim(min(d_list), max(d_list))
    # 仅在视觉上略微拉长 Dimension 轴（y 轴），不改变数值范围
    try:
        ax1.set_box_aspect((1.8, 2, 1.4))
    except Exception:
        pass
    # 采样比例刻度：用 Q/N 对应的位置
    ratio_ticks = [q / float(N) for q in Q_list]
    ax1.set_xticks(ratio_ticks)
    ax1.set_yticks(d_list)

    # 为了更清楚地看出三种方法在关键位置的相对关系：
    # 只在两条“边界线”上叠加散点：
    # 1) d 最大那一排（D 最大，对应 d_list[-1]）
    # 2) Q/N 最小那一排（R 最小，对应 Q_list[0]）
    offset = (r_max - r_min) * 0.01  # 很小的水平偏移
    # d 最大那一排：行索引 = len(d_list) - 1
    i_max_d = len(d_list) - 1
    R_row = R[i_max_d, :]
    D_row = D[i_max_d, :]
    ax1.scatter(R_row - offset, D_row, sim_div[i_max_d, :],
                color=cmap_div(0.95), s=14, depthshade=False)
    ax1.scatter(R_row, D_row, sim_prop[i_max_d, :],
                color=cmap_prop(0.95), s=14, depthshade=False)
    ax1.scatter(R_row + offset, D_row, sim_hybrid[i_max_d, :],
                color=cmap_hybrid(0.95), s=14, depthshade=False)
    # Q/N 最小那一排：列索引 = 0
    j_min_q = 0
    R_col = R[:, j_min_q]
    D_col = D[:, j_min_q]
    ax1.scatter(R_col - offset, D_col, sim_div[:, j_min_q],
                color=cmap_div(0.7), s=14, depthshade=False)
    ax1.scatter(R_col, D_col, sim_prop[:, j_min_q],
                color=cmap_prop(0.7), s=14, depthshade=False)
    ax1.scatter(R_col + offset, D_col, sim_hybrid[:, j_min_q],
                color=cmap_hybrid(0.7), s=14, depthshade=False)

    # 多样性 3D
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf4 = ax2.plot_surface(
        R, D, div_div,
        cmap=cmap_div,
        rstride=1, cstride=1,
        linewidth=0,
        antialiased=True,
        alpha=0.4,
    )
    surf5 = ax2.plot_surface(
        R, D, div_prop,
        cmap=cmap_prop,
        rstride=1, cstride=1,
        linewidth=0,
        antialiased=True,
        alpha=0.4,
    )
    # Hybrid 多样性，绿色系
    surf6 = ax2.plot_surface(
        R, D, div_hybrid,
        cmap=cmap_hybrid,
        rstride=1, cstride=1,
        linewidth=0,
        antialiased=True,
        alpha=0.4,
    )
    ax2.set_xlabel('Selection ratio', fontsize=FONT_SIZE)
    ax2.set_ylabel('Dimension', fontsize=FONT_SIZE)
    ax2.zaxis.set_rotate_label(True)
    ax2.set_zlabel('Diversity (entropy)', rotation=90)
    # ax2.set_title('3D surface of diversity with fixed N')
    ax2.set_xlim(r_min, r_max)
    ax2.set_ylim(min(d_list), max(d_list))
    try:
        ax2.set_box_aspect((1.8, 2, 1.4))
    except Exception:
        pass
    ax2.set_xticks(ratio_ticks)
    ax2.set_yticks(d_list)

    # 在多样性图上同样仅在这两条边界线上叠加散点
    ax2.scatter(R_row - offset, D_row, div_div[i_max_d, :],
                color=cmap_div(0.95), s=14, depthshade=False)
    ax2.scatter(R_row, D_row, div_prop[i_max_d, :],
                color=cmap_prop(0.95), s=14, depthshade=False)
    ax2.scatter(R_row + offset, D_row, div_hybrid[i_max_d, :],
                color=cmap_hybrid(0.95), s=14, depthshade=False)

    ax2.scatter(R_col - offset, D_col, div_div[:, j_min_q],
                color=cmap_div(0.7), s=14, depthshade=False)
    ax2.scatter(R_col, D_col, div_prop[:, j_min_q],
                color=cmap_prop(0.7), s=14, depthshade=False)
    ax2.scatter(R_col + offset, D_col, div_hybrid[:, j_min_q],
                color=cmap_hybrid(0.7), s=14, depthshade=False)

    # 用各自颜色的线把关键散点连起来，增强趋势感
    # d 最大那一排（三种方法各一条线）
    ax1.plot(R_row - offset, D_row, sim_div[i_max_d, :],
             color=cmap_div(0.95), linewidth=1.5)
    ax1.plot(R_row, D_row, sim_prop[i_max_d, :],
             color=cmap_prop(0.95), linewidth=1.5)
    ax1.plot(R_row + offset, D_row, sim_hybrid[i_max_d, :],
             color=cmap_hybrid(0.95), linewidth=1.5)
    ax2.plot(R_row - offset, D_row, div_div[i_max_d, :],
             color=cmap_div(0.95), linewidth=1.5)
    ax2.plot(R_row, D_row, div_prop[i_max_d, :],
             color=cmap_prop(0.95), linewidth=1.5)
    ax2.plot(R_row + offset, D_row, div_hybrid[i_max_d, :],
             color=cmap_hybrid(0.95), linewidth=1.5)

    # Q/N 最小那一列（三种方法各一条线）
    ax1.plot(R_col - offset, D_col, sim_div[:, j_min_q],
             color=cmap_div(0.7), linewidth=1.5)
    ax1.plot(R_col, D_col, sim_prop[:, j_min_q],
             color=cmap_prop(0.7), linewidth=1.5)
    ax1.plot(R_col + offset, D_col, sim_hybrid[:, j_min_q],
             color=cmap_hybrid(0.7), linewidth=1.5)
    ax2.plot(R_col - offset, D_col, div_div[:, j_min_q],
             color=cmap_div(0.7), linewidth=1.5)
    ax2.plot(R_col, D_col, div_prop[:, j_min_q],
             color=cmap_prop(0.7), linewidth=1.5)
    ax2.plot(R_col + offset, D_col, div_hybrid[:, j_min_q],
             color=cmap_hybrid(0.7), linewidth=1.5)

    # 整个图只保留一个简单图例，用颜色区分三种方法
    # 将图例放在画幅内部底部中央，避免被裁剪掉
    legend_handles = [
        Patch(facecolor=cmap_prop(0.8), label='Instance-balanced'),
        Patch(facecolor=cmap_div(0.8), label='Class-balanced'),
        Patch(facecolor=cmap_hybrid(0.8), label='Equivalence-class-based'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3)

    # 视角调整：更仰视一点，并按你期望的相反方向绕 z 轴旋转
    # 更仰视：减小仰角；旋转方向：相对默认视角向另一侧转动
    ax1.view_init(elev=12, azim=-65)
    ax2.view_init(elev=12, azim=-65)

    # 再次显式设置 z 轴标签，减小labelpad避免标签被截断
    ax1.zaxis.set_rotate_label(True)
    ax1.set_zlabel('Distribution alignment', rotation=90, labelpad=5, fontsize=FONT_SIZE)
    ax2.zaxis.set_rotate_label(True)
    ax2.set_zlabel('Diversity', rotation=90, labelpad=5, fontsize=FONT_SIZE)

    # 让 3D 灰色网格线适度变浅（但仍清晰可见），并把刻度数字调小一点
    for ax in (ax1, ax2):
        try:
            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                axis._axinfo['grid']['color'] = (0.7, 0.7, 0.7, 0.5)
                axis._axinfo['grid']['linewidth'] = 0.6
        except Exception:
            pass
        # 缩小 x/y/z 轴刻度数字字号，保持轴标签字号不变
        ax.tick_params(axis='x', which='major', labelsize=8)
        ax.tick_params(axis='y', which='major', labelsize=8)
        ax.tick_params(axis='z', which='major', labelsize=8)
        ax.dist = 8  # 默认大概 10；越小越“放大”（Matplotlib 新版本可能提示 deprecated，但仍常用）

    # 使用subplots_adjust为z轴标签预留空间，特别是右侧
    plt.subplots_adjust(left=0.02, right=0.93, bottom=0.01, top=1, wspace=0.15)
    plt.savefig('sampling_comparison_3d_d_q.png', dpi=300)
    print("\n3D 图表（d-Q）已保存为: sampling_comparison_3d_d_q.png")
    # plt.show()

if __name__ == "__main__":
    # 从 3 个 jsonl 文件加载分布参数
    jsonl_paths = [
        # 'local_vector_store/toole.jsonl',
        'local_vector_store/apibank-8b.jsonl',
        'local_vector_store/bfcl-qwen3-8b.jsonl',
    ]
    
    print("="*60)
    print("正在从 jsonl 文件加载分布参数（按维度学习概率）...")
    print("="*60)
    dist_params = load_distribution_params_from_jsonls(jsonl_paths)
    print(f"\n加载完成")
    if dist_params.get('dim_probs') is not None:
        print(f"支持的维度: {sorted(dist_params['dim_probs'].keys())}")
    print("="*60)
    print()
    
    # 2D: Q vs 指标（保留原函数，可按需调用）
    # run_experiment(seed=42, N=100, d=3, num_runs=10, plot=True, dist_params=dist_params)

    # 3D（当前需求）：固定 N，x=d, y=Q, z=相似度或多样性
    n_samples = 10000
    run_3d_experiment_d_q(
        seed=5671,
        N=n_samples,                         # 固定总样本数
        Q_list=tuple(range(int(n_samples*0.1), n_samples, int(n_samples*0.1))),  # Q = 10,20,...,100
        d_list=tuple(range(5, 16, 1)),     # d = 2,3,...,15
        num_runs=3,
        plot=True,
        dist_params=dist_params,  # 使用从 jsonl 文件加载的分布参数
    )
