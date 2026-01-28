# Scores 分布拟合建议

## 数据特征总结

根据分析，多个 scores 数组之间的成功率分布具有以下特征：

### bfcl-qwen3-8b.jsonl
- **总样本数**: 1,703
- **完全成功 (1.0)**: 65.83%
- **完全失败 (0.0)**: 0.18%
- **部分成功 (0 < p < 1)**: 34.00%
- **部分成功的均值**: 0.6165 ± 0.1714
- **分布特征**: 负偏（左偏），集中在高值

### deepseek-v3.jsonl
- **总样本数**: 553
- **完全成功 (1.0)**: 75.41%
- **完全失败 (0.0)**: 0%
- **部分成功 (0 < p < 1)**: 24.59%
- **分布特征**: 负偏（左偏），集中在高值

## 拟合方案建议

### 方案 1: 混合分布模型（推荐）⭐⭐⭐

**适用场景**: 数据有明显的"完全成功"峰值

**模型结构**:
```python
# 混合分布：完全成功 + Beta分布
def mixed_distribution(p_success, alpha, beta):
    """
    p_success: 完全成功的概率
    alpha, beta: Beta分布的参数
    """
    # 以概率 p_success 返回 1.0
    # 以概率 (1-p_success) 从 Beta(alpha, beta) 采样
    if np.random.random() < p_success:
        return 1.0
    else:
        return np.random.beta(alpha, beta)
```

**参数估计**:
- `p_success = 完全成功的比例` (约 0.66-0.75)
- `alpha, beta` 通过部分成功的数据拟合 Beta 分布得到

**优点**:
- 准确反映数据的双峰特征
- 可以分别建模完全成功和部分成功的分布
- 拟合效果好

**实现示例**:
```python
import numpy as np
from scipy import stats

# 估计参数
p_success = np.sum(success_rates == 1.0) / len(success_rates)
partial_rates = success_rates[(success_rates > 0) & (success_rates < 1)]

# 拟合 Beta 分布
alpha, beta, loc, scale = stats.beta.fit(partial_rates, floc=0, fscale=1)

# 生成样本
def sample_mixed(p_success, alpha, beta, n=1):
    samples = []
    for _ in range(n):
        if np.random.random() < p_success:
            samples.append(1.0)
        else:
            samples.append(np.random.beta(alpha, beta))
    return np.array(samples)
```

---

### 方案 2: Beta 分布（简化版）⭐⭐

**适用场景**: 如果不需要区分完全成功和部分成功

**模型结构**:
```python
# 直接用 Beta 分布拟合所有数据（包括 1.0）
success_rate ~ Beta(alpha, beta)
```

**参数估计**:
```python
from scipy import stats
alpha, beta, loc, scale = stats.beta.fit(success_rates, floc=0, fscale=1)
```

**优点**:
- 简单易用
- Beta 分布天然适合 0-1 区间的概率分布
- 可以处理偏态分布

**缺点**:
- 可能无法很好地捕捉"完全成功"的峰值
- 对于极端值（0.0 和 1.0）的拟合可能不够准确

---

### 方案 3: 截断正态分布 ⭐

**适用场景**: 如果部分成功的分布接近正态

**模型结构**:
```python
# 截断到 [0, 1] 区间的正态分布
success_rate ~ TruncatedNormal(mu, sigma, a=0, b=1)
```

**参数估计**:
```python
from scipy import stats

# 只对部分成功的数据拟合
partial_rates = success_rates[(success_rates > 0) & (success_rates < 1)]
mu, sigma = stats.norm.fit(partial_rates)

# 然后截断到 [0, 1]
```

**优点**:
- 如果部分成功的数据接近正态，拟合效果好
- 参数解释直观

**缺点**:
- 需要处理截断
- 无法很好地处理完全成功的情况

---

### 方案 4: 经验分布（非参数方法）⭐⭐

**适用场景**: 不需要参数化模型，直接使用数据

**模型结构**:
```python
# 直接从数据中采样
success_rate ~ Empirical(success_rates)
```

**实现**:
```python
def sample_empirical(success_rates, n=1):
    return np.random.choice(success_rates, size=n)
```

**优点**:
- 完全忠实于数据
- 不需要假设分布形式
- 实现简单

**缺点**:
- 无法外推
- 需要保存所有数据

---

## 推荐方案

### 对于 test_sample_method.py 的使用场景

如果需要在 `test_sample_method.py` 中生成类似的成功率分布，**推荐使用方案 1（混合分布）**：

```python
def generate_success_rates_mixed(n_samples, p_success=0.66, alpha=2.0, beta=1.2):
    """
    生成成功率分布（混合模型）
    
    参数:
        n_samples: 生成的样本数
        p_success: 完全成功的概率（从数据估计得到）
        alpha, beta: Beta分布参数（从部分成功的数据估计得到）
    """
    rates = []
    for _ in range(n_samples):
        if np.random.random() < p_success:
            rates.append(1.0)
        else:
            rates.append(np.random.beta(alpha, beta))
    return np.array(rates)
```

### 参数估计代码

```python
import json
import numpy as np
from scipy import stats

def estimate_mixed_distribution_params(jsonl_path):
    """从 jsonl 文件估计混合分布参数"""
    success_rates = []
    
    # 提取成功率
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            metadata = data.get('metadata', {})
            tool_call_results = metadata.get('tool_call_results', [])
            for result in tool_call_results:
                if 'scores' in result:
                    scores = np.array(result['scores'])
                    if len(np.unique(scores)) <= 2 and all(v in [0, 1] for v in np.unique(scores)):
                        success_rate = float(np.sum(scores) / len(scores))
                        success_rates.append(success_rate)
    
    success_rates = np.array(success_rates)
    
    # 估计完全成功的概率
    p_success = np.sum(success_rates == 1.0) / len(success_rates)
    
    # 估计 Beta 分布参数（只使用部分成功的数据）
    partial_rates = success_rates[(success_rates > 0) & (success_rates < 1)]
    if len(partial_rates) > 0:
        alpha, beta, _, _ = stats.beta.fit(partial_rates, floc=0, fscale=1)
    else:
        # 如果没有部分成功的数据，使用默认值
        alpha, beta = 2.0, 1.2
    
    return {
        'p_success': float(p_success),
        'alpha': float(alpha),
        'beta': float(beta),
        'n_samples': len(success_rates),
        'n_perfect': int(np.sum(success_rates == 1.0)),
        'n_partial': len(partial_rates)
    }

# 使用示例
params = estimate_mixed_distribution_params('local_vector_store/bfcl-qwen3-8b.jsonl')
print(f"估计的参数: {params}")
```

---

## 与 test_sample_method.py 的对比

**test_sample_method.py 使用**: 指数衰减分布（长尾分布）
- `raw = np.exp(-0.7 * np.arange(K))`
- 适用于长尾分布场景

**实际数据特征**: 负偏分布，集中在高值
- 65-75% 完全成功
- 不是长尾分布

**结论**: 
- ❌ **不适合**直接用 test_sample_method.py 中的指数衰减分布
- ✅ **适合**使用混合分布模型（完全成功 + Beta分布）

---

## 实施建议

1. **短期方案**: 使用经验分布（方案 4），直接从数据中采样
2. **中期方案**: 使用 Beta 分布（方案 2），简单且有效
3. **长期方案**: 使用混合分布（方案 1），最准确地反映数据特征
