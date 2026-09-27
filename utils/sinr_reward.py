"""
sinr_reward.py - 基于 SINR 的解析式奖励函数
==============================================
论文 4.2.2 节核心公式:
    R(S) = E_match(S, x) / (1 + lambda1 * E_noise(S, x) + lambda2 * E_inter(S))

三个能量分量:
    E_match  = |Feat(x) ∩ (∪_{e∈S} Feat(e))| / |Feat(x)|     (信号能量 - 逻辑覆盖率)
    E_noise  = Σ_{e∈S} |Feat(e) - Feat(x)|                     (噪声能量 - 冗余逻辑)
    E_inter  = Σ_{i<j} (Sim_cos(e_i, e_j) + I[Style(e_i)≠Style(e_j)])  (干扰能量)
"""

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. SQL 特征骨架提取 (不依赖 db_schema, 纯正则)
# ============================================================

# SQL 关键词/操作符字典 (按类别分组)
SQL_KEYWORDS = {
    # Structure keywords
    'select', 'from', 'where', 'group by', 'having', 'order by',
    'limit', 'offset', 'union', 'union all', 'intersect', 'except',
    # Join types
    'join', 'inner join', 'left join', 'right join', 'outer join',
    'left outer join', 'right outer join', 'full outer join', 'cross join',
    # Aggregation
    'count', 'sum', 'avg', 'max', 'min',
    # Modifiers
    'distinct', 'as', 'on', 'using', 'case', 'when', 'then', 'else', 'end',
    # Logical
    'and', 'or', 'not', 'in', 'exists', 'between', 'like', 'is null', 'is not null',
    # Comparison and Math operators
    '=', '!=', '<>', '>', '>=', '<', '<=', '+', '-', '*', '/',
    # Functions
    'cast', 'round', 'substr', 'strftime', 'julianday', 'date',
    # Sort
    'asc', 'desc',
    # Subquery indicators
    'subquery',  # virtual token for nested SELECT
}

# 多词关键词 (需要优先匹配)
MULTI_WORD_KEYWORDS = [
    'group by', 'order by', 'union all', 'inner join', 'left join',
    'right join', 'outer join', 'left outer join', 'right outer join',
    'full outer join', 'cross join', 'is not null', 'is null',
]


def extract_sql_features(sql_str: str) -> set:
    """
    从 SQL 字符串中提取结构特征集合 (Keywords + Structure + Operators)
    
    :param sql_str: SQL 查询字符串 (可以是预生成的粗糙 SQL)
    :return: 特征集合, e.g. {'select', 'where', 'join', 'group by', 'count', '>', ...}
    """
    if not sql_str or not isinstance(sql_str, str):
        return set()
    
    sql_lower = sql_str.lower().strip()
    features = set()
    
    # Step 1: 检测多词关键词 (优先匹配)
    for kw in MULTI_WORD_KEYWORDS:
        if kw in sql_lower:
            features.add(kw)
    
    # Step 2: 检测子查询结构
    # 统计嵌套 SELECT 的数量 (去掉最外层的 SELECT)
    select_count = len(re.findall(r'\bselect\b', sql_lower))
    if select_count > 1:
        features.add('subquery')
    
    # Step 3: 提取单词级关键词
    # 先用正则把 SQL 拆成 tokens
    tokens = re.findall(r'[a-z_]+|[!=<>]+', sql_lower)
    
    single_keywords = {
        'select', 'from', 'where', 'having', 'limit', 'offset',
        'union', 'intersect', 'except', 'join',
        'count', 'sum', 'avg', 'max', 'min',
        'distinct', 'case', 'when', 'then', 'else', 'end',
        'and', 'or', 'not', 'in', 'exists', 'between', 'like',
        'asc', 'desc', 'cast', 'round', 'substr', 'strftime', 'julianday', 'date'
    }
    
    for token in tokens:
        if token in single_keywords:
            features.add(token)
    
    # Step 4: 提取比较与数学运算符
    operators = {'=', '!=', '<>', '>', '>=', '<', '<=', '+', '-', '*', '/'}
    for op in operators:
        if op in sql_lower:
            features.add(op)
    
    # Step 5: 提取 CAST 函数特征
    if 'cast' in sql_lower and ' as ' in sql_lower:
        features.add('cast')
        
    return features


# ============================================================
# 2. SQL 风格指纹提取
# ============================================================

def extract_style_fingerprint(sql_str: str) -> dict:
    """
    提取 SQL 的风格指纹, 用于检测风格一致性
    
    检查项目:
    - 不等号风格: != vs <>
    - 别名风格: t1/t2 vs 完整表名
    - 引号风格: 单引号 vs 双引号
    - 缩进/大小写风格: 全大写 vs 全小写 vs 混合
    """
    if not sql_str:
        return {}
    
    fingerprint = {}
    
    # 1. 不等号风格
    if '<>' in sql_str:
        fingerprint['neq_style'] = '<>'
    elif '!=' in sql_str:
        fingerprint['neq_style'] = '!='
    else:
        fingerprint['neq_style'] = 'none'
    
    # 2. 别名风格 (检查 t1, t2 ... 模式)
    alias_pattern = re.findall(r'\b[tT]\d+\b', sql_str)
    if alias_pattern:
        fingerprint['alias_style'] = 'short'  # t1, t2 风格
    else:
        fingerprint['alias_style'] = 'full'   # 完整表名风格
    
    # 3. 引号风格
    double_quotes = sql_str.count('"')
    single_quotes = sql_str.count("'")
    if double_quotes > single_quotes:
        fingerprint['quote_style'] = 'double'
    else:
        fingerprint['quote_style'] = 'single'
    
    # 4. 关键词大小写风格
    upper_keywords = len(re.findall(r'\b(SELECT|FROM|WHERE|JOIN|GROUP BY|ORDER BY)\b', sql_str))
    lower_keywords = len(re.findall(r'\b(select|from|where|join|group by|order by)\b', sql_str))
    if upper_keywords > lower_keywords:
        fingerprint['case_style'] = 'upper'
    else:
        fingerprint['case_style'] = 'lower'
    
    return fingerprint


# ============================================================
# 3. SINR 三大能量分量
# ============================================================

def compute_e_match(target_features: set, example_features_list: list) -> float:
    """
    信号能量 E_match: 示例集 S 对目标问题 x 的逻辑特征覆盖程度
    
    公式: E_match = |Feat(x) ∩ (∪_{e∈S} Feat(e))| / |Feat(x)|
    
    :param target_features: 目标问题的 SQL 特征集合
    :param example_features_list: 每个候选示例的特征集合列表
    :return: 覆盖率 [0, 1]
    """
    if not target_features:
        return 0.0
    
    # 计算所有示例特征的并集
    union_features = set()
    for feat_set in example_features_list:
        union_features |= feat_set
    
    # 交集覆盖率
    intersection = target_features & union_features
    return len(intersection) / len(target_features)


def compute_e_noise(target_features: set, example_features_list: list) -> float:
    """
    噪声能量 E_noise: 示例集中包含但目标不需要的冗余逻辑
    
    公式: E_noise(S, x) = Σ_{e∈S} |Feat(e) - Feat(x)|
    
    :param target_features: 目标问题的 SQL 特征集合
    :param example_features_list: 每个候选示例的特征集合列表
    :return: 冗余特征总数 (归一化后)
    """
    total_noise = 0
    for feat_set in example_features_list:
        # 示例有但目标没有的特征
        noise = feat_set - target_features
        total_noise += len(noise)
    
    # 归一化: 除以总特征量, 防止样本数量影响
    total_features = sum(len(f) for f in example_features_list)
    if total_features == 0:
        return 0.0
    return total_noise / max(total_features, 1)


def compute_e_inter(example_sqls: list, example_embeddings: np.ndarray = None) -> float:
    """
    样本间干扰能量 E_inter: 度量示例集内部的异质性
    
    公式: E_inter(S) = Σ_{i<j} (Sim_cos(e_i, e_j) + I[Style(e_i) ≠ Style(e_j)])
    
    分两部分:
    1. 冗余性干扰: Cosine Similarity > 0.9 → 惩罚
    2. 风格一致性干扰: 风格指纹不一致 → 惩罚
    
    :param example_sqls: 候选示例的 SQL 字符串列表
    :param example_embeddings: 候选示例的嵌入向量矩阵 (n, d)
    :return: 干扰总分 (归一化后)
    """
    n = len(example_sqls)
    if n <= 1:
        return 0.0
    
    total_inter = 0.0
    n_pairs = n * (n - 1) / 2
    
    # Part 1: 冗余性干扰 (基于 Cosine Similarity)
    if example_embeddings is not None and len(example_embeddings) == n:
        try:
            import numpy as np
            emb_array = np.array(example_embeddings)
            if emb_array.ndim == 1:
                emb_array = emb_array.reshape(1, -1)
            elif emb_array.ndim > 2:
                emb_array = emb_array.reshape(emb_array.shape[0], -1)
                
            sim_matrix = cosine_similarity(emb_array)
            for i in range(n):
                for j in range(i + 1, n):
                    sim_ij = sim_matrix[i][j]
                    # 论文: 如果 Cosine Similarity > 0.9, 视为完全重复
                    if sim_ij > 0.9:
                        total_inter += sim_ij  # 惩罚与相似度成正比
        except Exception as e:
            # 如果特征矩阵异常，则跳过冗余惩罚
            pass
    
    # Part 2: 风格一致性干扰
    style_fingerprints = [extract_style_fingerprint(sql) for sql in example_sqls]
    for i in range(n):
        for j in range(i + 1, n):
            # I[Style(e_i) ≠ Style(e_j)]
            style_diff = 0
            fp_i, fp_j = style_fingerprints[i], style_fingerprints[j]
            for key in set(fp_i.keys()) | set(fp_j.keys()):
                if fp_i.get(key) != fp_j.get(key):
                    style_diff += 1
            # 归一化风格差异 (除以风格维度数)
            n_dims = max(len(set(fp_i.keys()) | set(fp_j.keys())), 1)
            total_inter += style_diff / n_dims
    
    # 归一化: 除以配对数
    return total_inter / max(n_pairs, 1)


# ============================================================
# 4. 组合 SINR 奖励函数
# ============================================================

def calculate_sinr(
    target_sql: str,
    candidate_sqls: list,
    candidate_embeddings: np.ndarray = None,
    lambda1: float = 0.5,
    lambda2: float = 0.3
) -> float:
    """
    计算 SINR 奖励函数
    
    公式: R(S) = E_match(S, x) / (1 + λ₁·E_noise(S, x) + λ₂·E_inter(S))
    
    :param target_sql: 目标问题的 SQL (预生成)
    :param candidate_sqls: 候选示例集的 SQL 字符串列表
    :param candidate_embeddings: 候选示例的嵌入向量矩阵 (可选, 用于计算 E_inter)
    :param lambda1: 噪声能量权重
    :param lambda2: 干扰能量权重
    :return: SINR 分数 (越高越好)
    """
    # 提取特征集合
    target_features = extract_sql_features(target_sql)
    example_features_list = [extract_sql_features(sql) for sql in candidate_sqls]
    
    # 计算三大能量分量
    e_match = compute_e_match(target_features, example_features_list)
    e_noise = compute_e_noise(target_features, example_features_list)
    e_inter = compute_e_inter(candidate_sqls, candidate_embeddings)
    
    # 组合 SINR
    sinr = e_match / (1.0 + lambda1 * e_noise + lambda2 * e_inter)
    
    return sinr
