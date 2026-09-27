import os
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
from prompt.ExampleSelectorTemplate import BasicExampleSelector
from utils.linking_utils.application import mask_question_with_schema_linking
from utils.signal_lib import SignalProcessor
from utils.mcts_optimizer import MCTS
from utils.sinr_reward import calculate_sinr

class SIGNAL_DETECTION(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        """
        基于信号检测理论的样本选择器
        集成预白化(Pre-whitening)、恒虚警检测(CFAR)与 MCTS 组合优化
        """
        # 1. 调用基类构造函数
        super().__init__(data, *args, **kwargs)
        
        # 2. 配置参数
        self.tokenizer = kwargs.get('tokenizer', 'gpt-3.5-turbo')
        self.p_fa = 0.01  # CFAR 虚警率
        
        # MCTS 参数
        self.mcts_simulations = int(os.environ.get("MCTS_SIMULATIONS", "200"))  # MCTS 模拟次数
        
        # LLM 缓存
        self.llm_cache_path = os.path.join("dataset", "process", "llm_decompose_cache.json")
        self.llm_cache = {}
        if os.path.exists(self.llm_cache_path):
            try:
                import json
                with open(self.llm_cache_path, 'r', encoding='utf-8') as f:
                    self.llm_cache = json.load(f)
            except Exception as e:
                print(f"[SignalSelector] Warning: Failed to load LLM cache: {e}")
        self.mcts_c_uct = 1.41       # UCT 探索系数
        self.sinr_lambda1 = 0.5      # 噪声权重
        self.sinr_lambda2 = 0.3      # 干扰权重
        
        # 3. 初始化 Embedding 模型
        self.default_model_name = "sentence-transformers/all-mpnet-base-v2"
        self.SELECT_MODEL = os.environ.get("EMBEDDING_MODEL_PATH", self.default_model_name)
        self.mask_token = "<mask>"
        self.value_token = "<unk>"

        try:
            print(f"[SignalSelector] Loading model from {self.SELECT_MODEL}...")
            self.bert_model = SentenceTransformer(self.SELECT_MODEL, device="cpu")
        except Exception as e:
            print(f"[SignalSelector] Warning: Failed to load from {self.SELECT_MODEL} ({e}), falling back to '{self.default_model_name}'...")
            self.bert_model = SentenceTransformer(self.default_model_name, device="cpu")

        # 4. 准备训练集数据 (Unmasked Embedding & BM25)
        import hashlib
        import numpy as np
        
        # 初始化 BM25 (Hybrid Pooling)
        from rank_bm25 import BM25Okapi
        self.tokenized_corpus = [q.get("question", "").lower().split(" ") for q in self.train_json]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # 为了区分不同的数据集 (Spider vs BIRD)，简单哈希一下训练集问题
        sample_text = "".join([q.get("question", "") for q in self.train_json[:100]])
        dataset_hash = hashlib.md5(sample_text.encode('utf-8')).hexdigest()[:8]
        cache_path = os.path.join("dataset", "process", f"train_embeddings_cache_UNMASKED_{len(self.train_json)}_{dataset_hash}.npy")
        
        if os.path.exists(cache_path):
            print(f"[SignalSelector] 命中无掩码缓存，直接加载 Embeddings: {cache_path}")
            self.train_embeddings = np.load(cache_path)
        else:
            print("[SignalSelector] 正在对训练集进行无掩码 Embedding 计算 (这将耗时几分钟)...")
            train_questions = [q.get("question", "") for q in self.train_json]
            self.train_embeddings = self.bert_model.encode(
                train_questions, 
                batch_size=64, 
                show_progress_bar=True, 
                convert_to_numpy=True
            )
            
            # 保存到缓存
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, self.train_embeddings)
            print(f"[SignalSelector] Embeddings 已缓存至: {cache_path}")


        # 准备训练集的 DB IDs 用于 H0 阈值的跨域难例标定
        train_db_ids = [item.get("db_id", "unknown") for item in self.train_json]

        # 5. 初始化核心信号处理器 (白化 & 阈值标定)
        self.processor = SignalProcessor(
            self.train_embeddings, 
            self.p_fa,
            train_db_ids=train_db_ids
        )

    def _decompose_query_with_llm(self, target):
        query = target.get("question", "")
        evidence = target.get("evidence", "").strip()
        q_clean = query.strip()
        if evidence:
            q_clean = f"{q_clean} | Evidence: {evidence}"
            
        if q_clean in self.llm_cache:
            # print(f"   [SignalSelector - LLM] Hit Cache for query: {query}")
            return self.llm_cache[q_clean]
        
        import sys
        print(f"   [SignalSelector - LLM] CACHE MISS FOR: '{q_clean}'", file=sys.stderr)

        # print(f"   [SignalSelector - LLM] 复杂查询触发拆解: {query}")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("   [SignalSelector - LLM] Warning: 未配置 OPENAI_API_KEY 环境变量，跳过拆解")
            return []
        
        import requests
        import json
        
        api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        url = f"{api_base.rstrip('/')}/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        prompt = f"""You are an advanced database query logic analyst. Please evaluate the complexity of the following natural language question and intelligently decompose it into indivisible, computationally independent atomic query fragments.
Strictly adhere to the following principles:
1. [Moderate Decomposition]: Usually split into 1 to 3 atomic questions. For extremely simple, straightforward single-table queries, return only 1 element (the original question itself); for difficult problems containing multi-table JOINs, nested subqueries, and complex aggregation filtering, reasonably split into 2 to 3 core relational steps (e.g., extract main framework information, apply constraint joins, execute aggregation/sorting operations, etc.).
2. [Absolute Information Fidelity]: You must 100% retain all independent entity values, proper noun constraints, and various attributive restrictions (such as average, maximum, less than a certain time, etc.) from the original question. Absolutely prohibit information summarization or loss due to paraphrasing. Keep the language in English.
3. [No Logic Omission]: The split sub-questions should each still constitute a valuable small logic fragment (e.g., "Find all basic user information" and "Calculate the average consumption of these users over the past three months").

Mandatory Output Format:
Output ONLY a valid JSON string array. Strictly do not include any extra explanations, thoughts, prefixes, or Markdown code block formatting tags.
Example: ["Sub-query part 1", "Sub-query part 2"]

Original Question: {query}"""
        
        if evidence:
            prompt += f"\nExternal Evidence: {evidence}\n(Ensure sub-queries incorporate necessary mathematical or logical conditions from this evidence)"

        
        model_name = os.environ.get("OPENAI_MODEL", "gpt-4")
        
        try:
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 150
            }
            res = requests.post(url, headers=headers, json=payload, timeout=60, verify=False)
            res.raise_for_status()
            content = res.json()["choices"][0]["message"]["content"].strip()
            
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            import ast
            try:
                sub_queries = ast.literal_eval(content)
            except Exception:
                try:
                    sub_queries = json.loads(content)
                except Exception:
                    sub_queries = [sq.strip().strip("'").strip('"') for sq in content.replace('[', '').replace(']', '').split(',')]
            
            if not isinstance(sub_queries, list):
                sub_queries = [str(sub_queries)]
                
            sub_queries = [str(sq).strip() for sq in sub_queries if str(sq).strip()]
            import sys
            print(f"   [SignalSelector - LLM] 成功拆解为 {len(sub_queries)} 个子查询: {sub_queries}", file=sys.stderr, flush=True)
            
            self.llm_cache[q_clean] = sub_queries
            try:
                with open(self.llm_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self.llm_cache, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"   [SignalSelector - LLM] Warning: Failed to save cache: {e}")
                
            return sub_queries
        except Exception as e:
            print(f"   [SignalSelector - LLM] 调用失败: {e}")
            return []

    def get_examples(self, target, num_example, cross_domain=False):
        """
        重写基类的 get_examples 方法
        Pipeline: 原子解构 → 候选池生成 → MCTS 组合优化
        
        :param target: 目标问题对象 (包含 question, db_id, 可能含 pre_skeleton)
        :param num_example: k-shot 的数量
        :param cross_domain: 是否跨域 (Spider 默认是 True)
        
        消融实验开关 (通过环境变量控制):
          ABLATION_NO_DECOMPOSE=1  跳过原子解构，仅用原问题检索+MCTS
          ABLATION_NO_MCTS=1       保留解构，但用贪婪Top-K替代MCTS
          ABLATION_NO_EINTER=1     保留全流程，但SINR中lambda2=0
        """
        # 消融实验开关
        ablation_no_decompose = os.environ.get("ABLATION_NO_DECOMPOSE", "") == "1"
        ablation_no_mcts = os.environ.get("ABLATION_NO_MCTS", "") == "1"
        ablation_no_einter = os.environ.get("ABLATION_NO_EINTER", "") == "1"
        
        if ablation_no_einter:
            effective_lambda2 = 0.0
        else:
            effective_lambda2 = self.sinr_lambda2
        # 1. 处理目标问题 (Unmasked Embedding & BM25 Tokenization)
        target_q_str = target.get("question", "")
        target_embedding = self.bert_model.encode([target_q_str], convert_to_numpy=True)[0]
        
        # 1.5. 获取 BM25 候选
        tokenized_query = target_q_str.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_pairs = [(float(score), index) for index, score in enumerate(bm25_scores)]
        bm25_pairs.sort(key=lambda x: x[0], reverse=True)

        # 2. 对原问题进行初始宽候选池检索 (Hybrid Pooling)
        actual_k = getattr(self, "NUM_EXAMPLE", int(num_example/100) if num_example >= 100 else num_example)  # 从实例获取真实的 K
        candidate_k = max(actual_k * 4, 30)
        
        # 获取 Cosine/预白化 候选
        top_indices, t_score, _ = self.processor.detect_and_retrieve(
            target_embedding, 
            k=candidate_k
        )
        
        # 提取 BM25 的 Top-K 索引
        bm25_top_indices = [idx for _, idx in bm25_pairs[:candidate_k]]
        
        candidate_pool = []       # 最终大集候选样本列表
        candidate_indices = []    # 对应的在训练集中的原始索引
        
        # 为了支持 w/o MCTS 的轮询提取，分别记录原问题和各个子问题的召回结果
        original_pool_indices = []
        subquery_pools_indices = [] # 列表的列表，每个子问题一个召回列表
        seen_indices = set()

        # 立即将原问题召回的最佳结果并入候选 (Hybrid 混合去重)
        # 采用拉链法交替并入，保证 BM25 和 Cosine 的多样性
        max_len = max(len(top_indices), len(bm25_top_indices))
        for i in range(max_len):
            # 添加 BM25 候选
            if i < len(bm25_top_indices):
                idx = bm25_top_indices[i]
                candidate = self.train_json[idx]
                if not (cross_domain and candidate["db_id"] == target["db_id"]):
                    if idx not in seen_indices:
                        original_pool_indices.append(idx)
                        seen_indices.add(idx)
            # 添加 Cosine 候选
            if i < len(top_indices):
                idx = top_indices[i]
                candidate = self.train_json[idx]
                if not (cross_domain and candidate["db_id"] == target["db_id"]):
                    if idx not in seen_indices:
                        original_pool_indices.append(idx)
                        seen_indices.add(idx)

        # 3. ========= 逢题必拆解的组合生成方案 (Always Decompose) =========
        if ablation_no_decompose:
            print(f"   [ABLATION] 跳过原子解构，仅使用原问题候选...")
            sub_queries = []
        else:
            print(f"   [Second Stage] 直接触发原子拆解并开启组合优化...")
            sub_queries = self._decompose_query_with_llm(target)
        
        if sub_queries:
            for sq in sub_queries:
                sq_target = target.copy()
                sq_target["question"] = sq
                
                # 对各个原子子查询进行无掩码特征检索
                sq_embedding = self.bert_model.encode([sq], convert_to_numpy=True)[0]
                
                # 子查询的特征匹配检索 (为每个原子组件独立获取部分最相关的样本)
                sq_top_indices, _, _ = self.processor.detect_and_retrieve(sq_embedding, k=10)
                
                current_sq_indices = []
                for idx in sq_top_indices:
                    candidate = self.train_json[idx]
                    if cross_domain and candidate["db_id"] == target["db_id"]:
                        continue
                    if idx not in seen_indices:
                        current_sq_indices.append(idx)
                        seen_indices.add(idx)
                if current_sq_indices:
                    subquery_pools_indices.append(current_sq_indices)
        
        # ================= 聚合并生成 MCTS 搜索全集 =================
        # 为了兼容 MCTS，我们先构造一个顺序合理的 candidate_pool
        all_ordered_indices = original_pool_indices.copy()
        for sq_indices in subquery_pools_indices:
            all_ordered_indices.extend(sq_indices)
            
        for index in all_ordered_indices:
            candidate_pool.append(self.train_json[index])
            candidate_indices.append(index)
            # 采用硬性上限控制，免于过度膨胀导致巨大的 MCTS 组合复杂度浪费
            if len(candidate_pool) >= candidate_k * 2:
                break
        
        if not candidate_pool:
            print("   [Warning] 拆解池构建失败或候选极端剔除被清空，无样板可供检索...")
        
        # 4. 判断是否启用 MCTS 组合优化
        target_sql = target.get("pre_skeleton", None) or target.get("query", "")
        has_sql_info = bool(target_sql and target_sql.strip())
        
        if ablation_no_mcts:
            # ========== [ABLATION] 轮询混抽 (Round-Robin): 公平提取局部和全局示例 ==========
            print(f"   [ABLATION] 跳过MCTS，使用轮询混抽 (Round-Robin) 拼接...")
            final_examples = []
            
            # 首先拿原问题的 Top 2 (或更少如果不足)
            num_orig = min(2, len(original_pool_indices))
            for i in range(num_orig):
                final_examples.append(self.train_json[original_pool_indices[i]])
                
            # 从子问题池子里轮询取数据，直到凑满 actual_k
            sq_pointers = [0] * len(subquery_pools_indices)
            
            while len(final_examples) < actual_k:
                added_in_round = False
                for pool_idx, pool_list in enumerate(subquery_pools_indices):
                    if len(final_examples) >= actual_k:
                        break
                    p = sq_pointers[pool_idx]
                    if p < len(pool_list):
                        final_examples.append(self.train_json[pool_list[p]])
                        sq_pointers[pool_idx] += 1
                        added_in_round = True
                        
                # 如果子问题都被抽光了，或者没有子问题，就回退到原问题的剩余候选补齐
                if not added_in_round:
                    break
                    
            # 如果还没满，用原问题的剩余部分补齐
            orig_ptr = num_orig
            while len(final_examples) < actual_k and orig_ptr < len(original_pool_indices):
                final_examples.append(self.train_json[original_pool_indices[orig_ptr]])
                orig_ptr += 1
                
        elif has_sql_info and len(candidate_pool) > actual_k:
            # ========== MCTS 路径: 使用 SINR 进行组合优化 ==========
            candidate_sqls = [c.get("query", "") for c in candidate_pool]
            candidate_embs = self.train_embeddings[candidate_indices]
            
            best_local_indices = MCTS.search(
                target_sql=target_sql,
                target_embedding=target_embedding,
                candidate_pool=candidate_pool,
                candidate_sqls=candidate_sqls,
                candidate_embeddings=candidate_embs,
                k=actual_k,
                n_simulations=self.mcts_simulations,
                c_uct=self.mcts_c_uct,
                lambda1=self.sinr_lambda1,
                lambda2=effective_lambda2,  # 消融实验可能为 0
            )
            
            from utils.sinr_reward import calculate_sinr
            best_sqls = [candidate_sqls[i] for i in best_local_indices]
            best_embs = candidate_embs[best_local_indices]
            best_sinr = calculate_sinr(target_sql, best_sqls, best_embs, self.sinr_lambda1, effective_lambda2)
            print(f"   [SignalSelector - MCTS] 最终最优组合 SINR: {best_sinr:.4f}")
            self.last_best_sinr = best_sinr
            
            final_examples = [candidate_pool[i] for i in best_local_indices]
        else:
            # ========== 贪心路径: 候选池太小或无 SQL 信息, 直接取原始 Top-K ==========
            final_examples = []
            for i in range(min(actual_k, len(original_pool_indices))):
                final_examples.append(self.train_json[original_pool_indices[i]])
            # 若不足，按已合并的有序候选池补齐
            idx = 0
            while len(final_examples) < actual_k and idx < len(candidate_pool):
                ex = candidate_pool[idx]
                if ex not in final_examples:
                    final_examples.append(ex)
                idx += 1

        return final_examples