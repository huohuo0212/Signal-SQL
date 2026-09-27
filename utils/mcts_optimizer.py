"""
mcts_optimizer.py - 基于 MCTS 的结构化组合优化器
=================================================
论文 4.2.3 节:
    将 Prompt 构建定义为序列决策过程, 使用蒙特卡洛树搜索 (MCTS) 寻找
    使 SINR 最大化的示例组合.
    
搜索策略:
    1. Selection:  UCT = Q(S,a) + c_uct * sqrt(ln(N(S)) / N(S,a))
    2. Expansion:  从候选池 C 中选择未展开的动作
    3. Simulation: 使用 SINR 公式快速估值 (不调用 LLM)
    4. Backpropagation: 沿路径更新 N 和 Q

最终输出:
    选择访问次数 N 最多的路径 (最稳健的选择)
"""

import math
import random
import numpy as np
from utils.sinr_reward import calculate_sinr


class MCTSNode:
    """MCTS 搜索树节点"""
    
    def __init__(self, state: frozenset, parent=None, action=None):
        """
        :param state: 当前已选示例的索引集合 (frozenset for hashability)
        :param parent: 父节点
        :param action: 从父节点到此节点选择的动作 (候选索引)
        """
        self.state = state          # 已选示例索引
        self.parent = parent        # 父节点
        self.action = action        # 到达该节点的动作
        self.children = {}          # {action_idx: MCTSNode}
        self.visits = 0             # 访问次数 N
        self.total_reward = 0.0     # 累积奖励 Q
        self.untried_actions = []   # 尚未扩展的动作
    
    @property
    def q_value(self):
        """平均奖励 Q/N"""
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits
    
    def uct_value(self, c_uct=1.41):
        """UCT 值: Q + c * sqrt(ln(N_parent) / N)"""
        if self.visits == 0:
            return float('inf')  # 未访问过的节点优先探索
        exploitation = self.q_value
        exploration = c_uct * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def is_fully_expanded(self):
        """是否所有动作都已扩展"""
        return len(self.untried_actions) == 0
    
    def is_terminal(self, k):
        """是否达到终止条件 (已选够 k 个)"""
        return len(self.state) >= k
    
    def best_child(self, c_uct=1.41):
        """选择 UCT 值最高的子节点"""
        return max(self.children.values(), key=lambda c: c.uct_value(c_uct))
    
    def most_visited_child(self):
        """选择访问次数最多的子节点 (用于最终决策)"""
        return max(self.children.values(), key=lambda c: c.visits)


class MCTS:
    """蒙特卡洛树搜索优化器"""
    
    @staticmethod
    def search(
        target_sql: str,
        target_embedding: np.ndarray,
        candidate_pool: list,
        candidate_sqls: list,
        candidate_embeddings: np.ndarray,
        k: int = 5,
        n_simulations: int = 200,
        c_uct: float = 1.41,
        lambda1: float = 0.5,
        lambda2: float = 0.3,
    ) -> list:
        """
        执行 MCTS 搜索, 从候选池中选择最优的 k 个示例
        
        :param target_sql: 目标问题的预生成 SQL
        :param target_embedding: 目标问题的嵌入向量
        :param candidate_pool: 候选示例列表 (经过 CFAR 筛选)
        :param candidate_sqls: 候选示例的 SQL 列表
        :param candidate_embeddings: 候选示例的嵌入矩阵 (n_candidates, d)
        :param k: 选择的示例数量
        :param n_simulations: 模拟迭代次数
        :param c_uct: UCT 探索系数
        :param lambda1: SINR 噪声权重
        :param lambda2: SINR 干扰权重
        :return: 选中的候选索引列表 (在 candidate_pool 中的索引)
        """
        n_candidates = len(candidate_pool)
        
        # 边界条件: 候选数 <= k, 直接全选
        if n_candidates <= k:
            return list(range(n_candidates))
        
        # 初始化根节点
        root = MCTSNode(state=frozenset())
        root.untried_actions = list(range(n_candidates))
        
        # 执行 N_sim 次模拟迭代
        for _ in range(n_simulations):
            node = root
            
            # ========== Step 1: Selection (选择) ==========
            # 沿 UCT 最优路径向下, 直到找到可扩展或终止的节点
            while not node.is_terminal(k) and node.is_fully_expanded() and node.children:
                node = node.best_child(c_uct)
            
            # ========== Step 2: Expansion (扩展) ==========
            if not node.is_terminal(k) and not node.is_fully_expanded():
                # 随机选择一个未尝试的动作
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                
                # 创建新子节点
                new_state = node.state | {action}
                child = MCTSNode(state=new_state, parent=node, action=action)
                
                # 设置子节点可用动作 (排除已选的)
                child.untried_actions = [
                    a for a in range(n_candidates) if a not in new_state
                ]
                
                node.children[action] = child
                node = child
            
            # ========== Step 3: Simulation (模拟/估值) ==========
            # 从当前节点开始, 随机补全到 k 个, 然后用 SINR 估值
            current_selection = set(node.state)
            available = [a for a in range(n_candidates) if a not in current_selection]
            
            # 随机补全
            n_needed = k - len(current_selection)
            if n_needed > 0 and available:
                rollout_additions = random.sample(available, min(n_needed, len(available)))
                current_selection |= set(rollout_additions)
            
            # 用 SINR 公式快速估值 (不调用 LLM!)
            selected_indices = list(current_selection)
            selected_sqls = [candidate_sqls[i] for i in selected_indices]
            selected_embeddings = candidate_embeddings[selected_indices] if candidate_embeddings is not None else None
            
            reward = calculate_sinr(
                target_sql=target_sql,
                candidate_sqls=selected_sqls,
                candidate_embeddings=selected_embeddings,
                lambda1=lambda1,
                lambda2=lambda2
            )
            
            # ========== Step 4: Backpropagation (反向传播) ==========
            # 沿路径向上更新 N 和 Q
            while node is not None:
                node.visits += 1
                node.total_reward += reward
                node = node.parent
        
        # ========== 最终输出: 选择访问次数 N 最多的路径 ==========
        # 从根节点开始, 逐层选择 most_visited_child
        result_indices = []
        node = root
        for _ in range(k):
            if not node.children:
                break
            node = node.most_visited_child()
            if node.action is not None:
                result_indices.append(node.action)
        
        # 如果 MCTS 搜索不足 k 个 (极端情况), 补充剩余
        if len(result_indices) < k:
            remaining = [i for i in range(n_candidates) if i not in result_indices]
            result_indices.extend(remaining[:k - len(result_indices)])
        
        return result_indices
