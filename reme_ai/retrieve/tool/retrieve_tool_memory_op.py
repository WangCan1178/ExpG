from typing import List
from collections import Counter
import math

from flowllm import C, BaseAsyncOp
from flowllm.schema.vector_node import VectorNode
from loguru import logger

from reme_ai.schema.memory import ToolMemory, vector_node_to_memory
from reme_ai.utils.file_lock import FileLockManager
from reme_ai.summary.tool.summary_tool_memory_op import SummaryToolMemoryOp


@C.register_op()
class RetrieveToolMemoryOp(BaseAsyncOp):
    file_path: str = __file__

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _determine_add_to(self, memory: ToolMemory) -> str:
        """
        确定 add_to 的值。
        默认返回 "icl"，当满足特定条件时返回 "config"。
        
        Args:
            memory: ToolMemory 对象
            
        Returns:
            "icl" 或 "config"
        """
        tool_call_results = memory.tool_call_results

        # 计算论文中的 Q，并按论文设定得到 MIN_CALL_COUNT = Q/2（向下取整，至少为 1）
        Q = SummaryToolMemoryOp._compute_Q(tool_call_results)
        min_call_count = max(1, Q // 2)
        
        # 如果 tool_call_results 数量 <= MIN_CALL_COUNT，返回默认值 "icl"
        if len(tool_call_results) <= min_call_count:
            return "icl"
        
        # 计算阈值：4/5 * 调用次数，向上取整
        threshold = math.ceil((4 / 5) * len(tool_call_results))
        
        # 条件1：检查不同经验中多次分布一致次数
        # 统计每个 scores 分布出现的次数
        scores_counter = Counter()
        for call in tool_call_results:
            scores = getattr(call, 'scores', []) or []
            # 将 scores 转换为 tuple 以便作为字典的 key
            scores_tuple = tuple(scores)
            scores_counter[scores_tuple] += 1
        
        # 检查是否有分布出现次数 >= threshold，且不是全满分的情况
        max_distribution_count = max(scores_counter.values()) if scores_counter else 0
        if max_distribution_count >= threshold:
            # 获取出现次数最多的分布
            most_common_distribution = max(scores_counter, key=scores_counter.get)
            # 检查是否是全满分分布（所有分数都是1.0）
            is_all_perfect = (sum(most_common_distribution) == len(most_common_distribution))
            if not is_all_perfect:
                print(f"Tool {memory.when_to_use}: max distribution count {max_distribution_count} >= threshold {threshold}, returning 'config'")
                return "config"
        
        # # 条件2：检查多次检测出是工具本身问题的次数
        # tool_problem_count = sum(1 for call in tool_call_results if getattr(call, 'tool_problem', 0) == 1)
        # if tool_problem_count >= threshold:
        #     logger.info(f"Tool {memory.when_to_use}: tool_problem count {tool_problem_count} >= threshold {threshold}, returning 'config'")
        #     return "config"
        
        # 默认返回 "icl"
        return "icl"

    def _format_tool_memories(self, memories: List[ToolMemory]) -> str:
        """Format tool memories into a structured document format"""
        lines = []
        lines.append(f"Retrieved {len(memories)} tool memory(ies):\n")

        for idx, memory in enumerate(memories, 1):
            lines.append(f"Tool: {memory.when_to_use}")
            lines.append(memory.content)
            
            if idx < len(memories):
                lines.append("\n---\n")

        return "\n".join(lines)

    async def async_execute(self):
        tool_names: str = self.context.get("tool_names", "")
        workspace_id: str = self.context.workspace_id

        if not tool_names:
            logger.warning("tool_names is empty, skipping processing")
            self.context.response.answer = "tool_names is required"
            self.context.response.success = False
            return

        # Split tool names by comma
        tool_name_list = [name.strip() for name in tool_names.split(",") if name.strip()]
        logger.info(f"workspace_id={workspace_id} retrieving {len(tool_name_list)} tools: {tool_name_list}")

        # Search for each tool in the vector store
        matched_tool_memories: List[ToolMemory] = []

        for tool_name in tool_name_list:
            # 使用文件锁保护读取操作，防止与写入操作冲突
            # 使用默认参数捕获循环变量，避免闭包问题
            async def search_with_lock(tn=tool_name):
                return await self.vector_store.async_search(
                    query=tn,
                    workspace_id=workspace_id,
                    top_k=1
                )
            
            nodes: List[VectorNode] = await FileLockManager.with_file_lock(
                workspace_id=workspace_id,
                operation=search_with_lock,
                timeout=30
            )

            if nodes:
                top_node = nodes[0]
                memory = vector_node_to_memory(top_node)

                # Ensure it's a ToolMemory and when_to_use matches
                if isinstance(memory, ToolMemory) and memory.when_to_use == tool_name:
                    matched_tool_memories.append(memory)
                    logger.info(f"Found tool_memory for tool_name={tool_name}, "
                                f"memory_id={memory.memory_id}, "
                                f"total_calls={len(memory.tool_call_results)}")
                else:
                    logger.warning(f"No exact match found for tool_name={tool_name}")
            else:
                logger.warning(f"No memory found for tool_name={tool_name}")

        if not matched_tool_memories:
            logger.info("No matching tool memories found")
            self.context.response.answer = "No matching tool memories found"
            self.context.response.success = False
            return

        # 构建返回的 metadata，只包含 content 和 add_to 字段
        metadata_list = []
        for memory in matched_tool_memories:
            add_to = self._determine_add_to(memory)
            metadata_list.append({
                "content": memory.content,
                "add_to": add_to
            })
            logger.info(f"Retrieved tool: {memory.when_to_use}, "
                       f"total_calls={len(memory.tool_call_results)}, "
                       f"content_length={len(memory.content)}, "
                       f"add_to={add_to}")

        # Set response - 只返回 success 和 metadata
        self.context.response.success = True
        self.context.response.metadata = metadata_list
