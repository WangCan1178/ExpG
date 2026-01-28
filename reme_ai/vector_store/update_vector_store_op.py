import asyncio
import json
import os
import time
from pathlib import Path
from typing import List

from flowllm import C, BaseAsyncOp
from flowllm.schema.vector_node import VectorNode
from loguru import logger

from reme_ai.schema.memory import BaseMemory


@C.register_op()
class UpdateVectorStoreOp(BaseAsyncOp):

    @staticmethod
    def _get_lock_file_path(workspace_id: str) -> Path:
        """获取工作空间的锁文件路径"""
        # 使用临时目录存放锁文件
        import tempfile
        lock_dir = Path(tempfile.gettempdir()) / "reme_ai_locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        # 使用 workspace_id 作为锁文件名，确保不同工作空间有独立的锁
        return lock_dir / f"{workspace_id}.lock"

    @staticmethod
    async def _acquire_file_lock(lock_path: Path, timeout: int = 30) -> bool:
        """
        异步尝试获取文件锁（使用文件存在作为锁机制）
        如果文件已存在，等待直到超时或文件被删除
        
        Args:
            lock_path: 锁文件路径
            timeout: 超时时间（秒）
            
        Returns:
            True 如果成功获取锁，False 如果超时
        """
        start_time = time.time()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        
        while time.time() - start_time < timeout:
            try:
                # 使用 O_EXCL 标志确保原子性创建（如果文件已存在会失败）
                # 这是原子操作，确保只有一个进程能成功创建文件
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    # 写入当前进程ID和时间戳，方便调试和清理僵尸锁
                    lock_content = f"{os.getpid()}\n{time.time()}\n"
                    os.write(fd, lock_content.encode('utf-8'))
                    os.fsync(fd)  # 确保数据写入磁盘
                finally:
                    os.close(fd)
                logger.debug(f"Acquired lock for {lock_path} (PID: {os.getpid()})")
                return True
            except FileExistsError:
                # 锁文件已存在，检查是否是僵尸锁（进程已不存在）
                try:
                    if lock_path.exists():
                        # 读取锁文件内容，检查进程是否还存在
                        try:
                            lock_content = lock_path.read_text().strip().split('\n')
                            if len(lock_content) >= 1:
                                lock_pid = int(lock_content[0])
                                # 检查进程是否还在运行（跨平台兼容）
                                try:
                                    # 在 Unix 系统上使用 os.kill，在 Windows 上使用其他方法
                                    if hasattr(os, 'kill'):
                                        os.kill(lock_pid, 0)  # 发送信号0，不杀死进程，只检查是否存在
                                    else:
                                        # Windows 系统，使用超时机制
                                        # 如果锁文件太旧（超过5分钟），认为是僵尸锁
                                        if len(lock_content) >= 2:
                                            try:
                                                lock_time = float(lock_content[1])
                                                if time.time() - lock_time > 300:  # 5分钟超时
                                                    raise ProcessLookupError("Lock file too old")
                                            except (ValueError, IndexError):
                                                pass
                                    # 进程存在，锁有效，异步等待
                                    await asyncio.sleep(0.1)
                                    continue
                                except (ProcessLookupError, OSError):
                                    # 进程不存在或无法检查，这是僵尸锁，删除它
                                    logger.warning(f"Found stale lock file (PID {lock_pid} not found or check failed), removing it")
                                    try:
                                        lock_path.unlink()
                                    except Exception:
                                        pass
                                    continue
                                except PermissionError:
                                    # 没有权限检查进程，可能是其他用户的进程，异步等待
                                    await asyncio.sleep(0.1)
                                    continue
                        except (ValueError, IndexError, IOError):
                            # 锁文件格式错误或无法读取，尝试删除
                            logger.warning(f"Lock file {lock_path} is corrupted, removing it")
                            try:
                                lock_path.unlink()
                            except Exception:
                                pass
                            continue
                except Exception as e:
                    logger.debug(f"Error checking lock file: {e}, waiting...")
                    await asyncio.sleep(0.1)
                    continue
            except Exception as e:
                logger.warning(f"Failed to acquire lock: {e}, retrying...")
                await asyncio.sleep(0.1)
        
        logger.warning(f"Failed to acquire lock for {lock_path} within {timeout} seconds")
        return False

    @staticmethod
    def _release_file_lock(lock_path: Path):
        """释放文件锁（删除锁文件）"""
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to release lock {lock_path}: {e}")

    async def async_execute(self):
        workspace_id: str = self.context.workspace_id
        no_persist: bool = bool(self.context.get("no_persist", False))

        # 如果上游标记了 no_persist，使用文件锁机制安全地写回
        if no_persist:
            logger.info("no_persist=True, using file lock for safe concurrent write-back")
            lock_path = self._get_lock_file_path(workspace_id)
            
            # 尝试获取文件锁（异步）
            if not await self._acquire_file_lock(lock_path, timeout=30):
                logger.error(f"Failed to acquire lock for workspace {workspace_id}, skipping write-back")
                self.context.response.metadata["update_result"] = {
                    "deleted_count": 0,
                    "inserted_count": 0,
                    "error": "Failed to acquire file lock"
                }
                return
            
            try:
                # 在锁保护下执行写回操作
                deleted_memory_ids: List[str] = self.context.response.metadata.get("deleted_memory_ids", [])
                if deleted_memory_ids:
                    await self.vector_store.async_delete(node_ids=deleted_memory_ids, workspace_id=workspace_id)
                    logger.info(f"[Locked] delete memory_ids={json.dumps(deleted_memory_ids, indent=2)}")

                insert_memory_list: List[BaseMemory] = self.context.response.metadata.get("memory_list", [])
                if insert_memory_list:
                    insert_nodes: List[VectorNode] = [x.to_vector_node() for x in insert_memory_list]
                    await self.vector_store.async_insert(nodes=insert_nodes, workspace_id=workspace_id)
                    logger.info(f"[Locked] insert insert_node.size={len(insert_nodes)}")

                # Store results in context
                self.context.response.metadata["update_result"] = {
                    "deleted_count": len(deleted_memory_ids) if deleted_memory_ids else 0,
                    "inserted_count": len(insert_memory_list) if insert_memory_list else 0
                }
            finally:
                # 确保释放锁
                self._release_file_lock(lock_path)
                logger.info(f"Released lock for workspace {workspace_id}")
        else:
            # 非并行模式，但为了数据安全，仍然使用文件锁防止并发写入导致文件损坏
            # 这可以防止多个异步请求同时写入同一个文件时出现 JSON 解析错误
            lock_path = self._get_lock_file_path(workspace_id)
            
            # 尝试获取文件锁（异步）
            if not await self._acquire_file_lock(lock_path, timeout=30):
                logger.error(f"Failed to acquire lock for workspace {workspace_id}, skipping write-back")
                self.context.response.metadata["update_result"] = {
                    "deleted_count": 0,
                    "inserted_count": 0,
                    "error": "Failed to acquire file lock"
                }
                return
            
            try:
                deleted_memory_ids: List[str] = self.context.response.metadata.get("deleted_memory_ids", [])
                if deleted_memory_ids:
                    await self.vector_store.async_delete(node_ids=deleted_memory_ids, workspace_id=workspace_id)
                    logger.info(f"delete memory_ids={json.dumps(deleted_memory_ids, indent=2)}")

                insert_memory_list: List[BaseMemory] = self.context.response.metadata.get("memory_list", [])
                if insert_memory_list:
                    insert_nodes: List[VectorNode] = [x.to_vector_node() for x in insert_memory_list]
                    await self.vector_store.async_insert(nodes=insert_nodes, workspace_id=workspace_id)
                    logger.info(f"insert insert_node.size={len(insert_nodes)}")

                # Store results in context
                self.context.response.metadata["update_result"] = {
                    "deleted_count": len(deleted_memory_ids) if deleted_memory_ids else 0,
                    "inserted_count": len(insert_memory_list) if insert_memory_list else 0
                }
            finally:
                # 确保释放锁
                self._release_file_lock(lock_path)
                logger.debug(f"Released lock for workspace {workspace_id}")
