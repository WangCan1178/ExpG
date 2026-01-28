"""文件锁工具模块，用于保护 JSONL 文件的并发读写操作。"""

import asyncio
import os
import time
from pathlib import Path
from typing import Callable, Any
from loguru import logger


class FileLockManager:
    """文件锁管理器，用于保护文件操作的并发安全。"""
    
    @staticmethod
    def _get_lock_file_path(workspace_id: str) -> Path:
        """获取工作空间的锁文件路径"""
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

    @staticmethod
    async def with_file_lock(workspace_id: str, operation: Callable, timeout: int = 30, *args, **kwargs) -> Any:
        """
        在文件锁保护下执行操作（用于读取操作）
        
        Args:
            workspace_id: 工作空间ID
            operation: 要执行的操作（异步函数）
            timeout: 获取锁的超时时间（秒）
            *args, **kwargs: 传递给 operation 的参数
            
        Returns:
            operation 的返回值
        """
        lock_path = FileLockManager._get_lock_file_path(workspace_id)
        
        # 尝试获取文件锁（异步）
        if not await FileLockManager._acquire_file_lock(lock_path, timeout=timeout):
            logger.warning(f"Failed to acquire lock for workspace {workspace_id}, operation may fail")
            # 即使获取锁失败，也尝试执行操作（但会有风险）
            # 或者可以选择抛出异常
            # raise RuntimeError(f"Failed to acquire file lock for workspace {workspace_id}")
        
        try:
            # 在锁保护下执行操作（operation 应该是异步函数）
            result = await operation(*args, **kwargs)
            return result
        finally:
            # 确保释放锁
            FileLockManager._release_file_lock(lock_path)
            logger.debug(f"Released lock for workspace {workspace_id}")

