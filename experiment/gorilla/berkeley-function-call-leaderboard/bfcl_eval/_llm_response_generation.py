import argparse
import multiprocessing as mp
import os
import shutil
import traceback
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import threading
import queue
from copy import deepcopy
from typing import TYPE_CHECKING

from bfcl_eval.constants.eval_config import (
    PROJECT_ROOT,
    RESULT_PATH,
    TEST_IDS_TO_GENERATE_PATH,
    RESULT_FILE_PATTERN
)
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.eval_checker.eval_runner_helper import load_file
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.utils import *
from tqdm import tqdm

from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler


def get_args():
    parser = argparse.ArgumentParser()
    # Refer to model_choice for supported models.
    parser.add_argument("--model", type=str, default="gorilla-openfunctions-v2", nargs="+")
    # Refer to test_categories for supported categories.
    parser.add_argument("--test-category", type=str, default="all", nargs="+")

    # Parameters for the model that you want to test.
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--include-input-log", action="store_true", default=False)
    parser.add_argument("--exclude-state-log", action="store_true", default=False)
    parser.add_argument("--num-threads", required=False, type=int)
    parser.add_argument("--num-gpus", default=1, type=int)
    parser.add_argument("--backend", default="sglang", type=str, choices=["vllm", "sglang"])
    parser.add_argument("--gpu-memory-utilization", default=0.9, type=float)
    parser.add_argument("--result-dir", default=None, type=str)
    parser.add_argument("--run-ids", action="store_true", default=False)
    parser.add_argument("--allow-overwrite", "-o", action="store_true", default=False)
    parser.add_argument(
        "--skip-server-setup",
        action="store_true",
        default=False,
        help="Skip vLLM/SGLang server setup and use existing endpoint specified by the LOCAL_SERVER_ENDPOINT and LOCAL_SERVER_PORT environment variables.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Method to use for generation (e.g., 'mem', 'm0', 'fewshot', 'draft'). If 'mem', add memory information from the vector store to the context during model generation.",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=None,
        help="Number of samples to generate for each test case. If specified, each test case will be sampled n_sample times and results will be saved with suffix _n_sample.",
    )
    # Optional local model path
    parser.add_argument(
        "--local-model-path",
        type=str,
        default=None,
        help="Specify the path to a local directory containing the model's config/tokenizer/weights for fully offline inference. Use this only if the model weights are stored in a location other than the default HF_HOME directory.",
    )
    args = parser.parse_args()

    return args


def build_handler(model_name, temperature, method=None):
    config = MODEL_CONFIG_MAPPING[model_name]
    handler = config.model_handler(
        model_name=config.model_name,
        temperature=temperature,
        registry_name=model_name,
        is_fc_model=config.is_fc_model,
        method=method,
    )
    return handler


def get_involved_test_entries(test_category_args, run_ids):
    all_test_categories, all_test_entries_involved = [], []
    if run_ids:
        all_test_categories, all_test_entries_involved = load_test_entries_from_id_file(
            TEST_IDS_TO_GENERATE_PATH
        )

    else:
        all_test_categories = parse_test_category_argument(test_category_args)
        for test_category in all_test_categories:
            all_test_entries_involved.extend(load_dataset_entry(test_category))

    return (
        all_test_categories,
        all_test_entries_involved,
    )


def collect_test_cases(args, model_name, all_test_categories, all_test_entries_involved):
    model_name_dir = model_name.replace("/", "_")
    model_result_dir = args.result_dir / model_name_dir

    existing_result = []
    n_sample = getattr(args, 'n_sample', None)
    
    # When using n_sample (including n_sample=1), collect existing IDs from sample files
    # Note: When n_sample=1, files are saved as _result_1.json (not _result.json)
    if n_sample is not None:
        # Collect existing IDs per sample (sample_idx: 0, 1, ..., n_sample-1)
        existing_ids_per_sample = [set() for _ in range(n_sample)]
        
        for test_category in all_test_categories:
            base_file_name = get_file_name_by_category(test_category, is_result_file=True)
            group_dir_path = model_result_dir / get_directory_structure_by_category(test_category)
            
            # Check each sample file separately
            for sample_idx in range(n_sample):
                file_name = base_file_name.replace("_result.json", f"_result_{sample_idx + 1}.json")
                file_path = group_dir_path / file_name
                
                if file_path.exists():
                    if not args.allow_overwrite:
                        # Load existing entries and extract IDs
                        for entry in load_file(file_path):
                            existing_ids_per_sample[sample_idx].add(entry["id"])
                    elif not args.run_ids:
                        file_path.unlink()
                    # If run_ids and allow_overwrite, do nothing (will overwrite)
            
            if is_memory(test_category):
                # Handle prerequisite files for memory test cases
                prereq_base_file_name = get_file_name_by_category(f"{test_category}_prereq", is_result_file=True)
                for sample_idx in range(n_sample):
                    prereq_file_name = prereq_base_file_name.replace("_result.json", f"_result_{sample_idx + 1}.json")
                    prereq_file_path = group_dir_path / prereq_file_name
                    
                    if prereq_file_path.exists():
                        if not args.allow_overwrite:
                            for entry in load_file(prereq_file_path):
                                existing_ids_per_sample[sample_idx].add(entry["id"])
                        elif not args.run_ids:
                            prereq_file_path.unlink()
            
            if is_memory(test_category):
                # Handle snapshot folder
                snapshot_folder = model_result_dir / "memory_snapshot" / test_category
                if snapshot_folder.exists():
                    if not args.allow_overwrite:
                        pass
                    elif not args.run_ids:
                        shutil.rmtree(snapshot_folder)
                    else:
                        pass
        
        # For batch sampling (n_sample > 1): only skip tasks that exist in ALL sample files
        # For single sampling (n_sample = 1): skip tasks that exist in the sample file
        # (This ensures we still generate missing samples)
        # if n_sample > 1:
        all_samples_existing_ids = set.intersection(*existing_ids_per_sample) if existing_ids_per_sample else set()
        # else:
        #     # n_sample == 1: use the IDs from the single sample file
        #     all_samples_existing_ids = existing_ids_per_sample[0] if existing_ids_per_sample else set()
        
        test_cases_to_generate = [
            test_case
            for test_case in all_test_entries_involved
            if test_case["id"] not in all_samples_existing_ids
        ]
    else:
        # Original logic for single sample (when n_sample is None)
        # Files are saved as _result.json (without number suffix)
        for test_category in all_test_categories:
            # TODO: Simplify the handling of memory prerequisite entries/categories
            # Build file paths with or without n_sample suffix
            base_file_name = get_file_name_by_category(test_category, is_result_file=True)
            result_file_paths = [
                model_result_dir
                / get_directory_structure_by_category(test_category)
                / base_file_name
            ]
            
            if is_memory(test_category):
                # Memory test cases have the pre-requisite entries in a separate file
                prereq_base_file_name = get_file_name_by_category(f"{test_category}_prereq", is_result_file=True)
                result_file_paths.append(
                    model_result_dir
                    / get_directory_structure_by_category(test_category)
                    / prereq_base_file_name
                )

            for file_path in result_file_paths:
                if file_path.exists():
                    # Not allowing overwrite, we will load the existing results
                    if not args.allow_overwrite:
                        existing_result.extend(load_file(file_path))
                    # Allow overwrite and not running specific test ids, we will delete the existing result file before generating new results
                    elif not args.run_ids:
                        file_path.unlink()
                    # Allow overwrite and running specific test ids, we will do nothing here
                    else:
                        pass

            if is_memory(test_category):
                # We also need to special handle the pre-requisite entries and the snapshot result for memory test cases
                snapshot_folder = model_result_dir / "memory_snapshot" / test_category
                if snapshot_folder.exists():
                    if not args.allow_overwrite:
                        pass
                    elif not args.run_ids:
                        shutil.rmtree(snapshot_folder)
                    else:
                        # TODO: If run_ids and id involes prereq entries, we should just delete those snapshot files
                        # It's not implemented yet, but it won't affect the accuracy, as those files will be overwritten anyway (assume generation success)
                        pass

        existing_ids = [entry["id"] for entry in existing_result]

        test_cases_to_generate = [
            test_case
            for test_case in all_test_entries_involved
            if test_case["id"] not in existing_ids
        ]

    # Skip format sensitivity test cases for FC models
    if (
        any(is_format_sensitivity(test_category) for test_category in all_test_categories)
        and MODEL_CONFIG_MAPPING[model_name].is_fc_model
    ):
        test_cases_to_generate = [
            test_case
            for test_case in test_cases_to_generate
            if not is_format_sensitivity(test_case["id"])
        ]

    test_cases_to_generate = clean_up_memory_prereq_entries(test_cases_to_generate)
    # TODO: Should we move these to the load_dataset_entry function?
    test_cases_to_generate = populate_initial_settings_for_memory_test_cases(
        test_cases_to_generate, model_result_dir
    )
    test_cases_to_generate = populate_initial_settings_for_web_search_test_cases(
        test_cases_to_generate
    )

    return sorted(test_cases_to_generate, key=sort_key)


def multi_threaded_inference(handler, test_case, include_input_log, exclude_state_log, n_sample=None, missing_samples=None):

    assert type(test_case["function"]) is list

    # 如果指定了missing_samples且为空，说明所有样本都已存在，直接返回空列表
    if missing_samples is not None and len(missing_samples) == 0:
        return []

    if n_sample is not None:
        # 多轮对话场景不支持批量采样，直接使用循环调用
        # 因为多轮对话需要按轮次循环处理，每轮都依赖于上一轮的结果
        if contain_multi_turn_interaction(test_case["id"]):
            # 多轮对话直接使用循环调用，跳过批量采样尝试
            pass
        else:
            # 尝试使用批量采样（一次调用生成多个候选结果）
            # 对于支持批量采样的 handler（如 OpenAICompletionsHandler），这会更高效
            # 注意：需要对 test_case 进行深拷贝，避免修改原始对象导致 system prompt 重复添加
            try:
                test_case_copy = deepcopy(test_case)
                result, metadata = handler.inference(
                    test_case_copy, include_input_log, exclude_state_log, n_sample=n_sample
                )
                
                # 检查返回结果是否是批量采样结果（列表）
                if isinstance(result, list) and len(result) == n_sample:
                    # 批量采样成功，返回多个结果
                    results_to_write = []
                    for sample_idx, single_result in enumerate(result):
                        result_to_write = {
                            "id": test_case["id"],
                            "result": single_result,
                            "sample_idx": sample_idx,
                            **metadata,
                        }
                        results_to_write.append(result_to_write)
                    return results_to_write
            except Exception as e:
                # 批量采样失败，回退到循环调用
                tqdm.write(f"⚠️ 批量采样失败，回退到循环调用: {e}")
        
        # 回退到循环调用（原有逻辑）
        # 注意：需要对 test_case 进行深拷贝，避免每次调用时修改同一个对象导致 system prompt 重复添加
        results_to_write = []
        # 如果指定了missing_samples，只生成缺失的样本；否则生成所有样本
        samples_to_generate = missing_samples if missing_samples is not None else list(range(n_sample))
        for sample_idx in samples_to_generate:
            try:
                # 深拷贝 test_case，避免修改原始对象
                test_case_copy = deepcopy(test_case)
                result, metadata = handler.inference(
                    test_case_copy, include_input_log, exclude_state_log
                )
            except Exception as e:
                # This is usually the case when the model getting stuck on one particular test case.
                # For example, timeout error or FC model returning invalid JSON response.
                # Since temperature is already set to 0.001, retrying the same test case will not help.
                # So we continue the generation process and record the error message as the model response
                error_block = (
                    "-" * 100
                    + "\n❗️❗️ Error occurred during inference. Continuing to next test case.\n"
                    + f"❗️❗️ Test case ID: {test_case['id']}, Sample: {sample_idx + 1}/{n_sample}, Error: {str(e)}\n"
                    + traceback.format_exc(limit=10)
                    + "-" * 100
                )
                tqdm.write(error_block)

                result = f"Error during inference: {str(e)}"
                metadata = {"traceback": traceback.format_exc()}

            result_to_write = {
                "id": test_case["id"],
                "result": result,
                "sample_idx": sample_idx,
                **metadata,
            }
            results_to_write.append(result_to_write)
        
        return results_to_write
    else:
        # Original behavior: single sample
        try:
            result, metadata = handler.inference(
                test_case, include_input_log, exclude_state_log
            )
        except Exception as e:
            # This is usually the case when the model getting stuck on one particular test case.
            # For example, timeout error or FC model returning invalid JSON response.
            # Since temperature is already set to 0.001, retrying the same test case will not help.
            # So we continue the generation process and record the error message as the model response
            error_block = (
                "-" * 100
                + "\n❗️❗️ Error occurred during inference. Continuing to next test case.\n"
                + f"❗️❗️ Test case ID: {test_case['id']}, Error: {str(e)}\n"
                + traceback.format_exc(limit=10)
                + "-" * 100
            )
            tqdm.write(error_block)

            result = f"Error during inference: {str(e)}"
            metadata = {"traceback": traceback.format_exc()}

        result_to_write = {
            "id": test_case["id"],
            "result": result,
            **metadata,
        }
        
        # Set sample_idx=0 when n_sample=1 to ensure correct file naming (_result_1.json)
        if n_sample == 1:
            result_to_write["sample_idx"] = 0

        return result_to_write


def generate_results(args, model_name, test_cases_total, existing_ids_per_sample=None):
    method = getattr(args, 'method', None)
    # 固定温度为0.5
    temperature = 0.5
    
    # 如果使用 m0 方法，预先初始化 mem0 实例，避免在任务处理过程中并发初始化导致 Qdrant 冲突
    if method == 'm0':
        try:
            import sys
            from pathlib import Path
            run_script_path = Path(__file__).resolve().parent.parent / "scripts" / "run.py"
            if run_script_path.exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location("run_module", run_script_path)
                run_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(run_module)
                # 预先初始化 mem0 实例
                memory_instance = run_module.get_mem0_memory_instance()
                if memory_instance is None:
                    print("⚠️ 警告: mem0 Memory 实例初始化失败，m0 方法可能无法正常工作")
                else:
                    print("✅ 已预先初始化 mem0 Memory 实例，准备开始处理任务")
        except Exception as e:
            print(f"⚠️ 预先初始化 mem0 Memory 实例时出错: {e}")
            import traceback
            traceback.print_exc()
    
    handler = build_handler(model_name, temperature, method=method)

    if isinstance(handler, OSSHandler):
        handler: OSSHandler
        is_oss_model = True
        # For OSS models, if the user didn't explicitly set the number of threads,
        # we default to 100 threads to speed up the inference.
        num_threads = (
            args.num_threads
            if args.num_threads is not None
            else LOCAL_SERVER_MAX_CONCURRENT_REQUEST
        )
    else:
        handler: BaseHandler
        is_oss_model = False
        # Default to 10 threads for API models to enable parallel requests
        num_threads = args.num_threads if args.num_threads is not None else 10

    # Use a separate thread to write the results to the file to avoid concurrent IO issues
    n_sample = getattr(args, 'n_sample', None)
    def _writer():
        """Consume result dicts from the queue and write them with exclusive access."""
        while True:
            item = write_queue.get()
            if item is None:
                break
            handler.write(item, result_dir=args.result_dir, update_mode=args.run_ids, n_sample=n_sample)
            write_queue.task_done()

    write_queue: queue.Queue = queue.Queue()

    writer_thread = threading.Thread(target=_writer, daemon=True)
    writer_thread.start()

    try:
        if is_oss_model:
            handler.spin_up_local_server(
                num_gpus=args.num_gpus,
                gpu_memory_utilization=args.gpu_memory_utilization,
                backend=args.backend,
                skip_server_setup=args.skip_server_setup,
                local_model_path=args.local_model_path,
            )

        # ───── dependency bookkeeping ──────────────────────────────
        dependencies = {
            test_case["id"]: set(test_case.get("depends_on", []))
            for test_case in test_cases_total
        }
        children_of = defaultdict(list)
        for test_case in test_cases_total:
            for dependency_id in test_case.get("depends_on", []):
                children_of[dependency_id].append(test_case["id"])

        id_to_test_case = {test_case["id"]: test_case for test_case in test_cases_total}

        # 分离多轮任务和非多轮任务
        multi_turn_cases = []
        non_multi_turn_cases = []
        for test_case in test_cases_total:
            if contain_multi_turn_interaction(test_case["id"]):
                multi_turn_cases.append(test_case["id"])
            else:
                non_multi_turn_cases.append(test_case["id"])

        ready_queue = deque(
            [
                test_case_id
                for test_case_id, dependency_ids in dependencies.items()
                if not dependency_ids
            ]
        )
        in_flight: dict[Future, str] = {}  # future -> test_case_id
        completed = set()
        
        # 跟踪正在执行的多轮任务数量，限制最多3个并发
        multi_turn_in_flight = 0
        MAX_MULTI_TURN_CONCURRENT = 3

        n_sample = getattr(args, 'n_sample', None)
        total_tasks = len(test_cases_total) * (n_sample if n_sample else 1)
        with ThreadPoolExecutor(max_workers=num_threads) as pool, tqdm(
            total=total_tasks,
            desc=f"Generating results for {model_name}",
            position=0,         
            leave=True,           
            dynamic_ncols=True,   
            mininterval=0.2,      
            smoothing=0.1,        
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:

            # seed initial ready tasks (允许多轮任务并行，但限制并发数)
            temp_queue = deque()
            while ready_queue and len(in_flight) < num_threads:
                test_case_id = ready_queue.popleft()
                test_case = id_to_test_case[test_case_id]
                
                # 如果是多轮任务，检查并发限制
                if contain_multi_turn_interaction(test_case_id):
                    if multi_turn_in_flight >= MAX_MULTI_TURN_CONCURRENT:
                        # 已达到多轮任务并发上限，暂存到 temp_queue
                        temp_queue.append(test_case_id)
                        continue
                    # 可以执行，增加计数
                    multi_turn_in_flight += 1
                
                # 获取该测试用例已存在的样本索引
                missing_samples = None
                if existing_ids_per_sample is not None and n_sample is not None:
                    missing_samples = [
                        sample_idx 
                        for sample_idx in range(n_sample)
                        if test_case_id not in existing_ids_per_sample[sample_idx]
                    ]
                future = pool.submit(
                    multi_threaded_inference,
                    handler,
                    test_case,
                    args.include_input_log,
                    args.exclude_state_log,
                    n_sample,
                    missing_samples,
                )
                in_flight[future] = test_case_id
            
            # 将暂存的任务放回 ready_queue
            ready_queue.extendleft(reversed(temp_queue))

            # main scheduler loop
            while in_flight or ready_queue:
                # 处理并行任务（包括多轮任务）
                if in_flight:
                    done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in done:
                        test_case_id = in_flight.pop(future)
                        result = future.result()
                        
                        # 如果是多轮任务，减少并发计数
                        if contain_multi_turn_interaction(test_case_id):
                            multi_turn_in_flight = max(0, multi_turn_in_flight - 1)

                        # Handle both single result and list of results (when n_sample > 1)
                        if isinstance(result, list):
                            # Multiple samples: enqueue each result separately
                            for result_dict in result:
                                write_queue.put(result_dict)
                                pbar.update()
                        else:
                            # Single sample: original behavior
                            write_queue.put(result)
                            pbar.update()

                        completed.add(test_case_id)

                        # unlock children
                        for child_id in children_of[test_case_id]:
                            dependencies[child_id].discard(test_case_id)
                            if not dependencies[child_id]:
                                ready_queue.append(child_id)

                # refill the pool up to max_workers (允许多轮任务并行，但限制并发数)
                temp_queue = deque()
                while ready_queue and len(in_flight) < num_threads:
                    test_case_id = ready_queue.popleft()
                    test_case = id_to_test_case[test_case_id]
                    
                    # 如果是多轮任务，检查并发限制
                    if contain_multi_turn_interaction(test_case_id):
                        if multi_turn_in_flight >= MAX_MULTI_TURN_CONCURRENT:
                            # 已达到多轮任务并发上限，暂存到 temp_queue
                            temp_queue.append(test_case_id)
                            continue
                        # 可以执行，增加计数
                        multi_turn_in_flight += 1
                    
                    # 获取该测试用例已存在的样本索引
                    missing_samples = None
                    if existing_ids_per_sample is not None and n_sample is not None:
                        missing_samples = [
                            sample_idx 
                            for sample_idx in range(n_sample)
                            if test_case_id not in existing_ids_per_sample[sample_idx]
                        ]
                    future = pool.submit(
                        multi_threaded_inference,
                        handler,
                        test_case,
                        args.include_input_log,
                        args.exclude_state_log,
                        n_sample,
                        missing_samples,
                    )
                    in_flight[future] = test_case_id
                
                # 将暂存的任务放回 ready_queue
                ready_queue.extendleft(reversed(temp_queue))

    finally:
        # Signal writer thread to finish and wait for it
        write_queue.put(None)
        writer_thread.join()

        if is_oss_model:
            handler.shutdown_local_server()


def main(args):

    # Note: The following environment variables are needed for the memory vector store implementation
    # Otherwise you get segfault or huggingface tokenizer warnings
    # disable HuggingFace tokenizers’ thread pool
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # limit all OpenMP/MKL threads to 1
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    # use spawn method for multiprocessing
    mp.set_start_method("spawn", force=True)

    if type(args.model) is not list:
        args.model = [args.model]
    if type(args.test_category) is not list:
        args.test_category = [args.test_category]

    (
        all_test_categories,
        all_test_entries_involved,
    ) = get_involved_test_entries(args.test_category, args.run_ids)

    for model_name in args.model:
        if model_name not in MODEL_CONFIG_MAPPING:
            raise ValueError(
                f"Unknown model_name '{model_name}'.\n"
                "• For officially supported models, please refer to `SUPPORTED_MODELS.md`.\n"
                "• For running new models, please refer to `README.md` and `CONTRIBUTING.md`."
            )
    # tqdm.write(f"Generating results for {args.model}")
    # if args.run_ids:
    #     tqdm.write("Running specific test cases. Ignoring `--test-category` argument.")
    # else:
    #     tqdm.write(f"Running full test cases for categories: {all_test_categories}.")

    if any(is_format_sensitivity(test_category) for test_category in all_test_categories):
        for model_name in args.model:
            if MODEL_CONFIG_MAPPING[model_name].is_fc_model:
                tqdm.write(
                    "⚠️ Warning: Format sensitivity test cases are only supported for prompting (non-FC) models. "
                    f"Since {model_name} is a FC model based on its config, the format sensitivity test cases will be skipped."
                )

    if args.result_dir is not None:
        args.result_dir = PROJECT_ROOT / args.result_dir
    else:
        args.result_dir = RESULT_PATH

    for model_name in args.model:
        result = collect_test_cases(
            args,
            model_name,
            all_test_categories,
            deepcopy(all_test_entries_involved),
        )
        if isinstance(result, tuple):
            test_cases_total, existing_ids_per_sample = result
        else:
            test_cases_total = result
            existing_ids_per_sample = None

        if len(test_cases_total) == 0:
            tqdm.write(
                f"✅ All selected test cases have been previously generated for {model_name}. No new test cases to generate."
            )
        else:
            generate_results(args, model_name, test_cases_total, existing_ids_per_sample)
            # Sort the result files by id at the end
            for model_result_json in args.result_dir.rglob(RESULT_FILE_PATTERN):
                sort_file_content_by_id(model_result_json)
