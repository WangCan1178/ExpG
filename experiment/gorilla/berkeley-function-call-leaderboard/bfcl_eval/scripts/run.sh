python run.py --model qwen3-8b --method few shot > dspy_qwen3-8b.log 2>&1
python run.py --model qwen3-8b --method draft > draft_qwen3-8b.log 2>&1
python run.py --model qwen3-8b --method mem > mem_qwen3-8b.log 2>&1
python run.py --model qwen3-8b --method m0 > m0_qwen3-8b.log 2>&1
python run.py --model deepseek-v3 --method draft > draft_deepseek-v3.log 2>&1
python run.py --model deepseek-v3 --method mem > mem_deepseek-v3.log 2>&1
python run.py --model deepseek-v3 --method m0 > m0_deepseek-v3.log 2>&1
python run.py --model qwen3-235b-a22b --method mem > mem_235b.log 2>&1