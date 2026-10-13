Table 3. All numbers are Pass@1 on BFCL-V3 with Qwen3-8B, with each row using exactly one replacement (no combinations).

|Phase|Variant|Accuracy|Reduction vs Original |
|------------------------|---------------------------------------------------------------------------------------------------|---------------------------------------------|------------------------|
|Original ExpG|-|67.46%|0.00%|
|Experience Acquisition |Directly use full trajectories (no experiences)|60.08% (sometimes hits context-length limits)|-7.38% |
|Experience Acquisition |Directly prompt the LLM to extract experiences (no multi-aspect attribution, unstructured experiences)|63.34%|-4.12% |
|Experience Distillation|Use only the most recent 1 experience|62.26%|-5.20% |
|Experience Distillation|Keep only the most recent 3 experiences|64.64%|-2.82% |
|Experience Distillation|Remove the experience filter step|65.94%|-1.52% |
|Experience Distillation|Remove the experience summary step|63.56% (sometimes hits context-length limits)|-3.90% |
|Experience Reuse|Only dynamic strategy|66.37%|-1.09% |
|Experience Reuse|Only stable strategy|63.56%|-3.90% |



|Method|Score|
|--------------|--------|
|no method|58.28|
|A-mem|60.52|
|LangMem|62.69|
|ExpG|66.96|