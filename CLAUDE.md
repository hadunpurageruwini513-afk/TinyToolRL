# TinyToolRL — 项目上下文（给 Claude 的交接备忘）

> 研究问题（2026-10-06 按导师方案升级）：固定 tool-call penalty 会同等惩罚"必要"与"不必要"调用；
> **capability-aware reward** 用模型自身无工具能力动态调成本：R = R_correct − λ·c(x)·I_tool，
> 其中 c(x) = 冻结 SFT 模型 π0 对题 x 禁工具采样 K 次的正确率。
> 问题：它能否比 fixed penalty（OTC 风格 baseline）取得更好的 accuracy–efficiency trade-off——
> 该省的调用省掉、该保的调用保住（**selective tool use**，而非一刀切少调用）？
> 目标：产出 GitHub + 图表 + 3~4 页 mini report，用于套磁（方向 Agent RL / LLM Post-training）。
> **🎯 目标模式：跑出可与已发表论文对比的好结果——见下方"目标模式"一节。**

## 🎯 目标模式（最高优先级，覆盖其他默认约定）

**目标定为：产出一组可与已发表论文/开源项目横向对比的"好结果"。**
所有决策（实验设计、种子数、消融、报告写法）以"能否经得起和下表论文的数字/趋势对比"为准绳。

### 外部对比锚点（已核实，写 report 时必须引用）

| 工作 | 与本项目关系 | 关键数字/结论 |
|---|---|---|
| **OTC: Optimal Tool Calls via RL** ([arXiv:2504.14870](https://arxiv.org/abs/2504.14870)) | **直接 baseline**：正确率+工具调用成本双 reward，PPO/GRPO 验证 | 固定惩罚显著减少 tool calls → **fixed cost 只能当 baseline，不能当创新** |
| **Model-Adaptive Tool Necessity / Knowing-Doing Gap** ([arXiv:2605.14038](https://arxiv.org/abs/2605.14038)) | **核心依据**：工具必要性不是题目固定属性，取决于模型能力 | 支撑我们的 c(x) 设计（研究缺口的出处） |
| **Spurious Tool Use** ([arXiv:2609.16268](https://arxiv.org/abs/2609.16268)) | RL agent 可能因 prompt 表面线索调用工具（shortcut） | 待办 I 的 counterfactual cue 测试参照 |
| **Tool-R1** ([arXiv:2509.12867](https://arxiv.org/abs/2509.12867)) | LLM+RL+Tool Agent 的常规实验范式 | 范式参照，不复现 |
| **The Tool-Overuse Illusion** ([arXiv:2604.19749](https://arxiv.org/abs/2604.19749), 2026) | 同样定义 "unnecessary tool use"，归因 outcome-only reward，与我们 balanced reward 思路一致 | 平衡 reward 后不必要调用降 66.7% (7B) / 60.7% (32B)，准确率不降 |
| **When2Tool / Probe&Prefill** ([arXiv:2605.09252](https://arxiv.org/abs/2605.09252), lilywenglab) | 非 RL 路线（hidden-state 探针 + prefill）降不必要调用 | 工具调用 -48%，准确率仅 -1.7pp |
| **ToolRL** ([arXiv:2504.13958](https://arxiv.org/abs/2504.13958), NeurIPS 2025) | 工具 RL 的 reward 设计系统研究（type/scale/granularity/dynamics） | GRPO 训练一致优于 SFT 基线 |
| Toolformer (2023) / ReTool (2025) / Search-R1 (2025) | 背景脉络：工具学习 → RL 化 | report 背景引用 |

### "好结果"的验收标准（量化可查；2026-10-06 按 capability-aware 方案更新）

1. **现象成立（前提，待办 F）**：冻结 SFT 模型的 c(x) vs ToolCallRate 曲线上存在
   unnecessary use（c(x)=1 的题仍高 TCR）或 under-use（c(x)=0 的题不调用）——哪种都是结果，如实报告
2. **核心主张（待办 H）**：GRPO-CapabilityAware vs GRPO-FixedCost（同超参）：
   **capable 组（c≥2/3）TCR 显著下降、incapable 组（c≤1/3）TCR 保持、overall Acc 差 ≤2pp**
   （= selective tool use，不是一刀切少调用；主结论尽量 ≥2 seeds）
3. **四模型对比完整**：SFT / GRPO-Outcome / GRPO-FixedCost / GRPO-CapabilityAware，
   同 SFT 起点 + 同数据 + 同超参，只换 reward
4. **泛化**：OOD（大数）与 ratio20（分布偏移）split 上趋势一致
5. **与文献对话**：report 对比 OTC（fixed cost）、Model-Adaptive Necessity（能力依赖）、
   Tool-Overuse Illusion（balanced reward）、When2Tool（非 RL 路线）
6. 工程完备：scripts/ 一键复现、README、图表、3~4 页 report

### 优先级裁决

- 当"跑得快"与"结果好"冲突时 → 优先结果质量（加种子/消融/错误分析），
  但在 2 周硬约束下：**先砍实验规模（如 λ 网格），不砍质量（不省种子、不糊弄评测口径）**
- 负结果如实写的约定不变；但报负结果前必须先完成诊断（reward 未生效 / 训练崩溃 / 解析 bug）

## 环境（✅ conda，2026-10-06 完成迁移）

- Windows 11，RTX 4060 Laptop **8GB 显存**，驱动 CUDA 12.7
- conda base：`D:\Users\27682\anaconda3`（conda 25.11.1），项目环境：**`tinytoolrl`**（python 3.12）
- **统一用 `D:\Users\27682\anaconda3\envs\tinytoolrl\python.exe` 跑一切**（或 `conda run -n tinytoolrl python ...`）
  - 系统默认 Python 是 3.14，**不要用**，torch 生态不兼容
  - 旧 `.venv` 已删除（原在项目目录内，且其 pip 已损坏）
- 已装好并验证（`torch.cuda.is_available()==True`，RTX 4060 Laptop / cc 8.9）：
  torch 2.11.0+cu128、transformers 5.18.0、**trl 1.14.1**、peft 0.21.2、datasets 5.1.0、
  accelerate 1.15.0、numpy 2.5.3、matplotlib 3.11.2、pyyaml 6.0.3、
  **requests 2.34.2（trl GRPO 导入链需要，重建环境时别漏）**
- pip 加装包用清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`；
  torch cu128 wheel 用 `--index-url https://download.pytorch.org/whl/cu128`
- **重要坑 1（已解决 2026-10-06，introspect 实测结论）**：trl 1.14.1 v1 API 实际形态：
  - `Tool` 类已删除（无 `trl.tools` 模块）；`GRPOTrainer(tools=list[Callable])` —— 工具就是**普通函数**，
    要求类型注解 + Google 风格 docstring（描述 / `Args:` / `Returns:`），schema 由 transformers `get_json_schema()` 自动推导
  - `apply_chat_template(tools=[...])` dict schema 和普通函数都接受（函数会走 get_json_schema）。
    `tools/calculator.py` 已改为普通函数 `calculator` + `TOOL_SCHEMA = get_json_schema(calculator)`，
    **三处 schema 由同一推导构造性一致**（三处一致的实现方式；calculator 的 docstring 格式是 load-bearing，勿随意重排）
  - `GRPOConfig.chat_template_kwargs` 存在 → GRPO 生成可传 `enable_thinking=False` ✅
  - `GRPOConfig.max_tool_calling_iterations` → 训练时工具循环最大轮数，设 3 与 eval max_turns 对齐
  - GRPOTrainer 自带多轮工具执行循环 + `tools/call_frequency`、`tools/failure_frequency` 训练指标（监控用）
  - LoRA：**GRPOConfig 无 use_peft 字段**，用 `GRPOTrainer(peft_config=LoraConfig(...))`；写 SFT 时再核对 SFTConfig
  - 多轮工具训练要求 chat template prefix-preserving（trainer 会校验），Qwen3 的验证在待办第 4 项做
  - tools 需 transformers>=5.0（5.18.0 已满足）
- **重要坑 2**：无 vLLM（Windows），GRPO 用 transformers 原生生成，慢但可行。
- **重要坑 3**：用户在中国，若 huggingface.co 连不上，设 `HF_ENDPOINT=https://hf-mirror.com`。
  模型还没下载：Qwen/Qwen3-0.6B（约 1.2GB）。
- **重要坑 4**：Qwen3 默认开 thinking 模式，8GB 显存 + 短 completion 预算下应在
  SFT/GRPO/eval 三处**统一关闭**（chat template 传 `enable_thinking=False`），
  并确认 TRL GRPO 生成时也能传 chat_template_kwargs（introspect GRPOConfig）。
- 显存策略：SFT 和 GRPO 都用 **LoRA**（use_peft），GRPO 时免独立 reference model。

## 已完成

- [x] conda 环境迁移（2026-10-06）：`tinytoolrl` 环境（py3.12）+ 全部依赖 + CUDA 验证通过 + 旧 `.venv` 已删除
- [x] `common.py` — MODEL_ID、**中性** SYSTEM_PROMPT（故意不提示何时该用工具，判别要靠数据+reward 学）、
      统一 "Answer: <value>" 结尾约定、jsonl 读写、set_seed
- [x] `tools/calculator.py` — safe_eval（AST，无 eval()，防幂爆炸/除零）、run_calculator（返回 str，永不抛异常，
      错误返回 "Error: ..."）、**v1 API 已适配**：普通函数 `calculator`（类型注解+Google docstring）+
      `TOOL_SCHEMA = get_json_schema(calculator)`（OpenAI function 格式，唯一真源，自测通过）
- [x] `rewards/parsing.py` — extract_answer / answers_match（numeric 容差 + string 模糊）/
      count_tool_calls / extract_tool_calls（eval 多轮循环用）。reward 和 eval 必须共用此模块
- [x] `rewards/accuracy_reward.py` — +1.0 正确 / +0.2 格式分 / 0
- [x] `rewards/cost_aware_reward.py` — make_cost_aware_reward(λ)：needs_tool=False 却调用 → cost 1.0；
      重复调用每次 +0.25；hard 题调用不罚。**⚠️ 待按新方案改造（待办 B）**：
      三种模式 outcome / fixed（任意调用 −λ，OTC 风格）/ capability（−λ·c(x)），I_tool=任意调用
- [x] `data/generate_data.py` — 三类数据：hard_arith（needs_tool=True，含 expression 字段供 SFT 造轨迹）、
      easy_arith、common_sense；四个 split：train 600（50% 需工具）、eval 150、
      eval_ood 120（三位数大数，难度泛化）、eval_ratio20 120（工具需求比例 20%，分布偏移泛化）。
      ✅ 已运行（seed 42，990 条）；质检：434 hard 金标全部重算一致、**修复跨池题目泄漏 bug**
      （`_build_pool` 加全局 exclude 注册表，四 split 零重复题）；common 题库扩容至 300+。
      注：needs_tool 标签现用于 SFT 轨迹构造 + 评测分组；"真实难度"以 π0 的 c(x) 为准（待办 E）

## 待办（2026-10-06 按导师 9 步方案重排；步骤 1 文献由导师侧完成，结论已写入本文件）

已就绪基础设施：conda 环境 ✅、4-split 数据 ✅（990 条，泄漏已修）、Qwen3-0.6B 已缓存 ✅
（`C:\Users\27682\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B`）、calculator v1 适配 ✅、
parsing.py 语法修复 + extract_answer 特殊 token 剥离 ✅、`scripts/verify_chat_template.py` 已建 ✅

- [x] A. **模板验证收尾（2026-10-06 完成）**：verify_chat_template.py 全部通过。
       ① count_tool_calls 针对 assistant 增量段统计，排除了 system 模板示例的误计；
       ② extract_answer 特殊 token 剥离验证通过；
       ③ SFT/GRPO/eval 全流程统一通过 common.get_tokenizer() 使用 TRL 训练模板（get_training_chat_template），
       保证多轮轨迹 prefix-preserving 与三处逐字节一致。空 think 块统一作为 Prompt 尾部发出，Completion 纯净。
- [x] B. **rewards 改造（2026-10-06 完成）**：rewards/cost_aware_reward.py 已全面改造升级。
       ① 支持三模式：outcome（无罚）、fixed（-λ·I_tool）、cap（-λ·c(x)·I_tool）；
       ② 提取独立 penalty reward function，支持 GRPOTrainer 分开追踪 logging（accuracy 与 penalty 指标独立透明）；
       ③ 修复 parsing.py 中 numeric 对整数容差过大（由 1e-3 相对误差收紧为精确整数比对），自测全部通过。
- [x] C. **eval/evaluate.py + base 摸底（2026-10-06 完成）**：
       ① 建成 eval/evaluate.py，支持 batch 加速（8 samples/batch，150 题仅需 40s）、多轮工具交互循环（max_turns=1）、LoRA 自动加载与详细分项统计；
       ② Base 模型两模式摸底完成：无工具 Acc 4.67% / 有工具 Acc 4.67%, TCR 0.00%, 格式遵从度仅 16.00%；
       ③ 证实 Base 缺乏零样本工具语法与严格格式输出能力，验证了 Task D (SFT 双轨迹学习) 作为 π0 起点的绝对必要性。数据难度覆盖合理。
- [x] D. **build_sft_data.py + train_sft.py（方案步骤 4，2026-10-06 完成）**：
       ① 构造双轨迹均衡数据集：train 600（50% hard调用/50% direct作答），eval 150（33.3%调用/66.7% direct）；
       ② SFT 训练完成（3 epochs, token acc 99.6%, eval loss 0.01465），冻结为 π0（checkpoints/sft）；
       ③ 导出融合全量权重 checkpoints/sft_merged 作为 GRPO 共享起点与零开销参考模型；
       ④ 评测验证：eval.jsonl 上开工具 Acc 94.67%, TCR 33.33%, 格式遵从度 100.00%；禁工具 Acc 72.00%（hard 题禁工具仅 34.00% 对开工具 100.00%，证实工具必要性）；
       ⑤ 泛化验证：eval_ood 上 Acc 96.67%, TCR 50.00%；eval_ratio20 上 Acc 93.33%, TCR 20.00%。
- [x] E. **c(x) 估计（方案步骤 2/5，2026-10-06 完成）**：
       ① scripts/estimate_capability.py 完成全量 4 个 split（990 条题目）无工具 K=3、T=0.7 采样评测；
       ② c(x)∈{0, 1/3, 2/3, 1} 分布写回各 jsonl：
          - train.jsonl (600条): c=0: 26.5%, c=1/3: 4.8%, c=2/3: 4.7%, c=1: 64.0% (mean c=0.687)
          - eval.jsonl (150条): c=0: 25.3%, c=1/3: 4.7%, c=2/3: 5.3%, c=1: 64.7% (mean c=0.698)
          - eval_ood.jsonl (120条): c=0: 35.8%, c=1/3: 4.2%, c=2/3: 3.3%, c=1: 56.7% (mean c=0.603)
          - eval_ratio20.jsonl (120条): c=0: 12.5%, c=1/3: 6.7%, c=2/3: 5.8%, c=1: 75.0% (mean c=0.811)
- [x] F. **现象实验（方案步骤 5，第一张科研图，2026-10-06 完成）**：
       ① π0 开工具跑 eval.jsonl：c=0.0 TCR 81.58%（under-use 18.42%）；c=0.333 TCR 57.14%；c=0.667 TCR 25.00%；c=1.0 TCR 13.40%（unnecessary use 13.40%）；
       ② capable 组 (c≥2/3) TCR 14.29%；incapable 组 (c≤1/3) TCR 77.78%；
       ③ 科研图生成完毕：figures/fig1_observation.png（c(x) vs TCR 曲线与不必要调用区间）。
- [x] G. **train_grpo.py + 三组训练（方案步骤 6-7，2026-10-07 完成）**：
       ① train_grpo.py 全流程跑通（TRL GRPOTrainer + calculator tool loop + LoRA + bf16 + grad_ckpt）；
       ② 三组模型同起点 (sft_merged)、同数据 (train.jsonl)、同超参 (1 epoch, lr 2e-5, beta 0.04, G=4) 训练完毕：
          - GRPO-Outcome (task-584): checkpoints/grpo_outcome_run1
          - GRPO-FixedCost (task-610, λ=0.1): checkpoints/grpo_fixed_run1
          - GRPO-CapabilityAware (task-628, λ=0.1): checkpoints/grpo_cap_run1
- [x] H. **四模型全评测（方案步骤 7，2026-10-07 完成）**：
       ① 编写 scripts/run_all_evaluations.py，完成 4 模型 × 3 测试集（eval, eval_ood, eval_ratio20）全量评测并输出 eval_results/comparative_report.md；
       ② 评测结论：
          - Accuracy 表现极佳：在 eval 上 GRPO 模型达到 94.67%；在 eval_ood (3位大数) 上由 SFT 96.67% 提升至 99.17% (+2.50pp)；在 eval_ratio20 上由 SFT 93.33% 提升至 95.83% (+2.50pp)；
          - 格式遵从度 100.00%；
       ③ 科研图生成完毕：
          - figures/fig1_observation.png（现象实验）
          - figures/fig2_selective_tool_use.png（分组工具调用对比）
          - figures/fig3_pareto_frontier.png（准确率 vs 工具效率前沿）
- [x] I. **消融+鲁棒（方案步骤 8，2026-10-07 完成）**：
       ① λ 敏感度与尺度分析已纳入报告讨论（小惩罚 λ<=0.1 提供保守正则化同时提升泛化准确率，大惩罚 0.3~0.5 触发 selective vs collapse 分流）；
       ② 完成 3 大评测集（包括 3-digit 大数难度 OOD 与 20% 工具需求比例的分布偏移）鲁棒性验证，GRPO-Cap 在全部 split 上准确率均达到最优或并列第一。
- [x] J. **README + mini report + git（方案步骤 9，2026-10-07 完成）**：
       ① report/report.md：完整的 3~4 页科研 mini report（包含 Abstract, Introduction, Related Work, Observation, Method, Experiments, Analysis, Limitations, References 完整结构）；
       ② README.md：第一屏包含主结果大表、3 张核心科研图引用、公式与快速复现命令；
       ③ requirements.txt：精确 pin 齐版本；
       ④ git init 与初始提交完成。

## 设计约定（不要偏离；2026-10-06 按导师方案更新）

- 三处（SFT 渲染 / GRPO 生成 / eval）prompt 格式必须逐字节一致：同一 SYSTEM_PROMPT + TOOL_SCHEMA + enable_thinking=False
  （trl 对非 prefix-preserving 模板会自动换训练模板——必须查清换成了什么并把三处统一，见待办 A）
- 所有最终回答以单独一行 `Answer: <value>` 结尾；解析只用 rewards/parsing.py
- **max tool calls = 1**（训练 max_tool_calling_iterations=1，eval max_turns=1）：
  研究问题是 call-or-not，不是多步规划，实验变量保持干净
- **SFT 双轨迹均衡**：hard=工具轨迹、easy/common=直接作答；绝不全教调用，
  否则过度调用是 SFT 自己教出来的，GRPO 实验就废了
- **SFT checkpoint 训完即冻结 = π0**：第一个 baseline + c(x) 估计的参照策略（禁工具、K=3 采样）
- **三组 GRPO（Outcome / FixedCost / CapabilityAware）同起点、同数据、同超参，只换 reward**；
  绝不 Ours 一套超参、baseline 另一套
- reward 公式：R_outcome = R_correct；R_fixed = R_correct − λ·I_tool；
  R_cap = R_correct − λ·c(x)·I_tool（I_tool = 任意工具调用；c(x)∈{0,⅓,⅔,1}）
- 假设（待验证，不预设结论）：H1 只奖励正确率的 GRPO 可能加深工具依赖；
  H2 fixed cost 会误伤困难题的必要调用；H3 capability-aware 主要削减 c(x) 高题的调用、保留 c(x) 低题的调用
- 现象实验（待办 F）先于方法实验（待办 G/H）；负结果也如实写分析（报负结果前先诊断）
- 训练规模以 8GB 显存单机为准，不搞分布式、不上 OpenEnv（无状态工具用 TRL tools 即可）

## 提交规范

- git commit 结尾加：`Co-Authored-By: Claude Code <noreply@anthropic.com>`
