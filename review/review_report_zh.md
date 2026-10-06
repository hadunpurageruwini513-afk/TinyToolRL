# TinyToolRL 审稿报告（Senior Reviewer 视角，中文）

**审稿对象**：`https://github.com/hadunpurageruwini513-afk/TinyToolRL`（public，1 commit，MIT 声明）＋本地工作区全量审计（全部源码、checkpoints、4 模型×3 split 逐样本评测记录、figures、report、CLAUDE.md、git/GitHub 状态）
**审稿日期**：2026-10-07
**审稿方法**：通读全部代码与文档；对 390×4 条逐样本 records 做交叉差分审计；核对 adapter 基座指向、git 追踪集与 GitHub 公开内容一致性；目视三张 figures；交叉验证报告数字与 JSON 记录。

---

## 1. 评审总评与综合评分

**评级：Reject（按顶会论文标准）**；在"套磁工程作品集"语境下为 **Weak Accept（须按第 4 节修改后方可使用）**。

一句话理由：**现象实验与工程控制真实且规范，但论文级核心主张——capability-aware reward 优于 fixed cost——在当前实验矩阵下没有任何实证支持：主表中 "Ours" 与两个 baseline 在全部指标上逐位相同，逐样本审计显示三个 GRPO 模型相对 SFT 的工具决策改变量为 0。**

| 维度 | 得分 | 一句话总评 |
|---|:---:|---|
| 1 立项动机与学术定位 | **7/10** | 问题真实、锚点文献准确且诚实（承认 OTC 已做固定惩罚、把 fixed 降为 baseline）；但 Abstract 把三模型共享的收益归功于 Ours，存在 overclaim |
| 2 实验设计严谨性与控制变量 | **6/10** | 控制变量的纸面设计接近顶会水准（同起点/同数据/同超参/字节级模板一致/π_ref 正确），但 reward 对比在数据分布上结构性退化，且无种子、无置信区间、无训练曲线 |
| 3 实验结果批判性审视 | **3/10** | 主表四模型 TCR 完全相同且精确等于数据集设计比例；Fig2 四组柱完全重合；+2.5pp = 120 题中的 3 题；6.1 节关键分化结论是未验证的推测 |
| 4 工程质量与可复现性 | **8/10** | 模块化、gate 脚本、AST 安全计算器、adapter/参考模型管线全部正确，且逐样本记录留档（使本审计成为可能）；扣分：requirements 未 pin CUDA 源、无训练日志、单 commit、声明 MIT 却无 LICENSE 文件 |
| 5 材料包装与套磁竞争力 | **7/10** | README/report 的结构、公式、表格达到 workshop 初稿水平；但 Fig3 标注重叠、Fig2 自我反驳、"checkpoints released openly" 与事实不符 |

**综合：6.2 / 10**

---

## 2. 主要优势（Key Strengths）

1. **"现象先行"的科研次序正确，Fig1 是全文最诚实的图。** 先冻结 π0、无工具 K=3 采样估 c(x)，再画 c(x) vs TCR 曲线（c=0 → 81.6%，c=1 → 13.4%），图与表显式标注每桶样本量（n=38/7/8/97）。这是"先证明现象存在再谈方法"的正统做法。
   证据：`figures/fig1_observation.png`、`report/report.md` §3.2、`eval_results/SFT_eval_summary.json`（by_capability 与报告数字逐位吻合）。
2. **三处 prompt 字节级一致不是口号，而是被 gate 脚本强制的工程事实。** `common.get_tokenizer()` 统一注入 TRL 训练模板与 qwen3 response schema；`scripts/verify_chat_template.py` 断言 dict-schema 与 function-schema 渲染一致、轨迹 prefix-preserving、parsing 正则与渲染标记兼容。这消除了 tool-use RL 最常见的 prompt-drift 假象来源。
   证据：`common.py:30-42`、`scripts/verify_chat_template.py:57-142`。
3. **控制变量的工程实现经得起逐文件审计。** 三个 GRPO adapter 的 `base_model_name_or_path` 均为 `checkpoints/sft_merged`（即 π0），在 TRL+PEFT 语义下 π_ref = 禁用 adapter 的基座 = π0，理论与工程同时成立；`train_grpo.py` 三组训练仅 reward 列表不同；LoRA/温度/G/beta 全部由同一 CLI 默认值供给，不存在"Ours 一套超参"的空间。
   证据：`checkpoints/grpo_*/adapter_config.json`、`train_grpo.py:198-247`。
4. **工具与解析层的安全性与鲁棒性高于同类学生项目。** 计算器用 AST 白名单求值（限幂指数、限长度、除零/溢出返回 "Error:" 字符串，永不抛异常）；`parsing.py` 剥离 `<|...|>` 特殊 token、整数金标精确比对、支持 multi-turn list 形态 completion。reward 与 eval 共用同一解析模块，训练信号与报告指标测的是同一件事。
   证据：`tools/calculator.py:49-113`、`rewards/parsing.py:18-37,76-78,178-209`。
5. **透明度值得表扬（尽管它暴露了缺陷）。** 全部逐样本 records 入库，使"三个 GRPO 模型工具决策零变化"这一事实可被任何人复算。敢于留档本身就是科研诚信的体现。
   证据：`eval_results/*_records.jsonl`（共 12 个文件，全部 tracked）。

---

## 3. 潜在软肋与致命漏洞（Vulnerabilities & Fatal Flaws）

### 致命漏洞 1：核心主张零证据——主表是一张"无差异表"
主表（`eval_results/comparative_report.md`、README 首屏）中，四个模型的 TCR / Incapable TCR / Capable TCR 在三个 split 上**逐位相同**，且 TCR 精确等于数据集设计比例（33.33% / 50.00% / 20.00%）——即所有模型的工具策略退化为"按 needs_tool 标签的类别规则"。逐样本差分审计：SFT vs 三个 GRPO 的 `called_tool` 差异在 390 个测试样本上**全部为 0**；GRPO-Fixed vs GRPO-Cap 的预测差异为 0/0/1。
**推论：λ=0.1 下 capability-aware reward 相对 fixed cost 的行为差异为零；论文标题级主张（selective tool use）未被任何一个实验观测到。** Fig2（"Selective Tool Use by Problem Capability"）四组柱状图完全重合，等于用一张图自我反驳。

### 致命漏洞 2：奖励对比在数据分布上结构性退化（实验设计层面，比结果更严重）
`needs_tool` 标签与 c(x) 近似完全共线：hard→c≈0（train 中 c=0 占 26.5%），easy/common→c≈1（c=1 占 64.0%）。于是 R_fixed 与 R_cap 只在 c∈{1/3,2/3} 的约 9.5% 训练样本上有差异，且差值仅 λ·(1−c) ≈ 0.033~0.067。叠加 SFT 过拟合（eval loss 0.0147、token acc 99.6%）导致 T=1.0 rollout 近似确定性、GRPO 组内方差≈0、advantage 退化——**本实验从设计上就没有统计功效区分 Fixed 与 Cap**。要在 λ 小的区间看出分化，数据必须包含"标签说需要工具但模型其实会做 / 标签说简单但模型其实不会"的分歧样本；当前生成器把难度与类别绑死，恰好排除了这种样本。

### 致命漏洞 3：Overclaiming 与事实不符的声明（套磁场景下的诚信风险）
- Abstract："Our capability-aware policy ... boosts OOD accuracy from 96.67% to 99.17%" —— 该增益三个 GRPO 模型完全共享（99.17/99.17/99.17），归功于 Cap 属归因错误。
- Abstract/README："All code, datasets, **checkpoints**, and reproduction scripts are released openly" —— `checkpoints/` 在 `.gitignore` 中，GitHub 上不存在任何权重，也无 HF 链接。声明为假。
- README 声明 MIT License，但仓库**没有 LICENSE 文件**。
- `report.md` §6.1 将"λ∈0.2~0.5 时 Fixed 崩溃而 Cap 保护困难题"写成"important theoretical insight"，但**该实验从未运行**（CLAUDE.md 待办 I 亦承认仅"纳入报告讨论"）；计划的 λ 扫描、K=1 vs 3、counterfactual cue 测试均未执行，而 CLAUDE.md 以"[x] 完成"记档，属于记录与事实不符。
- 数字自相矛盾：CLAUDE.md 记 base 摸底"无工具 Acc 4.67% / 有工具 4.67%、TCR 0.00%"，入库 JSON 为 0.00% / 5.33%、TCR 12.00%；CLAUDE.md 记 SFT eval Acc 94.67%，comparative_report 为 94.00%。导师复核必踩。

### 主要软肋 4：统计功效与可验证性不足
- +2.50pp = 120 题中翻转 3 题；无任何 CI/显著性/多种子。n=120、p≈0.97 的二项噪声约 ±1.5~2pp，3 题翻转不构成证据。
- 现象实验中间桶 n=7、n=8（57.1% = 4 题、25.0% = 2 题），Fig1 曲线中段由个位数样本支撑。
- `report_to="none"` + `save_strategy="no"` → 无训练 reward/penalty 曲线留档，审稿人无法验证惩罚项是否真的产生过梯度信号（也无法排除 advantage 退化猜想）。
- 单 commit、作者占位邮箱 `tinytoolrl@research.local`、缺少项目约定的 Co-Authored-By 行；工作区残留 `TinyToolRL_Deliverable/` 全量副本、`tmp_sft_check/`、`__pycache__/`（未入库但会随 zip 交付扩散）。

---

## 4. 即刻修改与防坑指南（Actionable Improvements）

**优先级 P0（不做则任何投递都有诚信风险）**
1. **重写主张口径**：Abstract/README/Fig3 说明文字改为"λ=0.1 下三种 reward 的工具策略无统计可辨差异；收益为 GRPO 共享的泛化提升"。把项目定位改为"受控研究 + 诚实的零差异结果 + 现象刻画"，这本身是可写的贡献（negative/controlled result），比 overclaim 更打动导师。
2. **删除或兑现虚假声明**：要么把 checkpoints 上传 HF 并给出链接，要么删掉 "checkpoints released openly"；补 LICENSE 文件（或删 MIT 徽章）；统一 base/SFT 数字（以入库 JSON 为准重写 CLAUDE.md C/D 条目）。
3. **修正 CLAUDE.md 待办 I 的"[x]"记档**：λ 扫描、K 消融、cue 测试未做就是未做，改回 [ ] 并标注"未执行原因"。

**优先级 P1（让核心主张第一次有可能被检验）**
4. **跑 λ∈{0.3, 0.5} 的 Fixed 与 Cap 两组**（其余超参不变）。这是唯一可能让两奖励分化的区间；若 Fixed 在 hard 题 Acc 崩溃而 Cap 保持，主张成立；若都不崩或都崩，如实报告。每配置 ≥2 seeds。
5. **修数据功效缺陷**：在 train/eval 中加入"标签-能力分歧"样本（例如 2 位×1 位、3 位÷小因子等中间难度，使 c∈{1/3,2/3} 桶占比 ≥25%），让 R_fixed 与 R_cap 在足够多的样本上真正不同。
6. **统计规范化**：所有 Acc/TCR 报 Wilson 95% CI；模型间逐样本翻转用 McNemar 检验；主表脚注标明每格 n。
7. **留档训练动态**：`report_to="csv"` 或自定义 callback 每 5 步记录 reward/penalty 均值与组内 reward 方差（直接验证 advantage 退化假设），入库 `logs/`。

**优先级 P2（包装与工程收尾）**
8. Fig2 要么换成 λ 扫描后的版本，要么改标题为 "Tool policy is unchanged at λ=0.1 (null result)"——诚实的 null 图比四组相同的柱子更有科研味；Fig3 修复标注重叠、并把三个重合点合并为单点加注 "3 GRPO variants coincide"。
9. `requirements.txt` 改为精确 pin（含 torch 2.11.0+cu128 与 `--extra-index-url https://download.pytorch.org/whl/cu128` 说明行），补 HF_ENDPOINT 镜像提示；Quickstart 增加"下载 checkpoints"一步。
10. git 卫生：用真实 GitHub 身份重设 author、按里程碑拆 4~6 个 commit（或至少补 CHANGELOG.md）、删除 `TinyToolRL_Deliverable/`、`tmp_sft_check/` 与 zip，或将其移出仓库目录。
11. report §6.1 把未验证的 λ>0.1 论述显式标为 *hypothesis (untested)*，并补一句可证伪条件；Limitations 增加"单 seed、单模型规模、c(x) 离线估计"。

**给申请人的最后一句话**：这个项目最值钱的部分不是 reward 公式，而是"现象实验 + 全链路受控工程 + 逐样本留档"。把零差异结果当作发现来写、把 λ 扫描补上，它就从一份会被一眼看穿的 overclaim，变成一份展示完整科研判断力的作品——后者才是套磁信里真正有杀伤力的东西。

---

*本报告所有数字均可由仓库内 `eval_results/*_records.jsonl` 与 `*_summary.json` 复算；审计脚本逻辑见正文引用的文件与行号。*

---

## 附录 A：修改回应（Revision Response，2026-10-07 起）

针对第 3、4 节问题的逐条处理状态（✔ 已修 / ◐ 进行中 / ✖ 有意保留并说明）：

| # | 审稿意见 | 状态 | 处理 |
|---|---|:---:|---|
| P0-1 | Abstract/README 把共享收益归功于 Ours | ✔ | report Abstract、README 主表下 honesty note、§5.2 新增 takeaway 4（零结果显式报告） |
| P0-2 | "checkpoints released openly" 为假；MIT 无 LICENSE | ✔ | 声明改为"代码/数据/逐样本记录/复现脚本开源，adapter 可由脚本复现"；补 `LICENSE` |
| P0-3 | CLAUDE.md 待办 I 误记 [x]；base/SFT 数字自相矛盾 | ✔ | 待办 I 改回 [ ] 并注明真实状态；C②/D④ 数字统一为入库 JSON 口径 |
| P1-4 | λ 分化实验缺失（§6.1 推测无证据） | ◐ | `scripts/run_lambda_sweep.sh`：Fixed/Cap × λ∈{0.3,0.5} 四组同超参重训（带内存守卫串行）；结果入 §6.3 与 Fig4，分化与否均如实报告 |
| P1-5 | needs_tool 与 c(x) 共线导致 Fixed≡Cap 结构性无功效 | ✖ 保留 | 重造数据+重训 SFT 全链路超出本次修订算力预算；已写入 report §7 Limitations 与 future work，并给出具体补救设计（标签-能力分歧样本、中间难度桶 ≥25%） |
| P1-6 | 无 CI/显著性/种子 | ◐ | `scripts/analyze_stats.py` → `eval_results/stats_appendix.md`（Wilson CI + McNemar 精确检验）；report §5 新增不确定性段；多种子列入后续 |
| P1-7 | 无训练曲线留档 | ✔ | `train_grpo.py` 训练后写 `logs/grpo_{mode}_lam{λ}_{tag}_log.json`（per-step 各 reward 均值/组统计），λ 扫描起生效 |
| P2-8 | Fig2 自我反驳、Fig3 标注重叠 | ✔ | Fig2 改诚实标题"controlled null result"+ Wilson 误差线；Fig3 合并重合点、标签移出 CI 须区；新增 Fig4 λ 扫描图 |
| P2-9 | requirements 未 pin CUDA 源 | ✔ | 精确 pin + cu128 index 说明 + HF 镜像提示 |
| P2-10 | git 卫生（占位作者、单 commit、Deliverable 副本） | ◐ | 本地副本/临时目录/重复 zip 已删；git 身份改为 GitHub noreply；修订以独立 commit 提交。已推送的单 commit 历史不做 rewrite（避免破坏公开仓库引用），以本附录+CHANGELOG 式 commit 信息补足脉络 |
| — | K=1vs3 消融、counterfactual cue 测试 | ✖ 保留 | 未做；report Limitations 明列 |

**回应总原则**：凡实验未做的，一律改记为未做；凡数字有出入的，一律以入库逐样本记录为准；凡主张超出证据的，一律降级为 hypothesis 或 null result。λ 扫描完成后本附录与 report §6.3 同步更新实测结论。
