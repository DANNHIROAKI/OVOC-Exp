# OpenViking × OpenClaw 插件性能测试实验计划（正式版·三组细粒度评估）

> 本文件替代旧版四组方案。若与旧版冲突，以本版为准。  
> 本版相对于旧版的三项强制改动如下：  
> 1. 移除 `No-OV / no-memory`（`plugins.slots.memory = none` 且 `plugins.slots.contextEngine = legacy`）组，只保留剩余三组；  
> 2. 在总准确率与总 token 之外，新增 **sample 级**与 **task 级**记录，要求对三组设置下的 **1540 个任务逐题记录** 是否通过、消耗多少 token、耗时多久；  
> 3. 版本强制固定为 **OpenClaw 2026.4.14** 与 **OpenViking 0.3.8**，正式实验必须遵守，不得替换。

本计划沿用你清洗后的 LoCoMo10 长程对话评测集（去除无真值 `category 5` 后共 **1540 条 case**），并继续要求正式评测统一通过 **OpenClaw Gateway `/v1/responses`** 进入系统，不允许绕过 Gateway 直接写 OpenViking。安装入口统一使用 `openclaw-openviking-doubao` 仓库，但正式实验一律以本计划锁定的版本号、配置与运行时校验结果为准。

---

## 1. 实验目标

本实验的目标是，在长程对话记忆场景下，系统评估 OpenViking 的 OpenClaw 插件在三种实际可用配置中的表现，并在**总量指标**之外补齐**细粒度行为记录**。具体目标如下：

1. 评估两种 OpenViking 配置相对于 OpenClaw stock 基线的任务完成率、总输入 token 成本与总耗时；
2. 评估在 OpenViking 已启用时，是否应继续保留 OpenClaw 原生 `memory-core`；
3. 以 **1540 条任务逐题明细** 的方式，分析不同配置下哪些任务成功、失败、花费更多 token、耗时更长；
4. 形成可复现、可审计、可回溯的正式 benchmark 方案。

对应地，本版实验回答四个问题：

- **RQ1**：相对于 `No-OV / stock`，`OV / no-memory` 在完成率、总 token 成本和总耗时上表现如何？
- **RQ2**：相对于 `No-OV / stock`，`OV / stock` 在完成率、总 token 成本和总耗时上表现如何？
- **RQ3**：在 OpenViking 已启用时，保留 `memory-core`（`OV / stock`）相对于关闭 `memory-core`（`OV / no-memory`）有何收益或代价？
- **RQ4**：在任务级别、类别级别与 conversation sample 级别上，三组配置分别在哪些 case 上胜出、失利、成本更高或耗时更长？

> 说明：由于本版删除了 `No-OV / no-memory`，因此**不再构成完整的 2×2 析因设计**。旧版中的交互项分析、差分中的差分分析与 “OV × memory-core 完整交互效应” 在本版中**不再作为正式结论**。

---

## 2. 锁定项与不变项

### 2.1 三组正式实验配置（唯一允许进入主表的组）

除下表以外，其他实现细节可按本计划补充；但**下表三组配置本身不得再修改**。

| 组 ID | 组别           | `plugins.slots.memory` | `plugins.slots.contextEngine` | `plugins.entries.openviking.enabled` | `plugins.deny`   | 任务完成率 | 成本：输入 token（总计） |
| ----- | -------------- | ---------------------- | ----------------------------- | ------------------------------------ | ---------------- | ---------- | ------------------------ |
| G1    | OV / no-memory | `none`                 | `openviking`                  | `true`                               | `[]`             |            |                          |
| G2    | No-OV / stock  | `memory-core`          | `legacy`                      | `false`                              | `["openviking"]` |            |                          |
| G3    | OV / stock     | `memory-core`          | `openviking`                  | `true`                               | `[]`             |            |                          |

补充说明：

- 旧版中的 `No-OV / no-memory` 组（`memory = none`，`contextEngine = legacy`）**正式移除**；
- 正式论文主表、附录表、日志归档、原始结果目录中，均只允许出现 `G1 / G2 / G3` 三组；
- 后续分析中的所有 pairwise comparison 也都只围绕这三组展开。

### 2.2 统一冻结的其他条件

以下条件在三组之间必须完全一致：

- 数据集：使用你清洗后的 `OpenViking-LoCoMo10`，即去除无真值 `category 5` 后的 **1540 条 case**；
- OpenClaw 版本：**2026.4.14**；
- OpenViking 版本：**0.3.8**；
- 生成模型：**seed-2.0-code**；
- 安装入口：`openclaw-openviking-doubao` 仓库，正式运行时记录其 commit SHA；
- 同一主机、同一网络环境、同一 provider 区域、同一 agent 配置、同一系统提示词、同一工具白名单；
- 除锁定表中的插件相关字段外，其余 OpenClaw 配置保持一致；
- API Key 只通过环境变量注入，不写入论文、日志、配置快照或公开仓库；
- judge 模型与 judge prompt 在三组之间保持一致。

### 2.3 运行时版本校验是强制项

由于本版把 OpenClaw 与 OpenViking 版本提升为**硬约束**，正式实验必须同时满足以下条件：

1. 安装前记录 `openclaw-openviking-doubao` 的 commit SHA；
2. 安装后执行运行时版本检查，例如：
   - `openclaw --version`
   - `openviking --version` 或等价的 Python / CLI 版本查询
3. 若实际安装版本不是 **OpenClaw 2026.4.14** 与 **OpenViking 0.3.8**，则该环境**不得进入正式 benchmark**；
4. 正式实验以 **运行时实际版本** 为准，不能只依赖 README、lockfile 或脚本注释中的文字。

---

## 3. 数据集与评测对象

### 3.1 正式评测集

本实验只使用清洗后的 `OpenViking-LoCoMo10` 作为正式评测集。该评测集沿用 LoCoMo 的 conversation sample 组织方式：每个 sample 含多段 session 与多个 QA，去除无真值 `category 5` 后，共 **1540 条 case**。

正式数据口径如下：

- 正式评测只使用该 1540-case 清洗集；
- 不在运行期额外手工删题；
- 保留 `sample_id`、`category`、`evidence` 等原始字段；
- 若原始数据缺少稳定唯一键，则在预处理阶段一次性生成并冻结 `case_uid`；
- `case_uid` 生成后不得变更，否则不同 rerun 之间无法对齐逐题结果。

### 3.2 评测单位与统计单位

本实验有两个层级的评测对象：

1. **任务层级（task / case）**：最终需要记录 **1540 个 case** 在三组下的逐题结果；
2. **对话样本层级（conversation sample）**：统计检验与置信区间的更合理聚类单位。

因此：

- 报表层面要输出 1540-case 总体结果；
- 但统计显著性与置信区间原则上按 `sample_id` 做 cluster / paired bootstrap；
- 不把 1540 个 QA 当成完全独立同分布样本来做主检验。

### 3.3 运行前数据校验

正式实验开始前，先执行一次固定的数据校验脚本，并把结果写入 `manifest.json`：

- `总 QA 数 = 1540`
- `category 5 = 0`
- `sample_id` 完整无缺失
- `case_uid` 全局唯一
- 各 `sample_id` 的题量统计固定保存

任何一项校验不通过，则不得启动正式 benchmark。

---

## 4. 工具链、安装路径与必须补齐的评测改造

### 4.1 安装与启动路径

建议继续使用 `openclaw-openviking-doubao` 仓库作为统一安装入口，并固定其 commit SHA。正式实验环境的启动、配置与日志路径，统一以该仓库 bootstrap 后的结果为基线；但版本锁定、运行时校验与正式配置仍以本计划为准。

### 4.2 正式评测必须走 OpenClaw Gateway

三组正式评测都必须统一通过 **OpenClaw Gateway `/v1/responses`** 发起请求，不允许在 OV 组使用任何绕过 OpenClaw 生命周期的捷径，例如：

- 直接调用 `ov add-memory`
- 直接走 OpenViking 独立 API 完成 ingest / query
- 使用评测脚本内部的 `--viking` 直写分支替代 Gateway 路径

原因很简单：本实验要比较的是**OpenClaw 插件形态下**的三组配置，而不是“OpenViking 独立服务”的能力上限。只要绕过 Gateway，就无法与 No-OV 组构成同口径对照。

### 4.3 `openclaw-eval` 必须补齐的四类正式实验级改造

#### 4.3.1 参数化 OpenClaw 工作区路径

评测脚本不得把 session 文件路径硬编码为单一默认目录。正式实验必须增加类似 `--openclaw-home` / `--workdir` 的参数，使以下路径都从当前临时工作区解析：

- OpenClaw home
- 当前 agent 目录
- session 文件目录
- reset session 时实际重命名的 `.jsonl` 文件路径

否则一旦使用多快照、多工作区或批量 rerun，就可能 reset 到错误实例，导致污染或错误归档。

#### 4.3.2 补齐系统级 token 统计

正式实验中的“成本：输入 token（总计）”必须覆盖系统真实消耗，而不只是一层 Gateway usage。正式要求如下：

- **Gateway usage**：记录 `/v1/responses` 返回中的 `input_tokens / output_tokens / total_tokens`；
- **OV 内部 usage**：OV 组额外记录 OpenViking 内部 VLM / LLM / Embedding 子调用的 usage；
- **分层存储**：至少区分
  - ingest 阶段 usage
  - QA 阶段 usage
  - OpenViking 内部 usage
- **最终主表**：主表中的 “输入 token（总计）” 必须是三者合并后的系统总量；
- judge 阶段 token **不计入主表**，但应另存审计文件。

若某次运行无法拿到 OV 内部 usage，则该运行**不能作为正式 run 进入主表**。

#### 4.3.3 新增耗时记录与逐题 telemetry

本版实验的新增重点是 **task 级细粒度记录**。正式评测必须在运行时采集以下时间与明细：

**A. `sample_ingest_metrics`（每个 `sample × group × rerun` 一行）**

至少记录：

- `run_id`
- `group_id`
- `rerun_id`
- `sample_id`
- `sessions_ingested`
- `ingest_start_ts`
- `ingest_end_ts`
- `ingest_elapsed_ms`
- `ingest_gateway_input_tokens`
- `ingest_gateway_output_tokens`
- `ingest_gateway_total_tokens`
- `ingest_ov_internal_input_tokens`
- `ingest_ov_internal_output_tokens`
- `ingest_ov_internal_total_tokens`
- `ingest_input_tokens_total`
- `ingest_total_tokens_total`
- `ov_barrier_wait_ms`（仅 OV 组）
- `post_reset_quiet_wait_ms`（仅 No-OV 组）

**`ingest_elapsed_ms` 的定义必须固定：**

- 起点：该 sample 第一条 ingest 请求发出前的时刻；
- 终点：
  - OV 组：异步完成屏障满足时刻；
  - No-OV 组：最后一次 session reset 完成并经过固定静默窗口时刻。

这样定义后，sample 级 ingest 成本与时间才公平可比。

**B. `task_metrics_direct`（每个 `case × group × rerun` 一行）**

至少记录：

- `run_id`
- `group_id`
- `rerun_id`
- `sample_id`
- `case_uid`
- `category`
- `question`
- `gold_answer`
- `prediction`
- `judge_correct`
- `judge_reasoning_raw`
- `qa_start_ts`
- `qa_end_ts`
- `qa_elapsed_ms`
- `qa_retry_count`
- `qa_error_flag`
- `qa_gateway_input_tokens`
- `qa_gateway_output_tokens`
- `qa_gateway_total_tokens`
- `qa_ov_internal_input_tokens`（若能关联到单题）
- `qa_ov_internal_output_tokens`（若能关联到单题）
- `qa_ov_internal_total_tokens`（若能关联到单题）
- `qa_input_tokens_direct`
- `qa_total_tokens_direct`

**`qa_elapsed_ms` 的定义必须固定：**

- 起点：该题对应的 `/v1/responses` 请求发出前；
- 终点：该题最终响应被完整解析并落盘后；
- 若发生自动重试，则 `qa_elapsed_ms` 包含重试等待时间，并用 `qa_retry_count` 单独记录重试次数；
- judge 耗时不计入 `qa_elapsed_ms`。

#### 4.3.4 构建可回算总量的“逐题摊销指标”

由于一个 sample 的 ingest 成本与时间会被同 sample 内多个 QA 共享，只记录单题 `qa_elapsed_ms` 和单题 `qa_input_tokens_direct` 还不足以与组级总成本完全对上。因此正式实验还必须额外生成 **`task_metrics_amortized`**：

对于任意 sample `s`，设其题目数为 `n_s`，则把该 sample 的共享 ingest 成本均匀摊到该 sample 内每道题上：

\[
\text{alloc\_ingest\_input}_{i}=\frac{\text{ingest\_input\_tokens\_total}_{s}}{n_s}
\]

\[
\text{alloc\_ingest\_elapsed}_{i}=\frac{\text{ingest\_elapsed\_ms}_{s}}{n_s}
\]

对属于该 sample 的每个任务 `i`，定义：

\[
\text{task\_input\_tokens\_amortized}_{i}=
\text{qa\_input\_tokens\_direct}_{i}+\text{alloc\_ingest\_input}_{i}
\]

\[
\text{task\_elapsed\_ms\_amortized}_{i}=
\text{qa\_elapsed\_ms}_{i}+\text{alloc\_ingest\_elapsed}_{i}
\]

同理可定义 `output / total tokens` 的摊销版本。

这样做的好处是：

- **组级总 token** 可以精确回算为所有 `task_input_tokens_amortized` 之和；
- **组级总时间** 可以精确回算为所有 `task_elapsed_ms_amortized` 之和；
- 同时保留“单题直接问答成本”与“单题分摊后端到端成本”两个视角。

### 4.4 judge 脚本必须保留 raw 审计信息

本版不再满足于只保留布尔 `grade`。正式要求 judge 输出至少保留：

- `is_correct`
- `reasoning`
- judge model id
- judge prompt version
- 原始 judge JSON

如果 judge 阶段只输出最终布尔值而不保存 reasoning，则该 run 不满足正式审计要求。

---

## 5. 环境冻结与运行前验证

### 5.1 `manifest.json` 的最低内容

正式实验开始前，记录以下信息到 `manifest.json`：

- OpenClaw 版本（实际运行时检测值）
- OpenViking 版本（实际运行时检测值）
- `openclaw-openviking-doubao` commit SHA
- `OpenViking-LoCoMo10` commit SHA
- `openclaw-eval` commit SHA
- OpenViking 官方仓库 commit SHA
- OpenClaw 官方仓库 commit SHA
- 生成模型别名与 provider 返回的实际模型标识
- judge 模型标识
- 主机 OS / Python / Node 版本
- 运行开始时间、结束时间
- 数据校验结果摘要

### 5.2 Gateway 端点与健康检查

正式 benchmark 前必须完成以下预检：

1. Gateway `/v1/responses` 已启用；
2. OpenClaw Gateway health 正常；
3. 对 OV 组，OpenViking 服务 health 正常；
4. 对 OV 组，在不计分的 smoke-test 中至少完成一次完整的 ingest → commit → archive → memory extraction → recall 闭环；
5. 对 No-OV 组，确认 `plugins.slots.contextEngine == legacy`；
6. 对 OV 组，确认 `plugins.slots.contextEngine == openviking` 且插件已正确注册。

若任一预检不通过，则该环境不得进入正式 benchmark。

### 5.3 版本不符即终止

以下任一情况出现，都必须立刻终止该环境，不得“先跑再说”：

- `openclaw --version != 2026.4.14`
- `openviking --version != 0.3.8`
- 运行时配置与 G1/G2/G3 锁定表不一致
- Gateway 与插件日志显示加载了错误版本或错误 context engine

---

## 6. 实验隔离原则

### 6.1 真正的污染控制单位是 `sample × group × rerun`

为防止长期记忆、agent 级状态或 session 残留跨样本污染，正式实验采用强隔离策略：

- 每个 `sample × group × rerun` 都从**干净快照**恢复独立工作区；
- 跑完立即归档并销毁临时工作区；
- 不允许不同 sample 共享同一个长期运行中的工作区。

### 6.2 快照数量从 4 份改为 3 份

由于正式实验只剩三组，推荐准备三份“空白已安装快照”：

- `snapshot-g1-ov-nomemory`
- `snapshot-g2-noov-stock`
- `snapshot-g3-ov-stock`

每次跑某个 `sample × group × rerun` 时，从对应快照复制出一个临时工作区，跑完即销毁。

### 6.3 显式指定 user，禁止依赖默认值

所有正式 run 都必须显式传入确定性的 `--user`，例如：

```text
g3-ov-stock-run1-sample07
```

不得使用脚本默认 user key。否则不同 sample 或不同 rerun 之间可能意外复用 user，从而污染长期记忆。

### 6.4 保留 session reset 逻辑，不得删除

ingest 后与每题 QA 后的 session reset 逻辑必须保留。原因是本实验测的是**长期记忆召回**，不是把上一个 session 的短上下文直接延续给下一题。任何删除 reset 的做法都会改变评测含义。

---

## 7. 正式执行流程

### 7.1 运行顺序：改为三组循环轮转

为了减少 provider 漂移、系统负载波动和时间窗口偏置，采用**按 sample 轮转三组**的策略，而不是整组跑完再换组。

推荐使用 3 阶循环：

- Sample 1：G1 → G2 → G3
- Sample 2：G2 → G3 → G1
- Sample 3：G3 → G1 → G2
- Sample 4：G1 → G2 → G3
- 后续继续循环

若做多次 fresh rerun，则每个 rerun 再整体平移一次起始组，进一步抵消时间偏置。

### 7.2 每个 `sample × group × rerun` 的标准流程

#### Step 1：恢复干净快照

- 恢复该组对应的 OpenClaw 工作区与 OpenViking 工作区；
- 写入该组锁定配置；
- 对 OV 组写入该工作区专用的 `ov.conf`；
- 注入环境变量形式的 API Key；
- 重启 Gateway，并等待健康检查通过；
- 再次执行运行时版本校验。

#### Step 2：预检

- G2（No-OV / stock）：检查 `plugins.slots.contextEngine == legacy`；
- G1 / G3（OV 组）：检查 `plugins.slots.contextEngine == openviking`，并确认日志出现插件注册成功信息；
- OV 组在正式 benchmark 前运行一次不计分的 `ov-healthcheck.py` 或等价 smoke-test。

#### Step 3：ingest 阶段

对当前 sample：

1. 只 ingest 该 sample 的全部 session，按原始时间顺序输入；
2. 三组统一固定追加同一条 `tail`：
   ```text
   [remember what's said, keep existing memory]
   ```
3. 明确传入：
   - `--sample <idx>`
   - `--user <deterministic_user_key>`
   - `--openclaw-home <current_workdir>`
4. ingest 全程记录 `sample_ingest_metrics`；
5. ingest 结束后执行 session reset。

#### Step 4：异步完成屏障 / 固定静默窗口

- **G1 / G3（OV 组）**：ingest 完成后不能立刻开始 QA。必须等待以下条件满足：
  - `commit_count > 0`
  - `latest_archive_overview` 存在
  - `memories_extracted > 0`
- 等待上限为 **300 秒**；
- **G2（No-OV / stock）**：在最后一次 reset 完成后等待固定短静默窗口（例如 2–5 秒）再进入 QA。

该等待时间必须计入 `ingest_elapsed_ms`。

#### Step 5：QA 阶段

- 仍然只针对当前 sample；
- 仍然使用同一个 `--user`；
- `--parallel 1`；
- 不设 `--count`，跑完该 sample 的全部 QA；
- 每答完一题就 reset 一次 session；
- 每道题都记录 `task_metrics_direct`。

#### Step 6：judge 与逐题明细拼表

当前 sample 的 QA 完成后，立刻执行：

1. judge，得到逐题 `is_correct` 与 `reasoning`；
2. 把数据集元信息、QA 原始输出、usage、耗时、judge 输出按 `case_uid` 合并；
3. 生成：
   - `task_metrics_direct`
   - `task_metrics_amortized`
4. 校验本 sample 的任务行数、字段完整性与 token / time 可回算性。

#### Step 7：归档与销毁

每个 `sample × group × rerun` 跑完后，保存以下文件并销毁临时工作区：

- `openclaw.json` 快照
- `ov.conf` 快照（若适用）
- ingest 原始输出
- QA 原始输出
- judge 原始输出
- `sample_ingest_metrics`
- `task_metrics_direct`
- `task_metrics_amortized`
- OpenClaw 日志
- OpenViking 日志
- manifest 增量记录

---

## 8. 指标定义

### 8.1 组级主指标

#### 8.1.1 任务完成率

对任意组 `g`，定义：

\[
\text{Task Completion Rate}_g=
\frac{\#\{\text{judge 判为 correct 的 case}\}}{1540}
\]

这是主文主表中的第一核心指标。

#### 8.1.2 输入 token 总成本

对任意组 `g`，定义：

\[
\text{Input Tokens Total}_g=
\sum \text{ingest input tokens}+
\sum \text{QA input tokens}
\]

其中，`ingest` 与 `QA` 的 input token 都必须已经包含：

- Gateway usage
- OV 内部 usage（若适用）

judge token 不计入主表。

#### 8.1.3 总耗时（SUT 端到端）

对任意组 `g`，定义：

\[
\text{Elapsed Time Total}_g=
\sum \text{ingest elapsed ms}+
\sum \text{QA elapsed ms}
\]

其中：

- ingest elapsed 包含 OV 屏障等待或 No-OV 固定静默窗口；
- QA elapsed 包含自动重试等待；
- judge 耗时不计入主表。

### 8.2 任务级直接指标（`task_metrics_direct`）

每条 task 明细至少报告：

- `judge_correct`：是否通过
- `qa_input_tokens_direct`
- `qa_output_tokens_direct`
- `qa_total_tokens_direct`
- `qa_elapsed_ms`
- `qa_retry_count`
- `qa_error_flag`

这组指标反映**单题直接问答成本与时延**，不包含共享 ingest 成本。

### 8.3 任务级摊销指标（`task_metrics_amortized`）

每条 task 明细还必须额外报告：

- `alloc_ingest_input_tokens`
- `alloc_ingest_output_tokens`
- `alloc_ingest_total_tokens`
- `alloc_ingest_elapsed_ms`
- `task_input_tokens_amortized`
- `task_total_tokens_amortized`
- `task_elapsed_ms_amortized`

这组指标反映**把 sample 共享 ingest 成本均摊到单题后的端到端成本**，可用于：

- 和组级总量严格对账；
- 观察“平均每题真实占用多少系统资源”。

### 8.4 次指标

建议同时报告：

- 输出 token 总计
- 总 token
- 每正确题成本：
  \[
  \frac{\text{Input Tokens Total}}{\text{Correct Cases}}
  \]
- 每正确题耗时：
  \[
  \frac{\text{Elapsed Time Total}}{\text{Correct Cases}}
  \]
- 直接 QA 平均耗时 / 中位数 / P90
- 摊销后单题耗时均值 / 中位数 / P90
- 各 `category` 完成率
- 各 `sample_id` 完成率
- API 错误率 / 超时率 / 重试率

---

## 9. 统计分析方案

### 9.1 三个预先注册的计划比较

本版只保留以下三个 planned comparisons：

1. **C1：G1 vs G2**
   - `OV / no-memory` vs `No-OV / stock`
   - 这是一个**部署层面的比较**：比较低 token 的 OV 配置与默认 stock 基线
   - 注意：该比较同时改变了 `contextEngine` 与 `memory`，因此**不解释为单因素主效应**

2. **C2：G3 vs G2**
   - `OV / stock` vs `No-OV / stock`
   - 这是最接近“在保留 stock memory 的前提下，引入 OpenViking 是否带来收益”的比较
   - 该比较是本版最重要的单因素对照之一

3. **C3：G3 vs G1**
   - `OV / stock` vs `OV / no-memory`
   - 该比较回答“在 OpenViking 已启用时，保留 `memory-core` 是否值得”

### 9.2 主检验方法

主检验建议使用**按 `sample_id` 聚类的 paired bootstrap**：

- 重采样单位：`sample_id`
- 配对方式：对同一 sample 的两组结果做成对比较
- 对以下指标分别计算 95% CI：
  - 完成率差值
  - 输入 token 差值
  - 总耗时差值
  - 每正确题成本差值
  - 直接 QA 平均耗时差值
  - 摊销后单题耗时差值

### 9.3 任务级统计只作补充

逐题 1540-case 明细非常重要，但任务级统计只作为补充证据，不作为主文主检验的唯一依据。可以附加报告：

- case 层面的 McNemar 检验（针对 `judge_correct`）
- category 分层结果
- 任务级 token / latency 分布图
- 胜出 / 失利 case 的误差分析

### 9.4 本版不再报告的分析

由于本版不再包含 `No-OV / no-memory`，因此以下分析不再作为正式输出：

- 4-cell 差分中的差分
- 完整的 `OV × memory-core` 交互项估计
- 四格析因 ANOVA / GEE 主效应与交互效应报告

如需恢复这些分析，必须重新跑回第四组，而不是从三组结果中“推出来”。

---

## 10. 失败处理与重跑规则

### 10.1 请求失败

保留评测脚本的自动重试机制；但所有重试都必须写入 `qa_retry_count` 或等价字段，不能悄悄吞掉。

### 10.2 何时判为无效 run

满足以下任一条件，该 `sample × group × rerun` 判为无效并整体重跑：

- 版本校验失败
- 配置预检失败
- OV 组健康检查失败
- ingest 阶段中断且不可恢复
- OV 组在 300 秒内未达到异步完成屏障
- `sample_ingest_metrics` 缺失或损坏
- `task_metrics_direct` 缺失、字段不全或行数不对
- `task_metrics_amortized` 缺失或无法与组级总量对账
- 日志快照缺失
- 配置快照缺失
- judge 原始输出缺失

### 10.3 重跑原则

- 只要某个 `sample × group × rerun` 无效，就必须从干净快照整体重跑该块；
- 不允许把不同时间、不同快照下的部分结果拼接成同一个 sample 结果；
- judge 阶段失败可只重跑 judge，但不得改动 SUT 输出；
- 若最终三组中的任一组总 task 行数不是 **1540**，则整组结果不得进入主表。

---

## 11. 结果呈现格式

### 11.1 主表（三组总量结果）

| 组 ID | 组别           | `plugins.slots.memory` | `plugins.slots.contextEngine` | `plugins.entries.openviking.enabled` | `plugins.deny`   | 任务完成率 | 成本：输入 token（总计） | 总耗时（SUT） | 直接 QA 平均耗时 |
| ----- | -------------- | ---------------------- | ----------------------------- | ------------------------------------ | ---------------- | ---------- | ------------------------ | ------------- | ---------------- |
| G1    | OV / no-memory | `none`                 | `openviking`                  | `true`                               | `[]`             |            |                          |               |                  |
| G2    | No-OV / stock  | `memory-core`          | `legacy`                      | `false`                              | `["openviking"]` |            |                          |               |                  |
| G3    | OV / stock     | `memory-core`          | `openviking`                  | `true`                               | `[]`             |            |                          |               |                  |

### 11.2 计划比较增益表

| 比较 | 比较性质 | 完成率差值（pp） | 相对提升（%） | 输入 token 差值 | 输入 token 降幅（%） | 总耗时差值 | 直接 QA 平均耗时差值 | 每正确题成本变化 |
| ---- | -------- | ---------------- | ------------- | --------------- | -------------------- | ---------- | -------------------- | ---------------- |
| C1: G1 vs G2 | 部署比较 | | | | | | | |
| C2: G3 vs G2 | 单因素核心比较 | | | | | | | |
| C3: G3 vs G1 | OV 内部 memory-core 比较 | | | | | | | |

### 11.3 `sample_id` 分层结果表

| `sample_id` | 题量 | G1 完成率 | G2 完成率 | G3 完成率 | G1 输入 token | G2 输入 token | G3 输入 token | G1 总耗时 | G2 总耗时 | G3 总耗时 |
| ----------- | ---- | --------- | --------- | --------- | ------------- | ------------- | ------------- | --------- | --------- | --------- |
| Sample 1    |      |           |           |           |               |               |               |           |           |           |
| Sample 2    |      |           |           |           |               |               |               |           |           |           |
| ...         |      |           |           |           |               |               |               |           |           |           |

### 11.4 `category` 分层结果表

| Category | G1 完成率 | G2 完成率 | G3 完成率 | G1 平均单题摊销 input tokens | G2 平均单题摊销 input tokens | G3 平均单题摊销 input tokens | G1 平均单题摊销耗时 | G2 平均单题摊销耗时 | G3 平均单题摊销耗时 |
| -------- | --------- | --------- | --------- | ---------------------------- | ---------------------------- | ---------------------------- | ------------------- | ------------------- | ------------------- |
| 1        |           |           |           |                              |                              |                              |                     |                     |                     |
| 2        |           |           |           |                              |                              |                              |                     |                     |                     |
| 3        |           |           |           |                              |                              |                              |                     |                     |                     |
| 4        |           |           |           |                              |                              |                              |                     |                     |                     |

### 11.5 逐题明细文件（不要求主文全文展示，但必须随 artifact 提供）

至少提供以下两个文件：

1. `task_metrics_direct_all_groups.parquet/csv`
2. `task_metrics_amortized_all_groups.parquet/csv`

推荐字段示意：

| group_id | rerun_id | sample_id | case_uid | category | judge_correct | qa_input_tokens_direct | qa_total_tokens_direct | qa_elapsed_ms | alloc_ingest_input_tokens | task_input_tokens_amortized | task_elapsed_ms_amortized |
| -------- | -------- | --------- | -------- | -------- | ------------- | ---------------------- | ---------------------- | ------------- | ------------------------- | --------------------------- | ------------------------- |

---

## 12. 最终交付物清单

正式实验结束后，至少产出如下结构：

```text
artifacts/
  manifest.json

  configs/
    group-g1-ov-nomemory.openclaw.json
    group-g2-noov-stock.openclaw.json
    group-g3-ov-stock.openclaw.json
    group-g1-ov-nomemory.ov.conf
    group-g3-ov-stock.ov.conf

  raw/
    ingest/{group}/{sample}.json
    qa/{group}/{sample}.jsonl
    judge_raw/{group}/{sample}.jsonl

  metrics/
    sample_ingest/{group}.parquet
    task_direct/{group}.parquet
    task_amortized/{group}.parquet
    task_metrics_direct_all_groups.parquet
    task_metrics_amortized_all_groups.parquet

  logs/
    openclaw/{group}/{sample}.log
    openviking/{group}/{sample}.log

  summary/
    main_table.md
    planned_comparisons.md
    sample_breakdown.md
    category_breakdown.md
    latency_breakdown.md
    per_task_schema.md
```

---

## 13. 参考仓库

- 清洗数据集：`https://github.com/DANNHIROAKI/OpenViking-LoCoMo10`
- 一键配置仓库：`https://github.com/DANNHIROAKI/openclaw-openviking-doubao`
- OpenViking 官方仓库：`https://github.com/volcengine/OpenViking`
- OpenClaw 官方仓库：`https://github.com/openclaw/openclaw`

---

## 14. 执行备注

1. 本版实验已经从“四组因子设计”改成“三组正式对照 + 逐题细粒度记录”；
2. `G2 = No-OV / stock` 是唯一保留的非 OV 基线，因此所有正式比较都以它或 OV 内部对照为锚点；
3. 正式主表继续报告“任务完成率 + 输入 token 总计”，但必须同时交付逐题 pass/fail、逐题 token、逐题耗时；
4. judge token 与 judge 耗时不进入主表，但必须保留原始输出以供审计；
5. 正式结论以本文件为准。
