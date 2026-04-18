# OpenViking × OpenClaw 三组正式 benchmark 仓库

这个仓库按锁定版实验计划实现了完整的三组 benchmark harness：

- **G1** = `OV / no-memory`
- **G2** = `No-OV / stock`
- **G3** = `OV / stock`

并且严格执行以下原则：

- 正式流量统一走 **OpenClaw Gateway `/v1/responses`**
- 锁定 **OpenClaw 2026.4.14** 与 **OpenViking 0.3.8**
- 以 `sample × group × rerun` 为污染隔离单位
- 保留 `sample_ingest_metrics`、`task_metrics_direct`、`task_metrics_amortized`
- judge 单独记录原始 JSON 与 reasoning，不计入主表 token / latency
- 主表、逐题明细、日志、配置快照、manifest 一次性生成

仓库已经内置了清洗后的 **1540-case OpenViking-LoCoMo10** 数据集副本：

- `vendor/data/openviking-locomo10-1540/locomo10_openviking_1540.json`
- `vendor/data/openviking-locomo10-1540/locomo10_openviking_1540.jsonl`
- `vendor/data/openviking-locomo10-1540/manifest.json`

另外，`docs/EXPERIMENT_PLAN.locked.md` 是你锁定的正式版实验计划副本，后续实现默认以它为准。

## 一条命令跑完整实验

1. 复制环境变量模板：

```bash
cp .env.example .env
```

2. 填好 `.env` 里的 endpoint / model / API key。

3. 直接运行：

```bash
bash run.sh
```

默认行为：

- 创建 `.venv`
- 安装 Python 依赖
- 克隆以下参考仓库到 `runtime/repos/`
  - `OpenViking-LoCoMo10`
  - `openclaw-openviking-doubao`
  - `openclaw`
  - `OpenViking`
  - `openclaw-eval`
- 安装 OpenClaw CLI 到 `runtime/toolchains/openclaw/`
- 安装 OpenViking Python runtime 到 `runtime/toolchains/openviking-venv/`
- 生成 `_base` 基础快照
- 复制出 G1 / G2 / G3 三个组快照
- 对 OV 组跑 smoke test
- 按 `sample` 轮转三组顺序正式跑 benchmark
- 汇总生成 artifacts

## 需要的环境变量

至少要设置这些：

```dotenv
OV_OC_ARK_API_KEY=...
OV_OC_OPENCLAW_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
OV_OC_OPENCLAW_MODEL_ID=...
OV_OC_OV_VLM_API_BASE=https://ark.cn-beijing.volces.com/api/v3
OV_OC_OV_VLM_MODEL=...
OV_OC_OV_EMBED_API_BASE=https://ark.cn-beijing.volces.com/api/v3
OV_OC_OV_EMBED_MODEL=...
OV_OC_JUDGE_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
OV_OC_JUDGE_MODEL=...
```

可选：

```dotenv
OPENCLAW_GATEWAY_TOKEN=
OPENVIKING_API_KEY=
OV_OC_RERUNS=1
OV_OC_STRICT_OV_USAGE=1
OV_OC_FORCE_EXPLICIT_COMMIT_FOR_OV_TELEMETRY=0
OV_OC_RUN_OV_SMOKE=1
OV_OC_BARRIER_REQUIRE_MEMORIES=0
```

说明：

- `OV_OC_OPENCLAW_MODEL_ID` 建议填你的 **Seed-2.0-Code** 自定义推理接入点 ID。
- `OV_OC_OV_VLM_MODEL` / `OV_OC_OV_EMBED_MODEL` 建议填 OpenViking 内部使用的 VLM / embedding 接入点 ID。
- `OV_OC_JUDGE_MODEL` 可填 judge 使用的模型或接入点 ID。
- 如果 `OPENCLAW_GATEWAY_TOKEN` / `OPENVIKING_API_KEY` 为空，脚本会自动生成临时值。

### 故障排除

- **`Gateway did not become healthy ... Last health payload: None`**：通常是本机 **18789** 已被其他 OpenClaw Gateway 占用，`openclaw health` 连到了旧进程且 **token 不一致**。本仓库已在每次启动 Gateway 前为当前 workdir **自动分配空闲端口** 并写入 `gateway.port`，一般无需手动停进程；若仍异常，可检查 `ss -tlnp | grep 18789` 并结束残留 `openclaw-gatewa` 后再跑。

- **OpenViking HTTP 400（租户 API）**：使用 **root API key** 调 ` /api/v1/sessions/...` 等租户接口时，必须同时带 **`X-OpenViking-Account`** 与 **`X-OpenViking-User`**（与插件默认一致时可均为 `default`）。基准里的 `OpenVikingInspector` 已按此发送。

- **`X-OpenViking-Agent` 不一致**：插件在 `plugins.entries.openviking.config.agentId` 为 **`default`** 时，实际发往 OpenViking 的是 OpenClaw 的 **`ctx.agentId`（一般为 `main`）**，而不是字面量 `default`。直接调 OpenViking HTTP 时需与插件一致。

- **Smoke / 短对话达不到自动 commit**：插件默认 **`commitTokenThreshold=20000`** pending token 才在 `afterTurn` 里自动 `commit`；短探针对话会一直 `below_threshold`。prepare 里的 smoke 会对临时 workdir **单独把阈值降到 128**，以便触发归档；正式块跑仍可用显式 commit + barrier（见 `barrier_require_extracted_memories`）。

- **损坏的组快照目录**：若 `runtime/snapshots/<group>/` 存在但没有 `.openclaw/openclaw.json`，prepare 会 **自动删掉并重做**（不必手动 `rm`）。若仍异常，可用 `bash run.sh --fresh` 清空 artifacts/work 后重跑。

- **`requests.exceptions.ReadTimeout`（`/v1/responses`）**：当前实现会对 `ReadTimeout` / `ConnectionError` / 常见可重试状态码（`429/5xx`）做有限重试（线性 backoff）；若仍超时，优先增大 `runtime.request_timeout_seconds`，并检查上游模型 endpoint 延迟是否异常。

## 目录结构

运行后重点看：

```text
artifacts/
  manifest.json

  configs/
    group-g1-ov-no-memory.openclaw.json
    group-g2-no-ov-stock.openclaw.json
    group-g3-ov-stock.openclaw.json
    group-g1-ov-no-memory.ov.conf
    group-g3-ov-stock.ov.conf

  raw/
    ingest/{group}/{rerun}/{sample}.json
    qa/{group}/{rerun}/{sample}.jsonl
    judge_raw/{group}/{rerun}/{sample}.jsonl

  metrics/
    sample_ingest/{group}.parquet
    task_direct/{group}.parquet
    task_amortized/{group}.parquet
    task_metrics_direct_all_groups.parquet
    task_metrics_amortized_all_groups.parquet

  logs/
    openclaw/{group}/{rerun}/{sample}.log
    openviking/{group}/{rerun}/{sample}.log

  summary/
    main_table.md
    planned_comparisons.md
    sample_breakdown.md
    category_breakdown.md
    latency_breakdown.md
    per_task_schema.md

  blocks/
    R1/G1/conv-26/...
    ...
```

其中：

- `blocks/` 是 block 级中间产物，用于 resume / 审计 / 局部重跑
- `metrics/` 与 `summary/` 是正式汇总结果
- `manifest.json` 记录版本、commit SHA、模型 ID、数据校验、汇总状态

## 运行顺序与隔离

默认轮转顺序：

- Sample 1: `G1 → G2 → G3`
- Sample 2: `G2 → G3 → G1`
- Sample 3: `G3 → G1 → G2`
- 之后继续循环

每个 block 都会：

1. 从该组干净快照复制独立 workdir
2. 动态渲染 `.openviking/ov.conf`
3. 启动 foreground OpenClaw Gateway
4. OV 组等待 local mode 的 OpenViking 服务就绪
5. ingest 当前 sample 全部 session，并在每个 session 后 reset
6. OV 组等待 barrier；No-OV 组等待固定 quiet window
7. QA 当前 sample 全部题，每题后 reset
8. 解析 OpenViking log，回填 OV 内部 token usage
9. 跑 judge
10. 生成 direct / amortized 指标并落盘

## 关于 OV 内部 usage 的严格模式

默认 `OV_OC_STRICT_OV_USAGE=1`。

这意味着：

- OV 组必须能从 `openviking.log` 和/或显式 commit telemetry 回收到内部 token usage
- 如果 ingest 或任一 QA 无法恢复 OV 内部 usage，该 block 会被判为 **invalid run**
- invalid block 会从干净快照整体重跑，超过最大重试次数后整个实验失败

这个默认值是故意的：它和你锁定计划里的正式 run 口径一致。

如果只是想先打通流程，可以临时把：

```dotenv
OV_OC_STRICT_OV_USAGE=0
```

这样仍然会出结果，但不应该把它直接写进正式主表。

## 常用命令

完整跑：

```bash
bash run.sh
```

全新清空后重跑：

```bash
bash run.sh --fresh
```

只准备环境：

```bash
bash run.sh prepare
```

只执行 benchmark（假设 snapshot 已准备好）：

```bash
bash run.sh run
```

只重做汇总：

```bash
bash run.sh summarize
```

快速自检（本次修复验证方式）：

```bash
source .venv/bin/activate
PYTHONPATH=src CUDA_VISIBLE_DEVICES=<最空闲GPU编号> python -m pytest tests/test_static_contracts.py -q
```

## 代码结构

```text
src/ovoc_bench/
  config.py       # YAML 配置加载
  dataset.py      # 数据集加载与 1540-case 校验
  gitmeta.py      # 参考仓库 clone / commit 记录
  openclaw.py     # OpenClaw 安装、配置、Gateway 控制、/v1/responses 调用
  openviking.py   # OpenViking 安装、HTTP inspector、barrier、log telemetry 解析
  judge.py        # judge 调用与原始 JSON 保存
  metrics.py      # sample/task direct/amortized 指标构造与对账
  summary.py      # 主表、planned comparisons、sample/category/latency 汇总
  runner.py       # benchmark 主编排器
  cli.py          # CLI 入口
```

## 目前实现中的几个设计选择

1. **OpenClaw 用 foreground `openclaw gateway run` 启动**  
   这样每个 block 的生命周期完全由 harness 控制，日志也更容易按 block 归档。

2. **OpenViking 走 OpenClaw 插件 local mode**  
   不绕过 Gateway 发正式 ingest / QA 请求；只在 health / barrier / telemetry 审计时访问 OV HTTP API。

3. **OV 配置快照只保留 redacted 版本**  
   实际运行时 `ov.conf` 在 block workdir 里动态渲染，artifact 中保存脱敏版。

4. **resume 以 block 为单位**  
   已完成且 `status.json.valid=true` 的 block 会自动跳过。

## 最后

这份仓库的目标不是 demo，而是把你的正式版计划落成一个可执行、可审计、可 resume 的 benchmark harness。

只要 `.env` 填对，并且本机能正常访问 OpenClaw / OpenViking / Ark，对外入口就是：

```bash
bash run.sh
```
