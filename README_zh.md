# PersonSim（中文对照版）

PersonSim 是一个紧凑的运行时，用于在有序、on-policy 的用户 Lifelong 流中评估记忆增强型个人助手。这个公开仓库从两条刻意保持可比性的记忆路径开始：

- `naive_rag` —— 参考基线：最近完成任务的 transcript，加上一条经语义检索得到的较早 transcript。
- `mem0` —— 通过同一 adapter contract 对 [Mem0](https://github.com/mem0ai/mem0) 的可选接入，用于展示如何加入其他 memory system。

仓库包含已发布的 Lifelong aggregate：20 位 persona、每位 50 个任务；还包含 `persona_instances_5x20.jsonl` 元数据（100 个 instance，五个 family 均衡）。

## 设计约束

- 每个 persona 在每个 condition 中只有一条顺序执行的 on-policy 轨迹。
- 每个 memory backend 都隔离到一个 persona，并且必须为每个 retrieval candidate 提供源任务溯源信息。
- 在 Tested Agent 看到任何内容之前，runner 会拒绝跨用户候选项。
- Evaluator 是后验执行的；其输出绝不会写入 memory。

## 安装

需要 Python 3.10 或更高版本。

```bash
git clone <YOUR_REPOSITORY_URL> persosim
cd persosim
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

如需 Mem0 支持，安装可选依赖：

```bash
pip install -e '.[mem0,dev]'
```

## 配置 provider

在本地复制模板，并填入一条兼容 OpenAI 的 API 路由。复制出的文件会被 Git 忽略。

```bash
cp configs/api.example.md configs/api.md
chmod 600 configs/api.md
```

`[global]` 默认保存所有角色共用的一组 `base_url` 与 `api_key`。角色段只选择模型与生成参数：

| 角色 | 用途 |
|---|---|
| `tested_agent` | 被比较的模型 |
| `user_simulator` | 为每个任务生成两条自然用户输入 |
| `evaluator` | 产生严格的后验 JSON 判定 |
| `embedding` | 为 NaiveRAG transcript 建索引和查询 |

`memory` 仅由 Mem0 等 adapter 使用，默认也继承 `[global]`。

如果某个角色必须使用另一个 provider，只在该角色段中加入 `base_url` 与 `api_key`。绝不要提交 `api.md`、`.env`、输出文件夹或 provider 响应日志。

## 运行 NaiveRAG

此命令为 `cap_002` 运行三个任务。它会创建一个新的运行根目录，并拒绝覆盖已存在的目录。

```bash
python scripts/run_lifelong.py \
  --run-id demo-naiverag-001 \
  --persona-id cap_002 \
  --task-limit 3 \
  --memory-system naive_rag \
  --tested-agent-model deepseek-v4-flash \
  --user-simulator-model gpt-4o-mini \
  --evaluator-model deepseek-v4-pro \
  --embedding-model text-embedding-3-small
```

## 运行 Mem0

Mem0 是可选的，并使用相同的任务流、角色设置、输出 schema 和可见记忆预算。

```bash
pip install -e '.[mem0]'

python scripts/run_lifelong.py \
  --run-id demo-mem0-001 \
  --persona-id cap_002 \
  --memory-system mem0 \
  --task-limit 3 \
  --tested-agent-model deepseek-v4-flash \
  --user-simulator-model gpt-4o-mini \
  --evaluator-model deepseek-v4-pro \
  --embedding-model text-embedding-3-small
```

## 加入其他 memory system

在 `src/memory.py` 中实现 `MemoryBackend`：

```python
class MyMemory:
    system_name = "my_memory"

    def write_task(self, write: MemoryWrite) -> None: ...
    def retrieve(self, query: str, limit: int) -> list[MemoryCandidate]: ...
    def close(self) -> None: ...
```

## 数据发布与许可证

代码采用 MIT 许可证。仓库包含发布的 Lifelong aggregate 和 persona-instance metadata。若要评估另一版冻结数据，请通过 `--data /path/to/20user_lifeline.json` 指向它，并将配对的 `20user_eval.json` 放在同一目录。
