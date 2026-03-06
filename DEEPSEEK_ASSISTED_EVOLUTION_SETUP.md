# DeepSeek 辅助 AGI 进化配置说明

## 1. 目标

本方案将外部 `deepseek-chat` 接入本地 AGI 进化守护进程，用于提供策略建议与失败兜底。
同时提供流量预算控制，避免 API 费用失控。

## 2. 密钥安全（不上传开源）

推荐方式：使用本地忽略目录 `secrets/`。

1. 在仓库根目录创建密钥文件（该路径被 `.gitignore` 忽略）：

```bash
mkdir -p secrets
echo "<YOUR_DEEPSEEK_API_KEY>" > secrets/deepseek_api_key.txt
chmod 600 secrets/deepseek_api_key.txt
```

2. 也可使用环境变量（优先级更高）：

```bash
export DEEPSEEK_API_KEY="<YOUR_DEEPSEEK_API_KEY>"
```

## 3. 启动命令（带外部辅助）

```bash
/Users/imymm/H2Q-Evo/.venv/bin/python tools/agi_self_evolution_daemon.py \
  --assist-provider deepseek \
  --assist-model deepseek-chat \
  --assist-base-url https://api.deepseek.com \
  --assist-key-file secrets/deepseek_api_key.txt \
  --interval-minutes 10 \
  --rounds 0 \
  --profile quick \
  --skip-rsa
```

## 4. 流量循环控制参数

- `--assist-max-calls-per-round`：每轮最多外部调用次数。
- `--assist-max-est-tokens-per-round`：每轮 token 预算。
- `--assist-max-est-tokens-total`：进程总 token 预算。
- `--no-assist-fallback`：禁用本地失败时使用外部答案兜底。

示例：

```bash
/Users/imymm/H2Q-Evo/.venv/bin/python tools/agi_self_evolution_daemon.py \
  --assist-provider deepseek \
  --assist-max-calls-per-round 2 \
  --assist-max-est-tokens-per-round 4000 \
  --assist-max-est-tokens-total 20000 \
  --rounds 12
```

## 5. 产物

- 单轮报告：`reports/agi_self_evolution_round_*.json`
- 日报：`reports/agi_self_evolution_daily_*.json`
- 日报 Markdown：`reports/AGI自我进化日报_*.md`
- 告警：`reports/agi_self_evolution_alert_*.json`

每条 round entry 中包含 `runtime.assist` 字段，可追踪外部辅助是否启用、是否成功、消耗 token。
