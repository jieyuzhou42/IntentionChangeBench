目标目录
intention_change_bench/
│
├── envs/
│   ├── base_env.py
│   ├── shopping_env.py
│   ├── booking_env.py
│   └── scheduling_env.py
│
├── agents/
│   ├── execution_agent.py
│   ├── oracle_executor.py
│   ├── noisy_executor.py
│   ├── llm_executor.py
│   └── human_simulator.py
│
├── simulators/
│   ├── user_state.py
│   ├── shift_trigger.py
│   ├── shift_policy.py
│   └── realization.py
│
├── evaluators/
│   ├── runtime_logger.py
│   └── metrics.py
│
├── tasks/
│   ├── schemas.py
│   └── generators.py
│
├── run_simulation.py
└── config.yaml


IntentionChangeBench + WebShop 的闭环数据生成器。目标是：

用 WebShop 官方环境 做 backend
在同一个 episode 里做多轮 interaction
agent 先执行 action
environment 返回结果 / feasibility signal
human simulator 再根据环境反馈做 intention shift
最后导出带 env_feedback / trigger / gold_delta / gold_current_intention 的 trajectory JSON
