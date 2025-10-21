# 数据目录说明

## 目录结构

```
data/
├── README.md              # 本文件
├── sample_train.jsonl     # 训练数据示例（可选）
├── sample_val.jsonl       # 验证数据示例（可选）
├── sample_test.json       # 测试数据示例（可选）
└── processed/             # 实际训练数据（需自备，已在.gitignore中忽略）
    ├── llm_input_enriched_train.jsonl
    ├── llm_input_enriched_val.jsonl
    └── llm_input_enriched_test.jsonl
```

## 数据格式

每行一个JSON对象，格式如下：

```json
{
  "flow_id": 1,
  "tuple5": ["192.168.1.100", "8.8.8.8", 53241, 53, 17],
  "window": [1640995200.0, 1640995260.0],
  "features": {
    "pkt_cnt_fwd": 2,
    "pkt_cnt_bwd": 2,
    "byte_cnt_fwd": 128,
    "byte_cnt_bwd": 256,
    "fwd_bwd_ratio_pkt": 1.0,
    "fwd_bwd_ratio_byte": 0.5,
    "tcp_syn_ratio": 0.0,
    "tcp_fin_ratio": 0.0,
    "tcp_rst_ratio": 0.0
  },
  "enriched": {
    "context_summary": "DNS query flow with normal request-response pattern",
    "events": [...],
    "event_count": 1,
    "payload": {...},
    "protocols": ["udp", "dns"],
    "proto_name": "UDP"
  },
  "label": "BENIGN"
}
```

## 必需字段

- `flow_id`: 流量ID
- `tuple5`: 五元组 [源IP, 目标IP, 源端口, 目标端口, 协议]
- `window`: 时间窗口 [开始时间, 结束时间]
- `features`: 数值特征字典
- `enriched`: 富化信息字典
- `label`: 安全标签（BENIGN, DDoS, PortScan等）

## 数据准备

1. 准备您的网络流量数据
2. 转换为上述JSON格式
3. 保存为JSONL文件（每行一个JSON对象）
4. 放置在 `data/processed/` 目录下

## 注意事项

⚠️ **重要**：
- 原始数据文件（.jsonl）较大，已在 `.gitignore` 中忽略
- 不要将真实数据上传到GitHub
- 建议使用示例数据测试流程
- 生产数据请妥善保管
