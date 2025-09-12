# Kronos模型快速开始指南

## 🚀 三种集成方式

### 1. 直接集成 (最简单)

```python
# 安装依赖
pip install numpy pandas torch einops huggingface_hub matplotlib tqdm safetensors

# 使用示例
import sys
sys.path.append('/path/to/Kronos')
from model import Kronos, KronosTokenizer, KronosPredictor

# 加载模型
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)

# 预测
pred_df = predictor.predict(df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp, pred_len=288)
```

### 2. 使用封装类 (推荐)

```python
# 运行示例
python simple_integration_example.py

# 在你的代码中使用
from simple_integration_example import SimpleKronosPredictor

predictor = SimpleKronosPredictor("kronos-small", "auto")
predictions = predictor.predict_next_24h(your_data)
signals = predictor.get_trading_signals(your_data)
```

### 3. API服务 (微服务架构)

```bash
# 启动API服务
python api_service_example.py

# 在另一个终端运行客户端
python client_example.py
```

## 📊 数据格式要求

你的数据需要包含以下列：
- `timestamps`: 时间戳
- `open`: 开盘价
- `high`: 最高价  
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量

```python
# 示例数据格式
data = {
    'timestamps': ['2024-01-01 09:00:00', '2024-01-01 09:05:00', ...],
    'open': [100.0, 101.0, ...],
    'high': [101.5, 102.0, ...],
    'low': [99.5, 100.5, ...],
    'close': [101.0, 101.5, ...],
    'volume': [1000, 1200, ...]
}
df = pd.DataFrame(data)
```

## 🎯 模型选择

| 模型 | 参数量 | 速度 | 精度 | 适用场景 |
|------|--------|------|------|----------|
| kronos-mini | 4.1M | 最快 | 中等 | 实时交易 |
| kronos-small | 24.7M | 快 | 好 | 一般应用 |
| kronos-base | 102.3M | 慢 | 最好 | 离线分析 |

## ⚡ 快速测试

1. **测试直接集成**:
```bash
cd Kronos
python simple_integration_example.py
```

2. **测试API服务**:
```bash
# 终端1: 启动服务
python api_service_example.py

# 终端2: 测试客户端
python client_example.py
```

## 🔧 常见问题

### Q: 内存不足怎么办？
A: 使用更小的模型或减少预测长度
```python
predictor = SimpleKronosPredictor("kronos-mini", "cpu")
predictions = predictor.predict_custom_hours(df, hours=12)  # 12小时而不是24小时
```

### Q: 预测质量不好怎么办？
A: 调整预测参数
```python
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=0.8,           # 降低温度
    top_p=0.95,      # 提高top_p
    sample_count=3   # 多次采样
)
```

### Q: 如何集成到现有交易系统？
A: 使用API服务方式，通过HTTP调用
```python
import requests

response = requests.post('http://localhost:5000/predict/signals', json={
    'ohlcv_data': your_data,
    'lookback': 400,
    'pred_len': 288
})

signals = response.json()['signals']
```

## 📚 更多示例

- `examples/prediction_example.py` - 基础预测示例
- `examples/prediction_batch_example.py` - 批量预测
- `integration_guide.md` - 详细集成指南
- `simple_integration_example.py` - 简化集成示例
- `api_service_example.py` - API服务示例
- `client_example.py` - API客户端示例

## 🆘 需要帮助？

1. 查看 `integration_guide.md` 获取详细文档
2. 运行示例代码了解使用方法
3. 检查数据格式是否符合要求
4. 确保安装了所有依赖包

开始使用Kronos进行金融预测吧！ 🎉

