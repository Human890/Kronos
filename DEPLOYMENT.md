# Kronos预测服务部署指南

## 项目概述

这是一个基于Kronos模型的金融K线数据预测服务，专门用于接收K线数据并返回预测结果。

## 核心文件

- `app.py` - 预测服务主程序
- `Dockerfile` - Docker镜像构建文件
- `docker-compose.yml` - Docker Compose配置
- `requirements.txt` - Python依赖
- `model/` - Kronos模型代码
- `java_examples/` - Java集成示例

## 快速部署

### 1. 构建并启动服务

```bash
# 构建Docker镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 2. 验证服务

```bash
# 健康检查
curl http://localhost:5000/health

# 查看服务信息
curl http://localhost:5000/info
```

## API接口

### 1. 健康检查
```
GET /health
```

### 2. 价格预测
```
POST /predict
Content-Type: application/json

{
    "kline_data": [
        {
            "timestamp": "2024-01-01T00:00:00",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0
        }
        // ... 更多K线数据，至少需要400个数据点
    ],
    "pred_hours": 24,
    "lookback": 400
}
```

### 3. 预测并生成交易信号
```
POST /predict/signals
Content-Type: application/json

{
    "kline_data": [...], // 同上
    "pred_hours": 24,
    "lookback": 400
}
```

## 数据格式说明

### K线数据格式
```json
{
    "timestamp": "2024-01-01T00:00:00",  // ISO格式时间戳
    "open": 100.0,                       // 开盘价
    "high": 101.0,                       // 最高价
    "low": 99.0,                         // 最低价
    "close": 100.5,                      // 收盘价
    "volume": 1000.0                     // 成交量
}
```

### 预测结果格式
```json
{
    "success": true,
    "predictions": [
        {
            "timestamp": "2024-01-02T00:00:00",
            "open": 100.8,
            "high": 101.2,
            "low": 100.5,
            "close": 101.0,
            "volume": 950.0
        }
        // ... 更多预测数据点
    ],
    "prediction_count": 288,
    "pred_hours": 24,
    "timestamp": "2024-01-01T12:00:00"
}
```

### 交易信号格式
```json
{
    "success": true,
    "signals": {
        "current_price": 100.5,
        "predictions": {
            "1h": {
                "price": 100.8,
                "change_pct": 0.3,
                "signal": "BUY"
            },
            "6h": {
                "price": 101.2,
                "change_pct": 0.7,
                "signal": "BUY"
            },
            "24h": {
                "price": 102.0,
                "change_pct": 1.5,
                "signal": "BUY"
            }
        },
        "overall_signal": "BUY",
        "confidence": 0.3
    },
    "predictions": [...], // 预测数据
    "timestamp": "2024-01-01T12:00:00"
}
```

## Java集成示例

参考 `java_examples/` 目录中的示例代码：

1. `KronosPredictorService.java` - 预测服务类
2. `PredictionController.java` - Spring Boot控制器

### 使用步骤

1. 将Java示例代码复制到你的项目中
2. 配置Kronos API地址：
   ```properties
   kronos.api.url=http://localhost:5000
   ```
3. 在你的控制器中注入 `KronosPredictorService`
4. 调用预测方法

## 环境变量

- `MODEL_NAME`: 模型名称 (默认: kronos-small)
- `DEVICE`: 设备类型 (默认: auto)
- `PORT`: 服务端口 (默认: 5000)

## 注意事项

1. **数据要求**: 至少需要400个历史K线数据点
2. **时间格式**: 时间戳必须是ISO格式
3. **数据完整性**: 确保OHLCV数据完整且无缺失值
4. **服务依赖**: 首次启动会下载模型，需要网络连接
5. **资源要求**: 建议至少4GB内存

## 故障排除

### 服务启动失败
```bash
# 查看详细日志
docker-compose logs kronos-predictor

# 检查模型下载状态
docker exec -it kronos-predictor ls -la /root/.cache/huggingface/
```

### 预测失败
1. 检查K线数据格式是否正确
2. 确认数据点数量是否足够（≥400个）
3. 验证时间戳格式是否为ISO格式

### 性能优化
1. 使用GPU加速：设置 `DEVICE=cuda:0`
2. 调整模型大小：使用 `kronos-mini` 获得更快速度
3. 减少预测时长：设置较小的 `pred_hours` 值

