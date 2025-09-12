#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kronos简化预测服务
专门用于接收K线数据并返回预测结果
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import sys
import os
import logging
from datetime import datetime
import traceback

# 添加Kronos模型路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    logger.info("Kronos模型模块导入成功")
except ImportError as e:
    logger.error(f"无法导入Kronos模型模块: {e}")
    sys.exit(1)

app = Flask(__name__)
CORS(app)  # 启用跨域支持

# 全局预测器实例
predictor = None

def init_predictor():
    """初始化预测器"""
    global predictor
    
    model_name = os.getenv('MODEL_NAME', 'kronos-small')
    device = os.getenv('DEVICE', 'auto')
    
    if device == "auto":
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    
    logger.info(f"正在初始化Kronos预测器: {model_name} 到设备: {device}")
    
    try:
        # 加载tokenizer和模型
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained(f"NeoQuasar/{model_name}")
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
        logger.info("预测器初始化成功!")
        return True
    except Exception as e:
        logger.error(f"预测器初始化失败: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy' if predictor is not None else 'unhealthy',
        'predictor_loaded': predictor is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    预测接口
    接收K线数据，返回预测结果
    """
    try:
        if predictor is None:
            return jsonify({
                'success': False,
                'error': '预测器未初始化'
            }), 500
        
        # 获取请求数据
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': '未提供JSON数据'
            }), 400
        
        # 验证必要字段
        if 'kline_data' not in data:
            return jsonify({
                'success': False,
                'error': '缺少kline_data字段'
            }), 400
        
        kline_data = data['kline_data']
        if not isinstance(kline_data, list) or len(kline_data) == 0:
            return jsonify({
                'success': False,
                'error': 'kline_data必须是非空数组'
            }), 400
        
        # 转换为DataFrame
        try:
            df = pd.DataFrame(kline_data)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'K线数据格式错误: {str(e)}'
            }), 400
        
        # 验证必要列
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({
                'success': False,
                'error': f'缺少必要字段: {missing_columns}'
            }), 400
        
        # 转换时间戳
        try:
            df['timestamps'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'时间戳格式错误: {str(e)}'
            }), 400
        
        # 获取预测参数
        lookback = data.get('lookback', 400)  # 默认使用400个历史数据点
        pred_hours = data.get('pred_hours', 24)  # 默认预测24小时
        pred_len = pred_hours * 12  # 假设5分钟K线，每小时12个点
        
        # 检查数据长度
        if len(df) < lookback:
            return jsonify({
                'success': False,
                'error': f'历史数据不足，需要至少{lookback}个数据点，当前只有{len(df)}个'
            }), 400
        
        # 准备输入数据
        x_df = df.iloc[:lookback][['open', 'high', 'low', 'close', 'volume']]
        x_timestamp = df.iloc[:lookback]['timestamps']
        
        # 生成未来时间戳
        last_timestamp = df['timestamps'].iloc[lookback-1]
        time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
        future_timestamps = pd.date_range(
            start=last_timestamp + time_diff,
            periods=pred_len,
            freq=time_diff
        )
        
        logger.info(f"开始预测: 历史数据{lookback}个点，预测{pred_hours}小时")
        
        # 进行预测
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=future_timestamps,
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=False
        )
        
        # 准备返回结果
        predictions = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            predictions.append({
                'timestamp': future_timestamps[i].isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })
        
        logger.info(f"预测完成: 生成了{len(predictions)}个预测点")
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'prediction_count': len(predictions),
            'pred_hours': pred_hours,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"预测错误: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict/signals', methods=['POST'])
def predict_with_signals():
    """
    预测并生成交易信号
    """
    try:
        # 先获取预测结果
        pred_response = predict()
        if pred_response[1] != 200:
            return pred_response
        
        pred_data = pred_response[0].get_json()
        if not pred_data['success']:
            return jsonify(pred_data), 500
        
        predictions = pred_data['predictions']
        data = request.get_json()
        kline_data = data['kline_data']
        
        # 当前价格
        current_price = kline_data[-1]['close']
        
        # 计算不同时间点的预测价格
        pred_1h = predictions[11]['close'] if len(predictions) > 11 else predictions[-1]['close']
        pred_6h = predictions[71]['close'] if len(predictions) > 71 else predictions[-1]['close']
        pred_24h = predictions[-1]['close']
        
        # 计算价格变化百分比
        change_1h = (pred_1h - current_price) / current_price * 100
        change_6h = (pred_6h - current_price) / current_price * 100
        change_24h = (pred_24h - current_price) / current_price * 100
        
        # 生成交易信号
        def generate_signal(change):
            if change > 2:
                return "STRONG_BUY"
            elif change > 0.5:
                return "BUY"
            elif change < -2:
                return "STRONG_SELL"
            elif change < -0.5:
                return "SELL"
            else:
                return "HOLD"
        
        signals = {
            'current_price': current_price,
            'predictions': {
                '1h': {'price': pred_1h, 'change_pct': change_1h, 'signal': generate_signal(change_1h)},
                '6h': {'price': pred_6h, 'change_pct': change_6h, 'signal': generate_signal(change_6h)},
                '24h': {'price': pred_24h, 'change_pct': change_24h, 'signal': generate_signal(change_24h)}
            },
            'overall_signal': generate_signal(change_24h),
            'confidence': min(abs(change_24h) / 5.0, 1.0)
        }
        
        return jsonify({
            'success': True,
            'signals': signals,
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"信号生成错误: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # 初始化预测器
    if init_predictor():
        port = int(os.getenv('PORT', 5000))
        logger.info(f"启动Kronos预测服务...")
        logger.info(f"服务地址: http://0.0.0.0:{port}")
        logger.info(f"API端点:")
        logger.info(f"  GET  /health - 健康检查")
        logger.info(f"  POST /predict - 价格预测")
        logger.info(f"  POST /predict/signals - 预测+交易信号")
        
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        logger.error("预测器初始化失败，服务无法启动")
        sys.exit(1)
