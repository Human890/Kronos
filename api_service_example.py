#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kronos模型API服务示例
可以作为微服务集成到你的项目中
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import traceback

# 添加Kronos模型路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import Kronos, KronosTokenizer, KronosPredictor

app = Flask(__name__)

# 全局预测器实例
predictor = None

def init_predictor(model_name="kronos-small", device="auto"):
    """初始化预测器"""
    global predictor
    
    if device == "auto":
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"正在初始化Kronos预测器: {model_name} 到设备: {device}")
    
    try:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained(f"NeoQuasar/{model_name}")
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
        print("预测器初始化成功!")
        return True
    except Exception as e:
        print(f"预测器初始化失败: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'predictor_loaded': predictor is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Predictor not initialized'
            }), 500
        
        data = request.json
        
        # 验证输入数据
        if 'ohlcv_data' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing ohlcv_data field'
            }), 400
        
        # 转换为DataFrame
        df = pd.DataFrame(data['ohlcv_data'])
        
        # 验证必要列
        required_columns = ['timestamps', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({
                'success': False,
                'error': f'Missing required columns: {missing_columns}'
            }), 400
        
        # 转换时间戳
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        
        # 获取预测参数
        lookback = data.get('lookback', 400)
        pred_len = data.get('pred_len', 288)  # 24小时
        temperature = data.get('temperature', 1.0)
        top_p = data.get('top_p', 0.9)
        sample_count = data.get('sample_count', 1)
        
        # 检查数据长度
        if len(df) < lookback:
            return jsonify({
                'success': False,
                'error': f'Insufficient data: need at least {lookback} points, got {len(df)}'
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
        
        # 进行预测
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=future_timestamps,
            pred_len=pred_len,
            T=temperature,
            top_p=top_p,
            sample_count=sample_count,
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
                'volume': float(row['volume']),
                'amount': float(row.get('amount', 0))
            })
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'prediction_count': len(predictions),
            'parameters': {
                'lookback': lookback,
                'pred_len': pred_len,
                'temperature': temperature,
                'top_p': top_p,
                'sample_count': sample_count
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"预测错误: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict/signals', methods=['POST'])
def predict_signals():
    """预测并生成交易信号"""
    try:
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Predictor not initialized'
            }), 500
        
        data = request.json
        
        # 获取预测结果
        pred_response = predict()
        if pred_response[1] != 200:  # 如果预测失败
            return pred_response
        
        pred_data = pred_response[0].get_json()
        if not pred_data['success']:
            return jsonify(pred_data), 500
        
        predictions = pred_data['predictions']
        
        # 计算交易信号
        current_price = data['ohlcv_data'][-1]['close']
        
        # 不同时间点的预测价格
        pred_1h = predictions[11]['close'] if len(predictions) > 11 else predictions[-1]['close']
        pred_6h = predictions[71]['close'] if len(predictions) > 71 else predictions[-1]['close']
        pred_24h = predictions[-1]['close']
        
        # 计算价格变化百分比
        change_1h = (pred_1h - current_price) / current_price * 100
        change_6h = (pred_6h - current_price) / current_price * 100
        change_24h = (pred_24h - current_price) / current_price * 100
        
        # 生成信号
        def generate_signal(change):
            if change > 3:
                return "STRONG_BUY"
            elif change > 1:
                return "BUY"
            elif change < -3:
                return "STRONG_SELL"
            elif change < -1:
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
            'confidence': min(abs(change_24h) / 5.0, 1.0)  # 简单的置信度计算
        }
        
        return jsonify({
            'success': True,
            'signals': signals,
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"信号生成错误: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/models', methods=['GET'])
def get_available_models():
    """获取可用模型列表"""
    models = {
        'kronos-mini': {
            'name': 'Kronos-mini',
            'params': '4.1M',
            'description': 'Lightweight model, suitable for fast prediction'
        },
        'kronos-small': {
            'name': 'Kronos-small',
            'params': '24.7M',
            'description': 'Small model, balanced performance and speed'
        },
        'kronos-base': {
            'name': 'Kronos-base',
            'params': '102.3M',
            'description': 'Base model, provides better prediction quality'
        }
    }
    
    return jsonify({
        'success': True,
        'models': models
    })

if __name__ == '__main__':
    # 初始化预测器
    if init_predictor("kronos-small", "auto"):
        print("启动Kronos预测API服务...")
        print("API端点:")
        print("  GET  /health - 健康检查")
        print("  POST /predict - 价格预测")
        print("  POST /predict/signals - 预测+交易信号")
        print("  GET  /models - 可用模型列表")
        print("\n服务启动在: http://localhost:5000")
        
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("预测器初始化失败，服务无法启动")

