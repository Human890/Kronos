#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kronos API客户端使用示例
展示如何调用Kronos预测API服务
"""

import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta


class KronosAPIClient:
    """Kronos API客户端"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self):
        """健康检查"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {'error': str(e)}
    
    def predict(self, ohlcv_data, lookback=400, pred_len=288, **kwargs):
        """
        预测价格
        
        Args:
            ohlcv_data: OHLCV数据列表
            lookback: 历史数据点数
            pred_len: 预测数据点数
            **kwargs: 其他预测参数
        
        Returns:
            dict: 预测结果
        """
        payload = {
            'ohlcv_data': ohlcv_data,
            'lookback': lookback,
            'pred_len': pred_len,
            **kwargs
        }
        
        try:
            response = requests.post(f"{self.base_url}/predict", json=payload)
            return response.json()
        except Exception as e:
            return {'error': str(e)}
    
    def predict_with_signals(self, ohlcv_data, lookback=400, pred_len=288, **kwargs):
        """
        预测价格并生成交易信号
        
        Args:
            ohlcv_data: OHLCV数据列表
            lookback: 历史数据点数
            pred_len: 预测数据点数
            **kwargs: 其他预测参数
        
        Returns:
            dict: 预测结果和交易信号
        """
        payload = {
            'ohlcv_data': ohlcv_data,
            'lookback': lookback,
            'pred_len': pred_len,
            **kwargs
        }
        
        try:
            response = requests.post(f"{self.base_url}/predict/signals", json=payload)
            return response.json()
        except Exception as e:
            return {'error': str(e)}
    
    def get_available_models(self):
        """获取可用模型列表"""
        try:
            response = requests.get(f"{self.base_url}/models")
            return response.json()
        except Exception as e:
            return {'error': str(e)}


def create_sample_data():
    """创建示例数据"""
    # 生成500个5分钟K线数据点
    dates = pd.date_range('2024-01-01', periods=500, freq='5min')
    np.random.seed(42)
    
    # 生成模拟的OHLCV数据
    base_price = 100
    prices = [base_price]
    for i in range(499):
        change = np.random.normal(0, 0.5)
        new_price = prices[-1] * (1 + change/100)
        prices.append(new_price)
    
    # 创建OHLCV数据
    ohlcv_data = []
    for i in range(500):
        open_price = prices[i]
        close_price = prices[i+1]
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.2))/100)
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.2))/100)
        volume = np.random.randint(100, 1000)
        
        ohlcv_data.append({
            'timestamps': dates[i].isoformat(),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
    
    return ohlcv_data


def demo_api_usage():
    """演示API使用"""
    print("=== Kronos API客户端演示 ===\n")
    
    # 创建客户端
    client = KronosAPIClient("http://localhost:5000")
    
    # 1. 健康检查
    print("1. 健康检查...")
    health = client.health_check()
    print(f"服务状态: {health}")
    print()
    
    if 'error' in health:
        print("❌ API服务不可用，请确保服务已启动")
        return
    
    # 2. 获取可用模型
    print("2. 获取可用模型...")
    models = client.get_available_models()
    if 'error' not in models:
        print("可用模型:")
        for model_id, info in models['models'].items():
            print(f"  - {model_id}: {info['name']} ({info['params']})")
    print()
    
    # 3. 创建示例数据
    print("3. 准备示例数据...")
    ohlcv_data = create_sample_data()
    print(f"生成了 {len(ohlcv_data)} 个数据点")
    print(f"时间范围: {ohlcv_data[0]['timestamps']} 到 {ohlcv_data[-1]['timestamps']}")
    print()
    
    # 4. 进行预测
    print("4. 进行价格预测...")
    prediction_result = client.predict(ohlcv_data, lookback=400, pred_len=288)
    
    if 'error' in prediction_result:
        print(f"❌ 预测失败: {prediction_result['error']}")
        return
    
    if prediction_result['success']:
        predictions = prediction_result['predictions']
        print(f"✅ 预测成功! 生成了 {len(predictions)} 个预测点")
        print("预测结果预览:")
        for i, pred in enumerate(predictions[:5]):  # 显示前5个
            print(f"  {i+1}. {pred['timestamp']}: 开盘={pred['open']:.2f}, 收盘={pred['close']:.2f}")
        print("  ...")
        print()
    else:
        print(f"❌ 预测失败: {prediction_result.get('error', 'Unknown error')}")
        return
    
    # 5. 预测并生成交易信号
    print("5. 预测并生成交易信号...")
    signals_result = client.predict_with_signals(ohlcv_data, lookback=400, pred_len=288)
    
    if 'error' in signals_result:
        print(f"❌ 信号生成失败: {signals_result['error']}")
        return
    
    if signals_result['success']:
        signals = signals_result['signals']
        print("✅ 交易信号生成成功!")
        print(f"当前价格: {signals['current_price']:.2f}")
        print("预测和信号:")
        for timeframe, data in signals['predictions'].items():
            print(f"  {timeframe}: 价格={data['price']:.2f}, 变化={data['change_pct']:+.2f}%, 信号={data['signal']}")
        print(f"整体建议: {signals['overall_signal']}")
        print(f"置信度: {signals['confidence']:.2f}")
        print()
    else:
        print(f"❌ 信号生成失败: {signals_result.get('error', 'Unknown error')}")
        return
    
    print("=== 演示完成 ===")


def real_time_prediction_example():
    """实时预测示例"""
    print("=== 实时预测示例 ===\n")
    
    client = KronosAPIClient("http://localhost:5000")
    
    # 模拟实时数据流
    print("模拟实时数据流...")
    
    # 获取最新数据 (这里使用示例数据)
    ohlcv_data = create_sample_data()
    
    # 每5分钟进行一次预测 (这里只是演示，实际使用时需要定时调用)
    print("进行实时预测...")
    
    result = client.predict_with_signals(ohlcv_data, lookback=400, pred_len=288)
    
    if result.get('success'):
        signals = result['signals']
        current_price = signals['current_price']
        overall_signal = signals['overall_signal']
        confidence = signals['confidence']
        
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 当前价格: {current_price:.2f}")
        print(f"📊 交易信号: {overall_signal}")
        print(f"🎯 置信度: {confidence:.2f}")
        
        # 根据信号给出建议
        if overall_signal in ["STRONG_BUY", "BUY"]:
            print("💡 建议: 考虑买入")
        elif overall_signal in ["STRONG_SELL", "SELL"]:
            print("💡 建议: 考虑卖出")
        else:
            print("💡 建议: 持有观望")
    else:
        print(f"❌ 预测失败: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    # 运行演示
    demo_api_usage()
    
    print("\n" + "="*50 + "\n")
    
    # 运行实时预测示例
    real_time_prediction_example()

