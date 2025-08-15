from fastapi import FastAPI, Request, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import re
import os
import json
import csv
from datetime import datetime
from typing import List, Dict, Optional, Literal
from collections import deque
import asyncio
import threading
import time
import pandas as pd
import subprocess
import mimetypes
import numpy as np
import math
from scipy import signal
from scipy.fft import fft, fftfreq
import io
import logging
from enum import Enum

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 初始化 FastAPI
app = FastAPI(docs_url="/swagger", redoc_url=None, title="固定相位滚动波形电力系统实时监控平台")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 目录配置
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = "uploaded_csv"
PICTURE_DIR = os.path.join(BASE_DIR, "pictures")
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# 确保目录存在
for directory in [UPLOAD_DIR, PICTURE_DIR, VIDEO_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# 静态文件挂载
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/api/download_image", StaticFiles(directory=PICTURE_DIR), name="download_image")
app.mount("/api/download_video", StaticFiles(directory=VIDEO_DIR), name="download_video")
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

templates = Jinja2Templates(directory="templates")

# ==============================================================================
# 电力系统类型枚举
# ==============================================================================
class PowerType(str, Enum):
    DC = "dc"              # a0 - 直流模式
    SINGLE_PHASE = "single_phase"  # a1 - 单相模式
    THREE_PHASE = "three_phase"    # a2 - 三相模式

class ClientStatus(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    INACTIVE = "inactive"
    REGISTERED = "registered"

# ==============================================================================
# 固定相位滚动波形生成器类 - 完全修复版本
# ==============================================================================
class FixedPhaseScrollingWaveformGenerator:
    """固定相位滚动波形生成器 - 修正版：直接计算完整波形值而非振幅"""
    
    def __init__(self):
        self.sampling_rate = 20000.0  # 20kHz采样率
        self.frequency = 50.0         # 基波频率50Hz
        self.points_per_cycle = 400   # 每周期固定400个点
        self.scroll_window_size = 1000  # 滚动窗口大小，最大2000个点
        self.max_window_size = 2000   # 最大窗口限制
        
        # 客户端滚动缓冲区 - 存储完整波形值
        self.client_scroll_buffers: Dict[str, Dict] = {}
        
        logger.info("✅ 初始化固定相位滚动波形生成器 - 修正版")
        logger.info(f"   - 每周期点数: {self.points_per_cycle}")
        logger.info(f"   - 相位计算: (连续位置 % {self.points_per_cycle}) / {self.points_per_cycle} * 2π")
        logger.info(f"   - 滚动窗口: {self.scroll_window_size}")
        
    def _calculate_fixed_phase(self, continuous_position: int) -> float:
        """计算固定相位：基于连续位置的固定相位系统"""
        return (continuous_position % self.points_per_cycle) / self.points_per_cycle * 2 * np.pi
        
    def _calculate_three_phase_fixed_phases(self, continuous_position: int) -> Dict[str, float]:
        """计算三相固定相位"""
        base_phase = self._calculate_fixed_phase(continuous_position)
        return {
            'a': base_phase,                    # A相：0°
            'b': base_phase - 2*np.pi/3,       # B相：-120°
            'c': base_phase - 4*np.pi/3        # C相：-240°
        }
        
    def initialize_client_buffer(self, client_id: str, power_type: PowerType):
        """初始化客户端滚动缓冲区"""
        self.client_scroll_buffers[client_id] = {
            'power_type': power_type,
            'continuous_position': 0,  # 连续位置计数器 - 关键修正
            'window_size': self.scroll_window_size,
            
            # 波形值缓冲区（固定大小，循环使用）- 存储完整波形值而非振幅
            'voltage_waveform': np.zeros(self.scroll_window_size),
            'current_waveform': np.zeros(self.scroll_window_size),
            'voltage_a_waveform': np.zeros(self.scroll_window_size),
            'voltage_b_waveform': np.zeros(self.scroll_window_size),
            'voltage_c_waveform': np.zeros(self.scroll_window_size),
            'current_a_waveform': np.zeros(self.scroll_window_size),
            'current_b_waveform': np.zeros(self.scroll_window_size),
            'current_c_waveform': np.zeros(self.scroll_window_size),
            
            # 最新的RMS值，用于计算峰值
            'latest_rms': {
                'voltage': 0.01,
                'current': 10.0,
                'voltage_a': 0.01,
                'voltage_b': 0.01,
                'voltage_c': 0.01,
                'current_a': 10.0,
                'current_b': 10.0,
                'current_c': 10.0,
            },
            
            'last_update_time': time.time(),
            'is_initialized': True
        }
        
        logger.info(f"✅ 初始化客户端 {client_id} 固定相位缓冲区 - 模式: {power_type.value}")
        
    def generate_smooth_scroll_data(self, client_id: str, latest_data: dict, num_new_points: int = 20) -> Dict:
        """生成固定相位的平滑滚动波形数据 - 修正版"""
        try:
            if client_id not in self.client_scroll_buffers:
                # 从最新数据检测电力类型
                power_type = self._detect_power_type_from_data(latest_data)
                self.initialize_client_buffer(client_id, power_type)
                
            buffer_info = self.client_scroll_buffers[client_id]
            power_type = buffer_info['power_type']
            
            # 更新最新RMS值
            self._update_latest_rms_values(buffer_info, latest_data, power_type)
            
            # 生成新的波形值并更新缓冲区 - 关键修正
            self._generate_and_update_waveform_data(buffer_info, num_new_points)
            
            # 生成固定相位波形数据
            waveform_data = self._generate_fixed_phase_waveform(buffer_info, power_type)
            
            buffer_info['last_update_time'] = time.time()
            
            logger.debug(f"📊 生成固定相位波形: {client_id}, 新增点数: {num_new_points}")
            
            return waveform_data
                
        except Exception as e:
            logger.error(f"❌ 固定相位波形生成失败 {client_id}: {e}")
            return self._generate_empty_scroll_data(power_type)
            
    def _detect_power_type_from_data(self, data: dict) -> PowerType:
        """从数据检测电力类型"""
        if 'voltage_a' in data or 'voltage_b' in data or 'voltage_c' in data:
            return PowerType.THREE_PHASE
        elif 'voltage' in data:
            voltage = data.get('voltage', 0)
            if isinstance(voltage, (int, float)) and abs(voltage) > 5:
                return PowerType.SINGLE_PHASE
            return PowerType.SINGLE_PHASE
        else:
            return PowerType.SINGLE_PHASE
            
    def _update_latest_rms_values(self, buffer_info: dict, latest_data: dict, power_type: PowerType):
        """更新最新的RMS值"""
        rms_dict = buffer_info['latest_rms']
        
        if power_type == PowerType.DC:
            rms_dict['voltage'] = float(latest_data.get('voltage', rms_dict['voltage']))
            rms_dict['current'] = float(latest_data.get('current', rms_dict['current']))
            
        elif power_type == PowerType.THREE_PHASE:
            rms_dict['voltage_a'] = float(latest_data.get('voltage_a', rms_dict['voltage_a']))
            rms_dict['voltage_b'] = float(latest_data.get('voltage_b', rms_dict['voltage_b']))
            rms_dict['voltage_c'] = float(latest_data.get('voltage_c', rms_dict['voltage_c']))
            rms_dict['current_a'] = float(latest_data.get('current_a', rms_dict['current_a']))
            rms_dict['current_b'] = float(latest_data.get('current_b', rms_dict['current_b']))
            rms_dict['current_c'] = float(latest_data.get('current_c', rms_dict['current_c']))
            
        else:  # SINGLE_PHASE
            rms_dict['voltage'] = float(latest_data.get('voltage', rms_dict['voltage']))
            rms_dict['current'] = float(latest_data.get('current', rms_dict['current']))
            
    def _generate_and_update_waveform_data(self, buffer_info: dict, num_new_points: int):
        """生成新的波形值并更新缓冲区 - 核心修正函数"""
        power_type = buffer_info['power_type']
        rms_values = buffer_info['latest_rms']
        continuous_position = buffer_info['continuous_position']
        
        if power_type == PowerType.DC:
            # 直流：生成带微小噪声的直流值
            base_voltage = rms_values['voltage']
            base_current = rms_values['current']
            
            for i in range(num_new_points):
                # 直流值加微小波动
                voltage_noise = np.random.normal(0, abs(base_voltage) * 0.01)
                current_noise = np.random.normal(0, abs(base_current) * 0.01)
                
                new_voltage_value = base_voltage + voltage_noise
                new_current_value = base_current + current_noise
                
                # 向右滚动
                buffer_info['voltage_waveform'] = np.roll(buffer_info['voltage_waveform'], -1)
                buffer_info['current_waveform'] = np.roll(buffer_info['current_waveform'], -1)
                
                # 在最右边添加新数据
                buffer_info['voltage_waveform'][-1] = new_voltage_value
                buffer_info['current_waveform'][-1] = new_current_value
                
        elif power_type == PowerType.SINGLE_PHASE:
            # 单相：生成完整的正弦波形值
            voltage_peak = rms_values['voltage'] * np.sqrt(2)
            current_peak = rms_values['current'] * np.sqrt(2)
            power_factor_phase = -np.pi/6  # 功率因数相位差
            
            for i in range(num_new_points):
                # 计算当前连续位置的相位
                current_pos = continuous_position + i
                voltage_phase = self._calculate_fixed_phase(current_pos)
                current_phase = voltage_phase + power_factor_phase
                
                # 计算完整的波形值（包含谐波和噪声）
                voltage_value = voltage_peak * np.sin(voltage_phase)
                voltage_value += voltage_peak * 0.05 * np.sin(3 * voltage_phase)  # 3次谐波
                voltage_value += voltage_peak * 0.02 * np.sin(5 * voltage_phase)  # 5次谐波
                voltage_value += np.random.normal(0, voltage_peak * 0.005)        # 噪声
                
                current_value = current_peak * np.sin(current_phase)
                current_value += current_peak * 0.03 * np.sin(3 * current_phase)
                current_value += current_peak * 0.015 * np.sin(5 * current_phase)
                current_value += np.random.normal(0, current_peak * 0.005)
                
                # 向右滚动
                buffer_info['voltage_waveform'] = np.roll(buffer_info['voltage_waveform'], -1)
                buffer_info['current_waveform'] = np.roll(buffer_info['current_waveform'], -1)
                
                # 在最右边添加新的波形值
                buffer_info['voltage_waveform'][-1] = voltage_value
                buffer_info['current_waveform'][-1] = current_value
                
        elif power_type == PowerType.THREE_PHASE:
            # 三相：生成三相正弦波形值
            voltage_peaks = {
                'a': rms_values['voltage_a'] * np.sqrt(2),
                'b': rms_values['voltage_b'] * np.sqrt(2),
                'c': rms_values['voltage_c'] * np.sqrt(2)
            }
            current_peaks = {
                'a': rms_values['current_a'] * np.sqrt(2),
                'b': rms_values['current_b'] * np.sqrt(2),
                'c': rms_values['current_c'] * np.sqrt(2)
            }
            power_factor_phase = -np.pi/6
            phase_offsets = [0, -2*np.pi/3, -4*np.pi/3]  # A, B, C相位差
            
            for i in range(num_new_points):
                current_pos = continuous_position + i
                base_phase = self._calculate_fixed_phase(current_pos)
                
                for phase_idx, phase in enumerate(['a', 'b', 'c']):
                    voltage_phase = base_phase + phase_offsets[phase_idx]
                    current_phase = voltage_phase + power_factor_phase
                    
                    # 计算完整的波形值
                    voltage_value = voltage_peaks[phase] * np.sin(voltage_phase)
                    voltage_value += voltage_peaks[phase] * 0.03 * np.sin(3 * voltage_phase)
                    voltage_value += voltage_peaks[phase] * 0.02 * np.sin(5 * voltage_phase)
                    voltage_value += np.random.normal(0, voltage_peaks[phase] * 0.005)
                    
                    current_value = current_peaks[phase] * np.sin(current_phase)
                    current_value += current_peaks[phase] * 0.02 * np.sin(3 * current_phase)
                    current_value += current_peaks[phase] * 0.015 * np.sin(5 * current_phase)
                    current_value += np.random.normal(0, current_peaks[phase] * 0.005)
                    
                    # 向右滚动
                    buffer_info[f'voltage_{phase}_waveform'] = np.roll(buffer_info[f'voltage_{phase}_waveform'], -1)
                    buffer_info[f'current_{phase}_waveform'] = np.roll(buffer_info[f'current_{phase}_waveform'], -1)
                    
                    # 在最右边添加新的波形值
                    buffer_info[f'voltage_{phase}_waveform'][-1] = voltage_value
                    buffer_info[f'current_{phase}_waveform'][-1] = current_value
        
        # 更新连续位置计数器 - 关键
        buffer_info['continuous_position'] += num_new_points
    
    def _generate_fixed_phase_waveform(self, buffer_info: dict, power_type: PowerType) -> Dict:
        """生成固定相位波形数据 - 直接返回已计算的波形值"""
        window_size = buffer_info['window_size']
        
        if power_type == PowerType.DC:
            return self._generate_dc_fixed_waveform(buffer_info)
        elif power_type == PowerType.SINGLE_PHASE:
            return self._generate_single_phase_fixed_waveform(buffer_info)
        elif power_type == PowerType.THREE_PHASE:
            return self._generate_three_phase_fixed_waveform(buffer_info)
        else:
            return self._generate_empty_scroll_data(power_type)
    
    def _generate_dc_fixed_waveform(self, buffer_info: dict) -> Dict:
        """生成直流固定波形 - 直接使用波形值"""
        window_size = buffer_info['window_size']
        
        # 直接使用已计算的波形值
        voltage_data = [{"x": i, "y": float(buffer_info['voltage_waveform'][i])} 
                       for i in range(window_size)]
        current_data = [{"x": i, "y": float(buffer_info['current_waveform'][i])} 
                       for i in range(window_size)]
        
        return {
            "voltage": voltage_data,
            "current": current_data,
            "new_points_count": 20,
            "buffer_size": window_size,
            "power_type": "dc",
            "phase_system": "fixed_dc_corrected",
            "scroll_info": {
                "window_size": window_size,
                "phase_type": "fixed_position_based_corrected",
                "voltage_rms": buffer_info['latest_rms']['voltage'],
                "current_rms": buffer_info['latest_rms']['current'],
                "continuous_position": buffer_info['continuous_position']
            }
        }
    
    def _generate_single_phase_fixed_waveform(self, buffer_info: dict) -> Dict:
        """生成单相固定相位波形 - 直接使用波形值"""
        window_size = buffer_info['window_size']
        
        # 直接使用已计算的波形值
        voltage_data = [{"x": i, "y": float(buffer_info['voltage_waveform'][i])} 
                       for i in range(window_size)]
        current_data = [{"x": i, "y": float(buffer_info['current_waveform'][i])} 
                       for i in range(window_size)]
        
        return {
            "voltage": voltage_data,
            "current": current_data,
            "new_points_count": 20,
            "buffer_size": window_size,
            "power_type": "single_phase",
            "phase_system": "fixed_position_based_corrected",
            "scroll_info": {
                "window_size": window_size,
                "points_per_cycle": self.points_per_cycle,
                "frequency": self.frequency,
                "phase_type": "fixed_position_based_corrected",
                "voltage_rms": buffer_info['latest_rms']['voltage'],
                "current_rms": buffer_info['latest_rms']['current'],
                "continuous_position": buffer_info['continuous_position'],
                "power_factor_angle": -30.0
            }
        }
    
    def _generate_three_phase_fixed_waveform(self, buffer_info: dict) -> Dict:
        """生成三相固定相位波形 - 直接使用波形值"""
        window_size = buffer_info['window_size']
        
        result = {
            "new_points_count": 20,
            "buffer_size": window_size,
            "power_type": "three_phase",
            "phase_system": "fixed_position_based_corrected",
            "scroll_info": {
                "window_size": window_size,
                "points_per_cycle": self.points_per_cycle,
                "frequency": self.frequency,
                "phase_type": "fixed_position_based_corrected",
                "rms_values": {
                    "voltage": [buffer_info['latest_rms']['voltage_a'], 
                               buffer_info['latest_rms']['voltage_b'], 
                               buffer_info['latest_rms']['voltage_c']],
                    "current": [buffer_info['latest_rms']['current_a'], 
                               buffer_info['latest_rms']['current_b'], 
                               buffer_info['latest_rms']['current_c']]
                },
                "power_factor_angle": -30.0,
                "phase_relationships": "A:0°, B:-120°, C:-240°",
                "continuous_position": buffer_info['continuous_position']
            }
        }
        
        phases = ['a', 'b', 'c']
        
        for phase in phases:
            # 直接使用已计算的波形值
            voltage_data = [{"x": i, "y": float(buffer_info[f'voltage_{phase}_waveform'][i])} 
                           for i in range(window_size)]
            current_data = [{"x": i, "y": float(buffer_info[f'current_{phase}_waveform'][i])} 
                           for i in range(window_size)]
            
            result[f"voltage_{phase}"] = voltage_data
            result[f"current_{phase}"] = current_data
        
        return result

    def _generate_empty_scroll_data(self, power_type: PowerType) -> Dict:
        """生成空的滚动数据"""
        if power_type == PowerType.DC:
            return {
                "voltage": [],
                "current": [],
                "new_points_count": 0,
                "buffer_size": 0,
                "power_type": "dc",
                "phase_system": "fixed_dc_corrected"
            }
        elif power_type == PowerType.THREE_PHASE:
            result = {
                "new_points_count": 0,
                "buffer_size": 0,
                "power_type": "three_phase",
                "phase_system": "fixed_position_based_corrected"
            }
            for phase in ['a', 'b', 'c']:
                result[f"voltage_{phase}"] = []
                result[f"current_{phase}"] = []
            return result
        else:
            return {
                "voltage": [],
                "current": [],
                "new_points_count": 0,
                "buffer_size": 0,
                "power_type": "single_phase",
                "phase_system": "fixed_position_based_corrected"
            }

    def adjust_window_size(self, client_id: str, new_size: int):
        """调整窗口大小"""
        if client_id in self.client_scroll_buffers:
            new_size = min(new_size, self.max_window_size)  # 限制最大窗口
            buffer_info = self.client_scroll_buffers[client_id]
            old_size = buffer_info['window_size']
            
            if new_size != old_size:
                # 调整所有波形缓冲区大小
                for key in buffer_info:
                    if key.endswith('_waveform'):
                        old_data = buffer_info[key]
                        if new_size > old_size:
                            # 扩大：在前面填充零
                            buffer_info[key] = np.concatenate([np.zeros(new_size - old_size), old_data])
                        else:
                            # 缩小：保留最新的数据
                            buffer_info[key] = old_data[-new_size:]
                
                buffer_info['window_size'] = new_size
                logger.info(f"📏 调整客户端 {client_id} 窗口大小: {old_size} -> {new_size}")

# ==============================================================================
# 标准波形生成器类（用于分析接口）
# ==============================================================================
class WaveformGenerator:
    """电力系统波形生成器"""
    
    def __init__(self):
        self.sampling_rate = 20000.0  # 20kHz采样率
        self.frequency = 50.0         # 基波频率50Hz
        self.points_per_cycle = 400   # 每周期固定400个点
        
    def calculate_time_for_points(self, num_points: int) -> float:
        """计算指定点数对应的时间长度（秒）"""
        cycles_needed = num_points / self.points_per_cycle
        return cycles_needed / self.frequency
        
    def generate_dc_waveform(self, voltage: float, current: float, num_points: int = 1000) -> Dict:
        """生成直流波形"""
        time_duration = self.calculate_time_for_points(num_points)
        time_points = np.linspace(0, time_duration, num_points)
        
        # 直流波形 - 添加少量噪声模拟真实情况
        noise_voltage = np.random.normal(0, abs(voltage) * 0.02, num_points)  # 2%噪声
        noise_current = np.random.normal(0, abs(current) * 0.02, num_points)
        
        voltage_waveform = np.full(num_points, voltage) + noise_voltage
        current_waveform = np.full(num_points, current) + noise_current
        
        return {
            "voltage": [{"x": i, "y": float(v)} for i, v in enumerate(voltage_waveform)],
            "current": [{"x": i, "y": float(c)} for i, c in enumerate(current_waveform)],
            "time_points": time_points.tolist(),
            "cycles": num_points / self.points_per_cycle,
            "frequency": 0,  # DC
            "sampling_info": {
                "points_per_cycle": self.points_per_cycle,
                "total_cycles": num_points / self.points_per_cycle,
                "time_duration": time_duration
            }
        }
    
    def generate_single_phase_waveform(self, voltage_rms: float, current_rms: float, 
                                     num_points: int = 1000, phase_offset: float = 0) -> Dict:
        """生成单相波形"""
        time_duration = self.calculate_time_for_points(num_points)
        time_points = np.linspace(0, time_duration, num_points)
        
        # 计算峰值（RMS * √2）
        voltage_peak = voltage_rms * np.sqrt(2)
        current_peak = current_rms * np.sqrt(2)
        
        # 生成基波正弦波
        omega = 2 * np.pi * self.frequency
        voltage_waveform = voltage_peak * np.sin(omega * time_points)
        current_waveform = current_peak * np.sin(omega * time_points + phase_offset)
        
        # 添加3次谐波（5%）和5次谐波（2%）
        voltage_waveform += voltage_peak * 0.05 * np.sin(3 * omega * time_points)
        voltage_waveform += voltage_peak * 0.02 * np.sin(5 * omega * time_points)
        
        current_waveform += current_peak * 0.03 * np.sin(3 * omega * time_points + phase_offset)
        current_waveform += current_peak * 0.015 * np.sin(5 * omega * time_points + phase_offset)
        
        # 添加随机噪声（1%）
        voltage_noise = np.random.normal(0, voltage_peak * 0.01, num_points)
        current_noise = np.random.normal(0, current_peak * 0.01, num_points)
        
        voltage_waveform += voltage_noise
        current_waveform += current_noise
        
        return {
            "voltage": [{"x": i, "y": float(v)} for i, v in enumerate(voltage_waveform)],
            "current": [{"x": i, "y": float(c)} for i, c in enumerate(current_waveform)],
            "time_points": time_points.tolist(),
            "cycles": num_points / self.points_per_cycle,
            "frequency": self.frequency,
            "sampling_info": {
                "points_per_cycle": self.points_per_cycle,
                "total_cycles": num_points / self.points_per_cycle,
                "time_duration": time_duration,
                "voltage_rms": voltage_rms,
                "current_rms": current_rms,
                "voltage_peak": voltage_peak,
                "current_peak": current_peak
            }
        }
    
    def generate_three_phase_waveform(self, voltage_a_rms: float, voltage_b_rms: float, voltage_c_rms: float,
                                    current_a_rms: float, current_b_rms: float, current_c_rms: float,
                                    num_points: int = 1000, phase_offsets: List[float] = None) -> Dict:
        """生成三相波形"""
        if phase_offsets is None:
            # 标准三相相位差：A相0°，B相-120°，C相-240°
            phase_offsets = [0, -2*np.pi/3, -4*np.pi/3]
        
        # 计算时间轴，确保精确的周期数
        time_duration = self.calculate_time_for_points(num_points)
        time_points = np.linspace(0, time_duration, num_points)
        
        # 存储RMS值用于验证
        voltage_rms_values = [voltage_a_rms, voltage_b_rms, voltage_c_rms]
        current_rms_values = [current_a_rms, current_b_rms, current_c_rms]
        
        # 计算峰值
        voltage_peaks = [v_rms * np.sqrt(2) for v_rms in voltage_rms_values]
        current_peaks = [i_rms * np.sqrt(2) for i_rms in current_rms_values]
        
        waveforms = {}
        phase_names = ['a', 'b', 'c']
        omega = 2 * np.pi * self.frequency
        
        # 功率因数角度（假设30度滞后）
        power_factor_angle = -np.pi/6
        
        sampling_info = {
            "points_per_cycle": self.points_per_cycle,
            "total_cycles": num_points / self.points_per_cycle,
            "time_duration": time_duration,
            "frequency": self.frequency,
            "phase_offsets_deg": [np.degrees(offset) for offset in phase_offsets],
            "rms_values": {
                "voltage": voltage_rms_values,
                "current": current_rms_values
            },
            "peak_values": {
                "voltage": voltage_peaks,
                "current": current_peaks
            }
        }
        
        for i, phase in enumerate(phase_names):
            # 生成电压波形
            voltage_waveform = voltage_peaks[i] * np.sin(omega * time_points + phase_offsets[i])
            
            # 生成电流波形（包含功率因数相位差）
            current_phase_offset = phase_offsets[i] + power_factor_angle
            current_waveform = current_peaks[i] * np.sin(omega * time_points + current_phase_offset)
            
            # 添加谐波成分
            voltage_waveform += voltage_peaks[i] * 0.03 * np.sin(3 * omega * time_points + 3 * phase_offsets[i])
            current_waveform += current_peaks[i] * 0.02 * np.sin(3 * omega * time_points + 3 * current_phase_offset)
            
            voltage_waveform += voltage_peaks[i] * 0.02 * np.sin(5 * omega * time_points + 5 * phase_offsets[i])
            current_waveform += current_peaks[i] * 0.015 * np.sin(5 * omega * time_points + 5 * current_phase_offset)
            
            # 添加随机噪声（0.5%）
            voltage_noise = np.random.normal(0, voltage_peaks[i] * 0.01, num_points)
            current_noise = np.random.normal(0, current_peaks[i] * 0.01, num_points)
            
            voltage_waveform += voltage_noise
            current_waveform += current_noise
            
            # 存储波形数据
            waveforms[f"voltage_{phase}"] = [{"x": j, "y": float(v)} for j, v in enumerate(voltage_waveform)]
            waveforms[f"current_{phase}"] = [{"x": j, "y": float(c)} for j, c in enumerate(current_waveform)]
        
        waveforms["time_points"] = time_points.tolist()
        waveforms["cycles"] = num_points / self.points_per_cycle
        waveforms["frequency"] = self.frequency
        waveforms["sampling_info"] = sampling_info
        
        return waveforms

# ==============================================================================
# 电力数据连接管理器 - 使用固定相位系统
# ==============================================================================
class OptimizedPowerConnectionManager:
    """优化的电力系统连接管理器 - 支持固定相位滚动波形"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.data_source_clients: Dict[str, Dict] = {}
        self.web_clients: Dict[str, Dict] = {}
        self.client_data_files: Dict[str, str] = {}
        self.realtime_data_buffer: Dict[str, deque] = {}
        self.MAX_BUFFER_SIZE = 1000
        self.connection_health: Dict[str, Dict] = {}
        
        # 数据缓存
        self.realtime_cache: Dict[str, Dict] = {}
        
        # 波形生成器
        self.waveform_generator = WaveformGenerator()
        self.scrolling_waveform_generator = FixedPhaseScrollingWaveformGenerator()  # 使用修正的固定相位生成器
        
        # 工作模式映射
        self.work_mode_map = {
            "a0": PowerType.DC,
            "a1": PowerType.SINGLE_PHASE,
            "a2": PowerType.THREE_PHASE
        }
        
        # 滚动波形更新任务 - 增强管理
        self.scroll_update_tasks: Dict[str, asyncio.Task] = {}
        self.task_monitoring: Dict[str, Dict] = {}
        
        # 连接监控任务
        self._start_connection_monitor()
        self._start_task_monitor()
        
        logger.info("🚀 电力连接管理器已启动 - 修正的固定相位滚动系统")
    
    def _start_connection_monitor(self):
        """启动连接监控任务"""
        async def monitor_connections():
            while True:
                try:
                    await self._check_connection_health()
                    await asyncio.sleep(30)
                except Exception as e:
                    logger.error(f"Connection monitor error: {e}")
                    await asyncio.sleep(5)
                    
        asyncio.create_task(monitor_connections())

    def _start_task_monitor(self):
        """启动任务监控"""
        async def monitor_tasks():
            while True:
                try:
                    await self._check_scroll_tasks()
                    await asyncio.sleep(10)
                except Exception as e:
                    logger.error(f"Task monitor error: {e}")
                    await asyncio.sleep(5)
                    
        asyncio.create_task(monitor_tasks())

    async def _check_scroll_tasks(self):
        """检查滚动任务状态"""
        current_time = time.time()
        failed_tasks = []
        
        for client_id, task in self.scroll_update_tasks.items():
            if task.done():
                try:
                    # 获取任务异常
                    exception = task.exception()
                    if exception:
                        logger.error(f"Scroll task for {client_id} failed: {exception}")
                    else:
                        logger.info(f"Scroll task for {client_id} completed normally")
                except Exception as e:
                    logger.error(f"Error checking task for {client_id}: {e}")
                
                failed_tasks.append(client_id)
        
        # 重启失败的任务
        for client_id in failed_tasks:
            logger.warning(f"Restarting scroll task for client {client_id}")
            self.scroll_update_tasks.pop(client_id, None)
            
            # 检查客户端是否仍需要滚动监控
            if (client_id in self.data_source_clients and 
                self.data_source_clients[client_id].get("scroll_monitoring", False)):
                await self._start_scroll_task(client_id)

    async def _check_connection_health(self):
        """检查连接健康状态"""
        current_time = datetime.now()
        disconnected_clients = []
        
        for client_id, info in self.data_source_clients.items():
            if info.get("last_update"):
                time_diff = (current_time - info["last_update"]).total_seconds()
                
                if time_diff > 120:
                    if info["status"] != ClientStatus.INACTIVE:
                        info["status"] = ClientStatus.INACTIVE
                        logger.warning(f"Client {client_id} marked as inactive")
                        
                elif time_diff > 300:
                    disconnected_clients.append(client_id)
        
        for client_id in disconnected_clients:
            self.disconnect(client_id)
            logger.info(f"Cleaned up disconnected client: {client_id}")
        
        await self.broadcast_client_list()

    async def connect(self, client_id: str, websocket: WebSocket, client_type: str = "data_source"):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
        if client_type == "data_source":
            self.data_source_clients[client_id] = {
                "connected_time": datetime.now(),
                "data_count": 0,
                "last_update": None,
                "status": ClientStatus.REGISTERED,
                "latest_data": None,
                "client_type": "adaptive_sensor",
                "description": "",
                "power_type": PowerType.SINGLE_PHASE,
                "auto_detected": False,
                "work_mode": None,
                "scroll_monitoring": False
            }
            
            self.realtime_data_buffer[client_id] = deque(maxlen=self.MAX_BUFFER_SIZE)
            self.realtime_cache[client_id] = {}
            
            self.connection_health[client_id] = {
                "ping_count": 0,
                "pong_count": 0,
                "last_ping": None,
                "last_pong": None,
                "latency": 0
            }
            
            logger.info(f"Data source client connected: {client_id}")
            
        else:
            self.web_clients[client_id] = {
                "connected_time": datetime.now(),
                "monitoring_client": None,
                "status": ClientStatus.CONNECTED,
                "last_data_index": 0,
                "scroll_mode": False
            }
            
            logger.info(f"Web client connected: {client_id}")
        
        await self.broadcast_client_list()

    async def _create_client_data_file(self, client_id: str, power_type: PowerType):
        """根据电力类型创建相应的数据文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if power_type == PowerType.DC:
            power_prefix = "dc"
            header = ['timestamp', 'time_seq', 'voltage', 'current']
        elif power_type == PowerType.THREE_PHASE:
            power_prefix = "threephase"
            header = ['timestamp', 'time_seq', 'voltage_a', 'voltage_b', 'voltage_c', 'current_a', 'current_b', 'current_c']
        else:
            power_prefix = "singlephase"
            header = ['timestamp', 'time_seq', 'voltage', 'current']
        
        filename = f"{power_prefix}_client_{client_id}_{timestamp}.csv"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        
        self.client_data_files[client_id] = filename
        logger.info(f"Created {power_type.value} data file for client {client_id}: {filename}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            self.active_connections.pop(client_id)
            
        if client_id in self.data_source_clients:
            self.data_source_clients[client_id]["status"] = ClientStatus.DISCONNECTED
            self.data_source_clients[client_id]["disconnected_time"] = datetime.now()
            
        if client_id in self.web_clients:
            self.web_clients.pop(client_id)
            
        if client_id in self.connection_health:
            self.connection_health.pop(client_id)
            
        if client_id in self.realtime_cache:
            self.realtime_cache.pop(client_id)
            
        # 停止滚动更新任务
        if client_id in self.scroll_update_tasks:
            self.scroll_update_tasks[client_id].cancel()
            del self.scroll_update_tasks[client_id]
            
        if client_id in self.task_monitoring:
            del self.task_monitoring[client_id]
            
        logger.info(f"Client disconnected: {client_id}")

    async def send_personal_message(self, message: dict, client_id: str):
        websocket = self.active_connections.get(client_id)
        if websocket:
            try:
                await websocket.send_json(message)
                return True
            except Exception as e:
                logger.error(f"Failed to send message to {client_id}: {e}")
                self.disconnect(client_id)
                return False
        return False

    async def broadcast_to_web_clients(self, message: dict):
        """只向Web客户端广播消息"""
        disconnected_clients = []
        for web_client_id in self.web_clients.keys():
            success = await self.send_personal_message(message, web_client_id)
            if not success:
                disconnected_clients.append(web_client_id)
        
        for client_id in disconnected_clients:
            self.disconnect(client_id)

    async def broadcast_client_list(self):
        """广播数据源客户端列表给所有Web界面"""
        client_list = []
        current_time = datetime.now()
        
        for client_id, info in self.data_source_clients.items():
            if info.get("last_update"):
                time_diff = (current_time - info["last_update"]).total_seconds()
                is_active = time_diff < 60
                status = ClientStatus.CONNECTED if is_active else ClientStatus.INACTIVE
            else:
                status = info.get("status", ClientStatus.REGISTERED)
            
            client_list.append({
                "id": client_id,
                "connected_time": info["connected_time"].strftime("%Y-%m-%d %H:%M:%S"),
                "data_count": info["data_count"],
                "last_update": info["last_update"].strftime("%H:%M:%S") if info["last_update"] else "无",
                "status": status.value,
                "filename": self.client_data_files.get(client_id, ""),
                "latest_data": info.get("latest_data"),
                "buffer_size": len(self.realtime_data_buffer.get(client_id, [])),
                "client_type": info.get("client_type", "unknown"),
                "description": info.get("description", ""),
                "power_type": info.get("power_type", PowerType.SINGLE_PHASE).value,
                "auto_detected": info.get("auto_detected", False),
                "work_mode": info.get("work_mode"),
                "scroll_monitoring": info.get("scroll_monitoring", False)
            })
        
        message = {
            "type": "client_list_update",
            "clients": client_list
        }
        
        await self.broadcast_to_web_clients(message)

    async def handle_stream_data(self, client_id: str, stream_data: dict):
        """处理数据流 - 集成修正的固定相位滚动效果"""
        try:
            data_points = stream_data.get('data', [])
            seq_num = stream_data.get('seq', 0)
            work_mode = stream_data.get('work_mode')
            
            if not data_points:
                return False
            
            logger.debug(f"Processing data stream from client {client_id}, work_mode: {work_mode}, points: {len(data_points)}")
            
            # 检查工作模式初始化
            if work_mode and work_mode in self.work_mode_map:
                power_type = self.work_mode_map[work_mode]
                logger.info(f"Client {client_id} initialized with work mode {work_mode} -> {power_type.value}")
            else:
                # 自动检测电力类型
                power_type = self._detect_power_type(data_points[0])
            
            # 更新客户端信息
            client_info = self.data_source_clients[client_id]
            
            # 如果是首次检测到电力类型，创建相应的数据文件
            if not client_info.get("auto_detected") or work_mode:
                client_info["power_type"] = power_type
                client_info["auto_detected"] = True
                client_info["work_mode"] = work_mode
                await self._create_client_data_file(client_id, power_type)
                
                # 初始化修正的固定相位滚动缓冲区
                self.scrolling_waveform_generator.initialize_client_buffer(client_id, power_type)
                logger.info(f"Auto-detected power type for {client_id}: {power_type.value}")
            
            client_info["data_count"] += len(data_points)
            client_info["last_update"] = datetime.now()
            client_info["status"] = ClientStatus.CONNECTED
            
            # 处理数据点并存储最新数据
            file_path = os.path.join(UPLOAD_DIR, self.client_data_files[client_id])
            processed_data = []
            
            for i, point in enumerate(data_points):
                if power_type == PowerType.DC:
                    processed_point = self._process_dc_data(point, seq_num + i)
                elif power_type == PowerType.THREE_PHASE:
                    processed_point = self._process_three_phase_data(point, seq_num + i)
                else:
                    processed_point = self._process_single_phase_data(point, seq_num + i)
                
                self.realtime_data_buffer[client_id].append(processed_point)
                processed_data.append(processed_point)
                
                # 写入CSV文件
                self._write_data_to_csv(file_path, processed_point, power_type)
            
            # 更新最新数据和缓存
            client_info["latest_data"] = processed_data[-1]
            self.realtime_cache[client_id] = {
                "latest_data": processed_data[-1],
                "buffer_size": len(self.realtime_data_buffer[client_id]),
                "last_update": datetime.now(),
                "power_type": power_type.value
            }
            
            # 广播实时更新
            await self.broadcast_realtime_update(client_id, processed_data[-1])
            
            # 如果开启了滚动监控，生成修正的固定相位滚动波形数据
            if client_info.get("scroll_monitoring", False):
                try:
                    scroll_data = self.scrolling_waveform_generator.generate_smooth_scroll_data(
                        client_id, processed_data[-1], num_new_points=15
                    )
                    if scroll_data.get("new_points_count", 0) > 0:
                        await self.broadcast_scroll_waveform_update(client_id, scroll_data)
                except Exception as e:
                    logger.error(f"Failed to generate corrected fixed phase scroll data for {client_id}: {e}")
            
            # 异步更新客户端列表
            await self.broadcast_client_list()
            
            logger.debug(f"Processed {len(processed_data)} {power_type.value} data points from {client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle stream data from {client_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _detect_power_type(self, data_point: dict) -> PowerType:
        """自动检测电力类型"""
        # 检查是否包含三相数据
        three_phase_keys = ['voltage_a', 'voltage_b', 'voltage_c']
        single_phase_keys = ['voltage']
        dc_keys = ['dc_voltage', 'dc_current']
        
        has_three_phase = any(key in data_point for key in three_phase_keys)
        has_single_phase = any(key in data_point for key in single_phase_keys)
        has_dc = any(key in data_point for key in dc_keys)
        
        if has_dc:
            return PowerType.DC
        elif has_three_phase:
            return PowerType.THREE_PHASE
        elif has_single_phase:
            return PowerType.SINGLE_PHASE
        else:
            return PowerType.SINGLE_PHASE

    def _process_dc_data(self, point: dict, time_seq: int) -> dict:
        """处理直流数据"""
        voltage = point.get('voltage', point.get('dc_voltage', 0))
        current = point.get('current', point.get('dc_current', 0))
        
        try:
            voltage = float(voltage)
            current = float(current)
        except (ValueError, TypeError):
            voltage = 0.0
            current = 0.0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "time_seq": time_seq,
            "voltage": voltage,
            "current": current,
            "power_type": "dc"
        }

    def _process_three_phase_data(self, point: dict, time_seq: int) -> dict:
        """处理三相电数据"""
        # 尝试多种可能的键名
        voltage_a = point.get('voltage_a', point.get('Va', point.get('voltage_phase_a', 0)))
        voltage_b = point.get('voltage_b', point.get('Vb', point.get('voltage_phase_b', 0)))
        voltage_c = point.get('voltage_c', point.get('Vc', point.get('voltage_phase_c', 0)))
        current_a = point.get('current_a', point.get('Ia', point.get('current_phase_a', 0)))
        current_b = point.get('current_b', point.get('Ib', point.get('current_phase_b', 0)))
        current_c = point.get('current_c', point.get('Ic', point.get('current_phase_c', 0)))
        
        try:
            voltage_a = float(voltage_a) if voltage_a is not None else 0.0
            voltage_b = float(voltage_b) if voltage_b is not None else 0.0
            voltage_c = float(voltage_c) if voltage_c is not None else 0.0
            current_a = float(current_a) if current_a is not None else 0.0
            current_b = float(current_b) if current_b is not None else 0.0
            current_c = float(current_c) if current_c is not None else 0.0
        except (ValueError, TypeError):
            voltage_a = voltage_b = voltage_c = 0.0
            current_a = current_b = current_c = 0.0
        
        # 检查B相数据异常
        if voltage_b < 1.0 and voltage_a > 50.0:
            voltage_b = voltage_a
        
        if current_b < 0.1 and current_a > 1.0:
            current_b = current_a
        
        return {
            "timestamp": datetime.now().isoformat(),
            "time_seq": time_seq,
            "voltage_a": voltage_a,
            "voltage_b": voltage_b,
            "voltage_c": voltage_c,
            "current_a": current_a,
            "current_b": current_b,
            "current_c": current_c,
            "power_type": "three_phase"
        }

    def _process_single_phase_data(self, point: dict, time_seq: int) -> dict:
        """处理单相电数据"""
        voltage = point.get('voltage', point.get('V', 0))
        current = point.get('current', point.get('I', 0))
        
        try:
            voltage = float(voltage)
            current = float(current)
        except (ValueError, TypeError):
            voltage = 0.0
            current = 0.0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "time_seq": time_seq,
            "voltage": voltage,
            "current": current,
            "power_type": "single_phase"
        }

    def _write_data_to_csv(self, file_path: str, data_point: dict, power_type: PowerType):
        """根据电力类型写入CSV数据"""
        try:
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if power_type == PowerType.DC:
                    writer.writerow([
                        data_point["timestamp"], data_point["time_seq"],
                        data_point["voltage"], data_point["current"]
                    ])
                elif power_type == PowerType.THREE_PHASE:
                    writer.writerow([
                        data_point["timestamp"], data_point["time_seq"],
                        data_point["voltage_a"], data_point["voltage_b"], data_point["voltage_c"],
                        data_point["current_a"], data_point["current_b"], data_point["current_c"]
                    ])
                else:
                    writer.writerow([
                        data_point["timestamp"], data_point["time_seq"],
                        data_point["voltage"], data_point["current"]
                    ])
        except Exception as e:
            logger.error(f"Failed to write data to CSV: {e}")

    async def broadcast_realtime_update(self, client_id: str, data_packet: dict):
        """广播实时数据更新"""
        message = {
            "type": "realtime_data_update",
            "client_id": client_id,
            "data": data_packet,
            "timestamp": datetime.now().isoformat()
        }
        
        # 只发送给正在监控此客户端的Web界面
        for web_client_id, web_info in self.web_clients.items():
            if web_info.get("monitoring_client") == client_id:
                await self.send_personal_message(message, web_client_id)

    async def broadcast_scroll_waveform_update(self, client_id: str, scroll_data: dict):
        """广播修正的固定相位滚动波形更新"""
        message = {
            "type": "scroll_waveform_update",
            "client_id": client_id,
            "scroll_data": scroll_data,
            "timestamp": datetime.now().isoformat()
        }
        
        # 只发送给正在监控此客户端且开启滚动模式的Web界面
        for web_client_id, web_info in self.web_clients.items():
            if (web_info.get("monitoring_client") == client_id and 
                web_info.get("scroll_mode", False)):
                await self.send_personal_message(message, web_client_id)

    async def start_monitoring(self, web_client_id: str, data_source_client_id: str, scroll_mode: bool = False):
        """开始监控指定的数据源客户端"""
        if web_client_id in self.web_clients and data_source_client_id in self.data_source_clients:
            self.web_clients[web_client_id]["monitoring_client"] = data_source_client_id
            self.web_clients[web_client_id]["scroll_mode"] = scroll_mode
            
            # 如果启用滚动模式，标记数据源客户端
            if scroll_mode:
                self.data_source_clients[data_source_client_id]["scroll_monitoring"] = True
                
                # 启动修正的固定相位滚动更新任务
                await self._start_scroll_task(data_source_client_id)
            
            # 发送确认消息
            await self.send_personal_message({
                "type": "monitoring_started",
                "data_source_client": data_source_client_id,
                "filename": self.client_data_files.get(data_source_client_id, ""),
                "power_type": self.data_source_clients[data_source_client_id].get("power_type", PowerType.SINGLE_PHASE).value,
                "scroll_mode": scroll_mode,
                "phase_system": "fixed_position_based_corrected"
            }, web_client_id)
            
            logger.info(f"Web client {web_client_id} started monitoring {data_source_client_id} (corrected fixed phase scroll: {scroll_mode})")
            return True
        return False

    async def _start_scroll_task(self, client_id: str):
        """启动修正的固定相位滚动更新任务"""
        if client_id not in self.scroll_update_tasks:
            task = asyncio.create_task(self._corrected_fixed_phase_scroll_update_loop(client_id))
            self.scroll_update_tasks[client_id] = task
            self.task_monitoring[client_id] = {
                "start_time": time.time(),
                "error_count": 0,
                "last_error": None
            }
            logger.info(f"📊 启动修正的固定相位滚动任务: {client_id}")

    async def _corrected_fixed_phase_scroll_update_loop(self, client_id: str):
        """修正的固定相位滚动更新循环任务"""
        error_count = 0
        max_errors = 10
        
        try:
            logger.info(f"🚀 修正的固定相位滚动循环启动: {client_id}")
            
            while (client_id in self.data_source_clients and 
                   self.data_source_clients[client_id].get("scroll_monitoring", False)):
                
                try:
                    latest_data = self.data_source_clients[client_id].get("latest_data")
                    if latest_data:
                        scroll_data = self.scrolling_waveform_generator.generate_smooth_scroll_data(
                            client_id, latest_data, num_new_points=8
                        )
                        
                        if scroll_data.get("new_points_count", 0) > 0:
                            await self.broadcast_scroll_waveform_update(client_id, scroll_data)
                            error_count = 0  # 重置错误计数
                        else:
                            logger.debug(f"No new corrected fixed phase scroll data generated for {client_id}")
                    
                    await asyncio.sleep(0.08)  # 80ms 更新间隔，更平滑
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error in corrected fixed phase scroll update loop for {client_id}: {e}")
                    
                    if client_id in self.task_monitoring:
                        self.task_monitoring[client_id]["error_count"] = error_count
                        self.task_monitoring[client_id]["last_error"] = str(e)
                    
                    if error_count >= max_errors:
                        logger.error(f"Too many errors ({error_count}) in corrected fixed phase scroll task for {client_id}, stopping")
                        break
                    
                    await asyncio.sleep(0.5)  # 错误后等待更长时间
                    
        except asyncio.CancelledError:
            logger.info(f"Corrected fixed phase scroll update loop cancelled for client {client_id}")
            raise
        except Exception as e:
            logger.error(f"Fatal error in corrected fixed phase scroll update loop for client {client_id}: {e}")
        finally:
            logger.info(f"Corrected fixed phase scroll update loop ended for client {client_id}")
            
            # 清理任务记录
            if client_id in self.scroll_update_tasks:
                del self.scroll_update_tasks[client_id]
            if client_id in self.task_monitoring:
                del self.task_monitoring[client_id]

    async def stop_monitoring(self, web_client_id: str):
        """停止监控"""
        if web_client_id in self.web_clients:
            monitored_client = self.web_clients[web_client_id].get("monitoring_client")
            
            # 检查是否还有其他Web客户端在监控同一个数据源
            if monitored_client:
                other_monitoring = any(
                    info.get("monitoring_client") == monitored_client and 
                    info.get("scroll_mode", False)
                    for wid, info in self.web_clients.items() 
                    if wid != web_client_id
                )
                
                # 如果没有其他客户端在滚动监控，停止滚动
                if not other_monitoring and monitored_client in self.data_source_clients:
                    self.data_source_clients[monitored_client]["scroll_monitoring"] = False
                    
                    # 停止滚动更新任务
                    if monitored_client in self.scroll_update_tasks:
                        self.scroll_update_tasks[monitored_client].cancel()
                        logger.info(f"Cancelled corrected fixed phase scroll task for {monitored_client}")
            
            self.web_clients[web_client_id]["monitoring_client"] = None
            self.web_clients[web_client_id]["scroll_mode"] = False
            
            await self.send_personal_message({
                "type": "monitoring_stopped"
            }, web_client_id)
            
            logger.info(f"Web client {web_client_id} stopped monitoring")

    def get_data_source_clients(self):
        """获取所有数据源客户端"""
        return list(self.data_source_clients.keys())
    
    def get_client_filename(self, client_id: str):
        """获取客户端数据文件名"""
        return self.client_data_files.get(client_id, "")

    def get_client_info(self, client_id: str):
        """获取客户端信息"""
        return self.data_source_clients.get(client_id, {})

    def get_client_buffer_data(self, client_id: str, limit: int = 100):
        """获取客户端缓冲区数据"""
        buffer_data = self.realtime_data_buffer.get(client_id, deque())
        return list(buffer_data)[-limit:] if len(buffer_data) > limit else list(buffer_data)

    async def handle_ping(self, client_id: str):
        """处理ping消息"""
        if client_id in self.connection_health:
            self.connection_health[client_id]["ping_count"] += 1
            self.connection_health[client_id]["last_ping"] = datetime.now()
        
        return await self.send_personal_message({"type": "pong"}, client_id)

# ==============================================================================
# 波形分析器类
# ==============================================================================
class OptimizedWaveAnalyzer:
    """优化的波形分析器"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.json', '.xlsx', '.xls', '.txt']
        self.waveform_generator = WaveformGenerator()
    
    def load_realtime_data(self, file_path: str, max_points: int = 1000) -> pd.DataFrame:
        """加载实时数据文件"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Data file not found: {file_path}")
                return pd.DataFrame()
                
            with open(file_path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for line in f) - 1
            
            if total_lines <= 0:
                return pd.DataFrame()
            
            if total_lines > max_points:
                skip_rows = total_lines - max_points
                df = pd.read_csv(file_path, encoding='utf-8', skiprows=range(1, skip_rows + 1))
            else:
                df = pd.read_csv(file_path, encoding='utf-8')
            
            logger.info(f"Loaded {len(df)} data points from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load realtime data from {file_path}: {e}")
            return pd.DataFrame()
    
    def detect_power_type_from_dataframe(self, df: pd.DataFrame) -> PowerType:
        """从DataFrame检测电力类型"""
        if df.empty:
            return PowerType.SINGLE_PHASE
            
        columns = df.columns.tolist()
        
        three_phase_columns = ['voltage_a', 'voltage_b', 'voltage_c']
        single_phase_columns = ['voltage']
        
        has_three_phase = any(col in columns for col in three_phase_columns)
        has_single_phase = any(col in columns for col in single_phase_columns)
        
        # 检查是否为直流（通过数据特征判断）
        if has_single_phase and not has_three_phase:
            voltage_col = 'voltage'
            if voltage_col in df.columns and len(df) > 10:
                voltage_data = df[voltage_col].values
                # 如果电压变化很小，可能是直流
                voltage_std = np.std(voltage_data)
                voltage_mean = np.mean(np.abs(voltage_data))
                if voltage_mean > 0 and voltage_std / voltage_mean < 0.1:  # 变化小于10%认为是直流
                    return PowerType.DC
        
        if has_three_phase:
            return PowerType.THREE_PHASE
        elif has_single_phase:
            return PowerType.SINGLE_PHASE
        else:
            return PowerType.SINGLE_PHASE

    def get_available_columns(self, df: pd.DataFrame, power_type: PowerType) -> List[str]:
        """获取可用的分析列"""
        if df.empty:
            return []
            
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if power_type == PowerType.DC:
            dc_columns = ['voltage', 'current']
            return [col for col in dc_columns if col in numeric_columns]
        elif power_type == PowerType.THREE_PHASE:
            three_phase_columns = [
                'voltage_a', 'voltage_b', 'voltage_c',
                'current_a', 'current_b', 'current_c'
            ]
            return [col for col in three_phase_columns if col in numeric_columns]
        else:
            single_phase_columns = ['voltage', 'current']
            return [col for col in single_phase_columns if col in numeric_columns]

    def generate_waveform_from_data(self, df: pd.DataFrame, power_type: PowerType, max_points: int = 1000) -> Dict:
        """根据数据生成波形"""
        try:
            if df.empty:
                logger.warning("DataFrame is empty, generating default waveform")
                # 生成默认波形
                if power_type == PowerType.DC:
                    return self.waveform_generator.generate_dc_waveform(12.0, 1.0, max_points)
                elif power_type == PowerType.THREE_PHASE:
                    return self.waveform_generator.generate_three_phase_waveform(0.01, 0.01, 0.01, 10, 10, 10, max_points)
                else:
                    return self.waveform_generator.generate_single_phase_waveform(0.01, 10, max_points)
            
            # 获取最新的数据点用于生成波形
            latest_data = df.iloc[-1] if len(df) > 0 else df.iloc[0]
            
            if power_type == PowerType.DC:
                # 直流模式：直接使用电压电流值
                voltage = float(latest_data.get('voltage', 12.0))
                current = float(latest_data.get('current', 1.0))
                return self.waveform_generator.generate_dc_waveform(voltage, current, max_points)
                
            elif power_type == PowerType.SINGLE_PHASE:
                # 单相模式：使用RMS值生成正弦波
                voltage_rms = float(latest_data.get('voltage', 0.01))
                current_rms = float(latest_data.get('current', 10.0))
                return self.waveform_generator.generate_single_phase_waveform(voltage_rms, current_rms, max_points)
                
            elif power_type == PowerType.THREE_PHASE:
                # 三相模式：使用三相RMS值生成三相正弦波
                voltage_a_rms = float(latest_data.get('voltage_a', 0.01))
                voltage_b_rms = float(latest_data.get('voltage_b', 0.01))
                voltage_c_rms = float(latest_data.get('voltage_c', 0.01))
                current_a_rms = float(latest_data.get('current_a', 10.0))
                current_b_rms = float(latest_data.get('current_b', 10.0))
                current_c_rms = float(latest_data.get('current_c', 10.0))
                
                return self.waveform_generator.generate_three_phase_waveform(
                    voltage_a_rms, voltage_b_rms, voltage_c_rms,
                    current_a_rms, current_b_rms, current_c_rms,
                    max_points
                )
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to generate waveform: {e}")
            return {}

    def analyze_signal_simple(self, data: np.ndarray, column_name: str = "voltage", power_type: PowerType = PowerType.SINGLE_PHASE) -> Dict:
        """简化的信号分析"""
        if len(data) == 0:
            return {}
        
        # 清理数据
        data = data[np.isfinite(data)]
        if len(data) == 0:
            return {}
        
        # 确定单位
        if 'voltage' in column_name:
            unit = 'V'
        elif 'current' in column_name:
            unit = 'A'
        else:
            unit = ''
        
        # 计算基本统计信息
        try:
            rms_value = np.sqrt(np.mean(np.square(data))) if len(data) > 0 else 0
            peak_value = np.max(np.abs(data)) if len(data) > 0 else 0
            
            # 计算周期数（假设每周期400个点）
            cycles_analyzed = len(data) / 400
            
            stats = {
                "count": {"title": "样本总数", "value": f"{len(data):,}", "unit": "", "icon": "fas fa-hashtag"},
                "cycles": {"title": "分析周期数", "value": f"{cycles_analyzed:.2f}", "unit": "个", "icon": "fas fa-sync"},
                "mean": {"title": "平均值", "value": f"{np.mean(data):.3f}", "unit": unit, "icon": "fas fa-calculator"},
                "max": {"title": "最大值", "value": f"{np.max(data):.3f}", "unit": unit, "icon": "fas fa-arrow-up"},
                "min": {"title": "最小值", "value": f"{np.min(data):.3f}", "unit": unit, "icon": "fas fa-arrow-down"},
                "rms": {"title": "RMS有效值", "value": f"{rms_value:.3f}", "unit": unit, "icon": "fas fa-bolt"},
                "peak": {"title": "峰值", "value": f"{peak_value:.3f}", "unit": unit, "icon": "fas fa-mountain"}
            }
            
            # 添加电力系统特有的分析
            if power_type != PowerType.DC and len(data) > 800:  # 至少2个周期的数据
                form_factor = rms_value / np.mean(np.abs(data)) if np.mean(np.abs(data)) > 0 else 0
                stats["form_factor"] = {"title": "波形因数", "value": f"{form_factor:.3f}", "unit": "", "icon": "fas fa-wave-square"}
                
                # 估算峰值因数
                crest_factor = peak_value / rms_value if rms_value > 0 else 0
                stats["crest_factor"] = {"title": "峰值因数", "value": f"{crest_factor:.3f}", "unit": "", "icon": "fas fa-chart-line"}
            
            return stats
            
        except Exception as e:
            logger.error(f"Signal analysis error: {e}")
            return {"error": {"title": "分析错误", "value": str(e), "unit": "", "icon": "fas fa-exclamation-triangle"}}

# 创建管理器和分析器实例
manager = OptimizedPowerConnectionManager()
analyzer = OptimizedWaveAnalyzer()

# ==============================================================================
# WebSocket端点
# ==============================================================================
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    client_type = "web" if client_id.startswith('web_') else "data_source"
    
    await manager.connect(client_id, websocket, client_type)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                logger.debug(f"Received from {client_id}: {message}")

                msg_type = message.get("type")

                if msg_type == "ping":
                    await manager.handle_ping(client_id)
                
                elif msg_type == "data_packet":
                    if client_type == "data_source":
                        await manager.handle_stream_data(client_id, message.get("data", {}))
                        await websocket.send_json({"type": "ack", "message": "数据包已接收"})
                
                elif msg_type == "start_monitoring":
                    if client_type == "web":
                        data_source_client = message.get("data_source_client")
                        scroll_mode = message.get("scroll_mode", False)
                        success = await manager.start_monitoring(client_id, data_source_client, scroll_mode)
                        if success:
                            await websocket.send_json({
                                "type": "monitoring_started", 
                                "data_source_client": data_source_client,
                                "scroll_mode": scroll_mode,
                                "phase_system": "fixed_position_based_corrected"
                            })
                        else:
                            await websocket.send_json({"type": "error", "message": "无法开始监控"})
                
                elif msg_type == "stop_monitoring":
                    if client_type == "web":
                        await manager.stop_monitoring(client_id)
                        await websocket.send_json({"type": "monitoring_stopped"})
                
                elif msg_type == "get_client_list":
                    if client_type == "web":
                        await manager.broadcast_client_list()
                
                else:
                    await websocket.send_json({"type": "ack", "message": f"收到消息类型: {msg_type}"})

            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON format"})

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")

# ==============================================================================
# HTTP路由
# ==============================================================================
@app.get("/", include_in_schema=False)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "active_page": "tasks"})

@app.get("/wave", response_class=HTMLResponse)
def wave_page(request: Request):
    """修正的固定相位滚动波形电力分析主页面"""
    return templates.TemplateResponse("wave.html", {
        "request": request,
        "active_page": "wave",
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M")
    })

# ==============================================================================
# 数据接收接口
# ==============================================================================
@app.post("/api/stream_data")
async def receive_stream_data(data: str = Form(...)):
    """接收电力数据流"""
    try:
        try:
            stream_data = json.loads(data)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return {"status": "error", "message": "数据格式错误"}
        
        client_id = stream_data.get('client_id')
        if not client_id:
            return {"status": "error", "message": "缺少客户端ID"}
        
        # 检查工作模式初始化
        work_mode = stream_data.get('work_mode')
        
        # 如果客户端未注册，自动注册
        if client_id not in manager.data_source_clients:
            logger.info(f"Auto-registering streaming client {client_id}")
            
            # 根据工作模式确定电力类型
            if work_mode in manager.work_mode_map:
                power_type = manager.work_mode_map[work_mode]
            else:
                power_type = PowerType.SINGLE_PHASE
                
            manager.data_source_clients[client_id] = {
                "connected_time": datetime.now(),
                "data_count": 0,
                "last_update": datetime.now(),
                "status": ClientStatus.REGISTERED,
                "client_type": "auto_detected",
                "description": f"Auto-registered {work_mode} sensor" if work_mode else "Auto-registered sensor",
                "latest_data": None,
                "power_type": power_type,
                "auto_detected": False,
                "work_mode": work_mode,
                "scroll_monitoring": False
            }
            manager.realtime_data_buffer[client_id] = deque(maxlen=manager.MAX_BUFFER_SIZE)
            manager.realtime_cache[client_id] = {}
        
        # 处理数据流
        success = await manager.handle_stream_data(client_id, stream_data)
        
        if success:
            data_count = len(stream_data.get('data', []))
            power_type = manager.data_source_clients[client_id].get("power_type", PowerType.SINGLE_PHASE)
            
            logger.debug(f"Successfully processed {data_count} {power_type.value} data points from {client_id}")
            
            return {
                "status": "success",
                "message": "数据接收成功",
                "processed": data_count,
                "seq": stream_data.get('seq', 0),
                "time": datetime.now().strftime("%H:%M:%S"),
                "power_type": power_type.value,
                "work_mode": work_mode,
                "phase_system": "fixed_position_based_corrected"
            }
        else:
            return {"status": "error", "message": "数据处理失败"}
            
    except Exception as e:
        logger.error(f"Stream data handler failed: {e}")
        return {"status": "error", "message": "服务器内部错误"}

# ==============================================================================
# 实时分析接口
# ==============================================================================
@app.post("/api/realtime_analyze")
async def realtime_analyze(
    client_id: str = Form(...),
    selected_column: str = Form("voltage"),
    model: str = Form("time_domain"),
    enable_filter: bool = Form(False),
    max_points: int = Form(1000),
    window_size: int = Form(10),
    show_all_phases: bool = Form(False),
    analysis_mode: str = Form("monitoring")
):
    """实时分析接口 - 支持修正的固定相位系统"""
    try:
        if client_id not in manager.data_source_clients:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": f"客户端 {client_id} 不存在"}
            )
        
        filename = manager.get_client_filename(client_id)
        if not filename:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "客户端数据文件不存在"}
            )
        
        file_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(file_path):
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "数据文件不存在"}
            )
        
        # 加载数据
        df = analyzer.load_realtime_data(file_path, max_points * 2)
        
        if df.empty:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "数据文件为空"}
            )
        
        # 检测电力类型
        power_type = analyzer.detect_power_type_from_dataframe(df)
        
        # 获取可用列
        available_columns = analyzer.get_available_columns(df, power_type)
        
        # 生成波形数据
        waveform_data = analyzer.generate_waveform_from_data(df, power_type, max_points)
        
        # 根据电力类型和显示需求选择分析列
        if power_type == PowerType.THREE_PHASE and show_all_phases:
            if selected_column.startswith('voltage'):
                analysis_columns = ['voltage_a', 'voltage_b', 'voltage_c']
            elif selected_column.startswith('current'):
                analysis_columns = ['current_a', 'current_b', 'current_c']
            else:
                analysis_columns = [selected_column]
        else:
            if selected_column and selected_column in available_columns:
                analysis_columns = [selected_column]
            else:
                analysis_columns = [available_columns[0]] if available_columns else ['voltage']
        
        # 处理多列数据
        wave_data_dict = {}
        stats_dict = {}
        
        for column in analysis_columns:
            # 使用生成的波形数据
            if column in waveform_data:
                wave_data_dict[column] = waveform_data[column]
            else:
                # 如果没有生成的波形数据，使用原始数据
                if column in df.columns:
                    raw_data = df[column].values
                    valid_mask = np.isfinite(raw_data)
                    data = raw_data[valid_mask]
                    
                    if len(data) > 0:
                        # 生成简单的波形数据
                        wave_data_dict[column] = [{"x": i, "y": float(v)} for i, v in enumerate(data[-max_points:])]
                    else:
                        wave_data_dict[column] = []
                else:
                    wave_data_dict[column] = []
            
            # 统计分析
            if column in df.columns:
                raw_data = df[column].values
                valid_mask = np.isfinite(raw_data)
                data = raw_data[valid_mask]
                
                if len(data) > 0:
                    sampling_rate = 20000.0  # 20kHz采样率
                    if analysis_mode == "monitoring":
                        stats = analyzer.analyze_signal_simple(data, column, power_type)
                    else:
                        stats = analyzer.analyze_signal_simple(data, column, power_type)  # 简化版本
                    
                    stats_dict[column] = stats
                else:
                    stats_dict[column] = {}
            else:
                stats_dict[column] = {}
        
        # 构建响应数据
        response_data = {
            "client_id": client_id,
            "filename": filename,
            "columns": df.columns.tolist(),
            "available_columns": available_columns,
            "selected_column": analysis_columns[0] if analysis_columns else selected_column,
            "data_count": len(df),
            "power_type": power_type.value,
            "analysis_mode": analysis_mode,
            "phase_system": "fixed_position_based_corrected",
            "analysis_params": {
                "model": model,
                "enable_filter": enable_filter,
                "max_points": max_points,
                "window_size": window_size,
                "sampling_rate": 20000.0,
                "show_all_phases": show_all_phases,
                "points_per_cycle": 400,
                "phase_calculation": "position_based_fixed_corrected"
            },
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 根据模式返回数据
        if len(analysis_columns) == 1:
            main_column = analysis_columns[0]
            response_data.update({
                "stats": stats_dict.get(main_column, {}),
                "wave_data": wave_data_dict.get(main_column, [])
            })
        else:
            response_data.update({
                "analysis_columns": analysis_columns,
                "stats": stats_dict,
                "wave_data_dict": wave_data_dict,
                "is_multi_phase": True
            })
        
        # 成功消息
        if analysis_mode == "monitoring":
            message = f"修正的固定相位实时监控更新完成 - 客户端: {client_id}"
        else:
            message = f"修正的固定相位深度分析完成 - 客户端: {client_id}, 模型: {model}"
        
        return {
            "status": "success",
            "message": message,
            "data": response_data
        }
        
    except Exception as e:
        logger.error(f"Realtime analysis error: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"分析错误: {str(e)}"}
        )

# ==============================================================================
# 健康检查和状态接口
# ==============================================================================
@app.get("/api/health")
async def health_check():
    """系统健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(manager.active_connections),
        "data_source_clients": len(manager.data_source_clients),
        "web_clients": len(manager.web_clients),
        "version": "6.1.0 - Corrected Fixed Phase Scrolling Waveform System",
        "features": [
            "🎯 修正的固定相位系统 - 彻底解决波形生成问题",
            "📐 连续位置相位计算 - (连续位置 % 400) / 400 * 2π",
            "📊 完整波形值存储 - 直接计算并存储正弦波形值",
            "🔄 真正的滚动更新 - 自然的示波器效果",
            "⚡ 支持a0/a1/a2工作模式自动识别",
            "🔌 直流/单相/三相电力系统完整支持",
            "🎛️ 20kHz采样率，每周期400点精确时间轴",
            "💓 客户端心跳监控和自动重连",
            "📈 实时数据分析和修正的固定相位波形生成",
            "🌊 三相波形同步显示，相位关系保持",
            "🛠️ 增强错误处理和任务自动恢复"
        ],
        "phase_system": {
            "type": "fixed_position_based_corrected",
            "calculation": "(continuous_position % 400) / 400 * 2π",
            "points_per_cycle": 400,
            "window_size_max": 2000,
            "update_direction": "right_to_left",
            "waveform_storage": "complete_calculated_values",
            "phase_continuity": "guaranteed_by_continuous_position"
        },
        "scroll_config": {
            "window_size": manager.scrolling_waveform_generator.scroll_window_size,
            "max_window_size": manager.scrolling_waveform_generator.max_window_size,
            "update_interval": "80ms",
            "phase_system": "fixed_corrected",
            "error_recovery": "enabled"
        }
    }

@app.get("/api/system_status")
async def system_status():
    """系统状态信息"""
    return {
        "server_time": datetime.now().isoformat(),
        "uptime": "运行中",
        "version": "6.1.0 - Corrected Fixed Phase Scrolling Waveform System",
        "phase_system": "fixed_position_based_corrected",
        "features": [
            "🎯 修正的固定相位系统 - 彻底解决波形生成问题",
            "📐 连续位置相位计算：(连续位置 % 400点) / 400 * 2π",
            "📊 完整波形值存储：直接计算并存储正弦波形值而非振幅",
            "🔄 真正的滚动更新：从右往左移动，模拟真实示波器",
            "⚡ 支持a0(直流)/a1(单相)/a2(三相)工作模式",
            "🔌 直流/单相/三相电力系统完整支持",
            "🎛️ 20kHz采样率，每周期精确400个数据点",
            "💓 WebSocket实时通信和客户端健康监控",
            "📈 数据流处理、CSV存储和实时分析",
            "🌊 三相波形同步显示，相位关系严格保持：A:0°, B:-120°, C:-240°",
            "🛠️ 增强错误处理、异常恢复和任务自动重启",
            "🔧 固定相位缓冲区：最大2000点窗口，循环使用"
        ],
        "connections": {
            "total": len(manager.active_connections),
            "data_sources": len(manager.data_source_clients),
            "web_clients": len(manager.web_clients)
        },
        "buffer_status": {
            client_id: len(buffer) for client_id, buffer in manager.realtime_data_buffer.items()
        },
        "scroll_tasks": {
            "active_scroll_clients": len(manager.scroll_update_tasks),
            "scroll_client_list": list(manager.scroll_update_tasks.keys()),
            "task_monitoring": manager.task_monitoring
        },
        "corrected_fixed_phase_config": {
            "sampling_rate": "20 kHz",
            "points_per_cycle": 400,
            "base_frequency": "50 Hz",
            "time_precision": "50 μs per point",
            "phase_calculation": "continuous_position_based: (continuous_pos % 400) / 400 * 2π",
            "window_max_size": 2000,
            "waveform_storage": "complete_calculated_sine_values",
            "scroll_direction": "right_to_left",
            "phase_stability": "完全固定，基于连续位置，无累积误差",
            "three_phase_offset": "A:0°, B:-120°, C:-240°",
            "power_factor_simulation": "-30° (感性负载)",
            "error_recovery": "自动任务重启机制",
            "key_improvement": "直接计算完整波形值而非仅存振幅"
        }
    }

# ==============================================================================
# 修正的固定相位专用接口
# ==============================================================================
@app.post("/api/adjust_window_size")
async def adjust_window_size(
    client_id: str = Form(...),
    new_size: int = Form(1000)
):
    """调整客户端窗口大小"""
    try:
        if client_id not in manager.data_source_clients:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": f"客户端 {client_id} 不存在"}
            )
        
        # 限制窗口大小范围
        new_size = max(400, min(new_size, 2000))  # 最小1个周期，最大2000点
        
        # 调整滚动生成器的窗口大小
        manager.scrolling_waveform_generator.adjust_window_size(client_id, new_size)
        
        return {
            "status": "success",
            "message": f"客户端 {client_id} 窗口大小已调整为 {new_size}",
            "client_id": client_id,
            "new_window_size": new_size,
            "max_allowed": 2000,
            "min_allowed": 400
        }
        
    except Exception as e:
        logger.error(f"Failed to adjust window size for {client_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"调整窗口大小失败: {str(e)}"}
        )

@app.get("/api/fixed_phase_status")
async def get_fixed_phase_status():
    """获取修正的固定相位系统状态"""
    try:
        phase_info = {}
        
        for client_id, buffer_info in manager.scrolling_waveform_generator.client_scroll_buffers.items():
            client_info = manager.get_client_info(client_id)
            
            phase_info[client_id] = {
                "power_type": buffer_info['power_type'].value,
                "continuous_position": buffer_info['continuous_position'],
                "window_size": buffer_info['window_size'],
                "waveform_buffer_sizes": {
                    "voltage": len(buffer_info.get('voltage_waveform', [])),
                    "current": len(buffer_info.get('current_waveform', [])),
                    "voltage_a": len(buffer_info.get('voltage_a_waveform', [])),
                    "voltage_b": len(buffer_info.get('voltage_b_waveform', [])),
                    "voltage_c": len(buffer_info.get('voltage_c_waveform', [])),
                    "current_a": len(buffer_info.get('current_a_waveform', [])),
                    "current_b": len(buffer_info.get('current_b_waveform', [])),
                    "current_c": len(buffer_info.get('current_c_waveform', []))
                },
                "latest_rms_values": buffer_info.get('latest_rms', {}),
                "last_update": buffer_info['last_update_time'],
                "scroll_monitoring": client_info.get("scroll_monitoring", False),
                "has_active_task": client_id in manager.scroll_update_tasks,
                "task_status": manager.task_monitoring.get(client_id, {}),
                "phase_calculation_demo": {
                    "continuous_pos_0": f"0 % 400 / 400 * 2π = 0 rad (0°)",
                    "continuous_pos_100": f"100 % 400 / 400 * 2π = {100/400*2*np.pi:.3f} rad ({100/400*360:.1f}°)",
                    "continuous_pos_200": f"200 % 400 / 400 * 2π = {200/400*2*np.pi:.3f} rad ({200/400*360:.1f}°)",
                    "continuous_pos_400": f"400 % 400 / 400 * 2π = 0 rad (0°，新周期)",
                    "continuous_pos_500": f"500 % 400 / 400 * 2π = {100/400*2*np.pi:.3f} rad ({100/400*360:.1f}°，第二周期)",
                    "note": "连续位置保证相位的连续性和周期性"
                }
            }
        
        return {
            "status": "success",
            "fixed_phase_clients": phase_info,
            "total_clients": len(phase_info),
            "active_tasks": len(manager.scroll_update_tasks),
            "generator_config": {
                "window_size": manager.scrolling_waveform_generator.scroll_window_size,
                "max_window_size": manager.scrolling_waveform_generator.max_window_size,
                "sampling_rate": manager.scrolling_waveform_generator.sampling_rate,
                "frequency": manager.scrolling_waveform_generator.frequency,
                "points_per_cycle": manager.scrolling_waveform_generator.points_per_cycle,
                "phase_system": "fixed_position_based_corrected",
                "phase_formula": "(continuous_position % points_per_cycle) / points_per_cycle * 2π",
                "waveform_storage": "complete_calculated_sine_values",
                "scroll_direction": "right_to_left",
                "improvement": "直接计算完整波形值，确保真正的正弦波形"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get corrected fixed phase status: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"获取修正的固定相位状态失败: {str(e)}"}
        )

@app.get("/api/demo_phase_calculation")
async def demo_phase_calculation():
    """演示修正的固定相位计算"""
    try:
        demo_positions = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800]
        points_per_cycle = 400
        
        phase_demo = []
        for pos in demo_positions:
            phase_rad = (pos % points_per_cycle) / points_per_cycle * 2 * np.pi
            phase_deg = (pos % points_per_cycle) / points_per_cycle * 360
            cycle_num = pos // points_per_cycle
            in_cycle_pos = pos % points_per_cycle
            
            # 计算完整的波形值
            amplitude = 0.01 * np.sqrt(2)  # 0.01V RMS的峰值
            sine_value = amplitude * np.sin(phase_rad)
            
            phase_demo.append({
                "continuous_position": pos,
                "cycle_number": cycle_num,
                "position_in_cycle": in_cycle_pos,
                "phase_radians": round(phase_rad, 4),
                "phase_degrees": round(phase_deg, 1),
                "formula": f"({pos} % {points_per_cycle}) / {points_per_cycle} * 2π",
                "sine_value": round(np.sin(phase_rad), 4),
                "voltage_value": round(sine_value, 2)
            })
        
        three_phase_demo = []
        for pos in [0, 100, 200, 300, 400, 500]:
            base_phase = (pos % points_per_cycle) / points_per_cycle * 2 * np.pi
            phases = {
                'A': base_phase,
                'B': base_phase - 2*np.pi/3,
                'C': base_phase - 4*np.pi/3
            }
            
            # 计算三相电压值
            voltage_peak = 0.01 * np.sqrt(2)
            voltages = {
                'A': voltage_peak * np.sin(phases['A']),
                'B': voltage_peak * np.sin(phases['B']),
                'C': voltage_peak * np.sin(phases['C'])
            }
            
            three_phase_demo.append({
                "continuous_position": pos,
                "phase_A_deg": round(np.degrees(phases['A']), 1),
                "phase_B_deg": round(np.degrees(phases['B']), 1),
                "phase_C_deg": round(np.degrees(phases['C']), 1),
                "voltage_A": round(voltages['A'], 2),
                "voltage_B": round(voltages['B'], 2),
                "voltage_C": round(voltages['C'], 2),
                "sine_A": round(np.sin(phases['A']), 4),
                "sine_B": round(np.sin(phases['B']), 4),
                "sine_C": round(np.sin(phases['C']), 4)
            })
        
        return {
            "status": "success",
            "phase_system": "fixed_position_based_corrected",
            "formula": "(continuous_position % points_per_cycle) / points_per_cycle * 2π",
            "points_per_cycle": points_per_cycle,
            "single_phase_demo": phase_demo,
            "three_phase_demo": three_phase_demo,
            "key_improvements": [
                "✅ 使用连续位置计数器，确保相位连续性",
                "✅ 直接计算完整波形值：amplitude * sin(phase)",
                "✅ 存储完整波形值而非仅振幅",
                "✅ 真正的正弦波形输出",
                "✅ 周期性保证：连续位置400与位置0相位相同",
                "✅ 三相关系严格维持：B相滞后A相120°，C相滞后A相240°",
                "✅ 滚动时生成新的完整波形值"
            ],
            "waveform_generation": {
                "single_phase": "voltage_value = voltage_peak * sin(phase) + harmonics + noise",
                "three_phase_A": "voltage_A = voltage_peak_A * sin(phase)",
                "three_phase_B": "voltage_B = voltage_peak_B * sin(phase - 2π/3)",
                "three_phase_C": "voltage_C = voltage_peak_C * sin(phase - 4π/3)",
                "power_factor": "current_phase = voltage_phase - π/6 (30° lag)"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to generate corrected phase calculation demo: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"生成修正的相位计算演示失败: {str(e)}"}
        )

# ==============================================================================
# 客户端管理接口
# ==============================================================================
@app.post("/api/register_client")
async def register_client(
    client_id: str = Form(...),
    client_type: str = Form("adaptive_sensor"),
    description: str = Form(""),
    power_type: PowerType = Form(PowerType.SINGLE_PHASE),
    work_mode: str = Form(None)
):
    """注册新的电力数据源客户端"""
    try:
        current_time = datetime.now()
        
        # 根据工作模式覆盖电力类型
        if work_mode in manager.work_mode_map:
            power_type = manager.work_mode_map[work_mode]
        
        manager.data_source_clients[client_id] = {
            "connected_time": current_time,
            "data_count": 0,
            "last_update": None,
            "status": ClientStatus.REGISTERED,
            "client_type": client_type,
            "description": description,
            "latest_data": None,
            "power_type": power_type,
            "auto_detected": False,
            "work_mode": work_mode,
            "scroll_monitoring": False
        }
        
        # 初始化数据缓冲区和缓存
        manager.realtime_data_buffer[client_id] = deque(maxlen=manager.MAX_BUFFER_SIZE)
        manager.realtime_cache[client_id] = {}
        
        await manager._create_client_data_file(client_id, power_type)
        await manager.broadcast_client_list()
        
        logger.info(f"Client {client_id} registered successfully as {power_type.value} with work mode {work_mode}")
        
        return {
            "status": "success",
            "message": f"客户端 {client_id} 注册成功",
            "client_id": client_id,
            "registered_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "power_type": power_type.value,
            "work_mode": work_mode,
            "phase_system": "fixed_position_based_corrected"
        }
        
    except Exception as e:
        logger.error(f"Failed to register client {client_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"注册失败: {str(e)}"}
        )

@app.get("/api/data_source_clients")
async def get_data_source_clients():
    """获取所有数据源客户端列表"""
    try:
        clients = []
        for client_id, info in manager.data_source_clients.items():
            if info["last_update"]:
                time_diff = (datetime.now() - info["last_update"]).total_seconds()
                is_active = time_diff < 60
            else:
                is_active = False
            
            clients.append({
                "id": client_id,
                "connected_time": info["connected_time"].strftime("%Y-%m-%d %H:%M:%S"),
                "data_count": info["data_count"],
                "last_update": info["last_update"].strftime("%H:%M:%S") if info["last_update"] else "无",
                "status": ClientStatus.CONNECTED.value if is_active else info["status"].value if isinstance(info["status"], ClientStatus) else info["status"],
                "filename": manager.client_data_files.get(client_id, ""),
                "latest_data": info.get("latest_data"),
                "buffer_size": len(manager.realtime_data_buffer.get(client_id, [])),
                "client_type": info.get("client_type", "unknown"),
                "description": info.get("description", ""),
                "power_type": info.get("power_type", PowerType.SINGLE_PHASE).value if isinstance(info.get("power_type"), PowerType) else info.get("power_type", "single_phase"),
                "auto_detected": info.get("auto_detected", False),
                "work_mode": info.get("work_mode"),
                "scroll_monitoring": info.get("scroll_monitoring", False),
                "phase_system": "fixed_position_based_corrected"
            })
        
        clients.sort(key=lambda x: x["last_update"] if x["last_update"] != "无" else "00:00:00", reverse=True)
        
        return {"status": "success", "clients": clients}
        
    except Exception as e:
        logger.error(f"Failed to get client list: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"获取客户端列表失败: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 启动修正的固定相位滚动波形电力系统监控平台")
    logger.info("📐 相位计算公式: (连续位置 % 400) / 400 * 2π")
    logger.info("📊 波形存储: 完整计算的正弦波形值，从右往左滚动更新")
    logger.info("🔄 滚动方向: 右→左，模拟真实示波器")
    logger.info("⚡ 支持模式: a0(直流) / a1(单相) / a2(三相)")
    logger.info("🎛️ 采样配置: 20kHz, 400点/周期, 最大2000点窗口")
    logger.info("✅ 关键修正: 直接计算并存储完整波形值而非仅振幅")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)