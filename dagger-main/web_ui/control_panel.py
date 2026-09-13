"""
DAgger 控制面板 Web UI。

启动后访问 http://localhost:<port> 即可使用（端口由 dagger_params.yaml 的 web_ui.port 配置，默认 5002）。
支持模式切换、录制控制、相机预览、推理状态监控。
"""
import json
import time
import threading
import base64
from collections import deque
import numpy as np
from flask import Flask, jsonify, Response
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger
from std_msgs.msg import String
from sensor_msgs.msg import Image
from rcl_interfaces.msg import Log
from cv_bridge import CvBridge
import cv2


# 内嵌 HTML 模板
INDEX_HTML = '''<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DAgger 控制面板</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #1a1a2e; color: #eee; min-height: 100vh; padding: 15px; }
        .container { max-width: 1600px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 10px; font-size: 20px; }
        .mode-badge { text-align: center; margin-bottom: 10px; }
        .badge { display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; }
        .badge.idle { background: #6b7280; color: #fff; }
        .badge.human { background: #fbbf24; color: #1a1a2e; }
        .badge.policy { background: #4ade80; color: #1a1a2e; }
        .main-grid { display: grid; grid-template-columns: 300px 1fr 320px; gap: 12px; }
        @media (max-width: 1200px) { .main-grid { grid-template-columns: 1fr; } }
        .card { background: #16213e; border-radius: 10px; padding: 12px; margin-bottom: 10px; }
        .card h2 { font-size: 11px; color: #888; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }
        .btn { width: 100%; padding: 10px; border: none; border-radius: 6px; font-size: 13px; cursor: pointer; margin-bottom: 6px; transition: all 0.2s; }
        .btn:hover:not(:disabled) { transform: translateY(-1px); }
        .btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
        .btn-primary { background: #e94560; color: white; }
        .btn-success { background: #0f3460; color: white; }
        .btn-danger { background: #533483; color: white; }
        .btn-warning { background: #fbbf24; color: #1a1a2e; }
        .btn-start { background: #4ade80; color: #1a1a2e; font-weight: bold; font-size: 15px; padding: 14px; }
        .btn-stop { background: #e94560; color: white; font-weight: bold; font-size: 15px; padding: 14px; }
        .btn-hint { font-size: 9px; color: #666; text-align: center; margin-top: -2px; margin-bottom: 6px; }
        .row { display: flex; gap: 6px; }
        .row .btn { flex: 1; }
        .node-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
        .node-item { background: #0f3460; border-radius: 6px; padding: 6px; text-align: center; }
        .node-name { font-size: 9px; color: #888; margin-bottom: 2px; }
        .node-hz { font-size: 14px; font-weight: bold; }
        .node-hz span { font-size: 8px; font-weight: normal; color: #888; }
        .node-status { font-size: 8px; margin-top: 2px; }
        .node-status.active { color: #4ade80; }
        .node-status.inactive { color: #f87171; }
        .server-status { background: #0f3460; border-radius: 6px; padding: 10px; margin-bottom: 8px; }
        .server-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
        .server-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
        .server-dot.online { background: #4ade80; box-shadow: 0 0 6px #4ade80; }
        .server-dot.connecting { background: #fbbf24; box-shadow: 0 0 6px #fbbf24; }
        .server-dot.offline { background: #f87171; box-shadow: 0 0 6px #f87171; }
        .server-label { font-size: 14px; font-weight: bold; }
        .server-label.online { color: #4ade80; }
        .server-label.connecting { color: #fbbf24; }
        .server-label.offline { color: #f87171; }
        .server-detail { font-size: 10px; color: #888; margin-left: 18px; word-break: break-all; }
        .session-status { background: #0f3460; border-radius: 6px; padding: 10px; text-align: center; margin-bottom: 8px; }
        .session-state { font-size: 14px; font-weight: bold; margin-bottom: 4px; }
        .session-state.running { color: #4ade80; }
        .session-state.idle { color: #6b7280; }
        .session-state.paused { color: #fbbf24; }
        .session-info { font-size: 10px; color: #888; }
        .inference-info-line { font-size: 10px; color: #888; margin-top: 2px; }
        .recording-info-line { font-size: 10px; color: #fbbf24; margin-top: 2px; }
        .modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 1000; justify-content: center; align-items: center; }
        .modal-overlay.show { display: flex; }
        .modal-box { background: #16213e; border-radius: 12px; padding: 20px; max-width: 400px; text-align: center; }
        .modal-box h3 { margin-bottom: 10px; font-size: 16px; }
        .modal-box p { color: #888; font-size: 12px; margin-bottom: 15px; }
        .modal-box .row { justify-content: center; }
        .camera-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; }
        .camera-item { background: #0f3460; border-radius: 6px; overflow: hidden; }
        .camera-item img { width: 100%; height: auto; display: block; }
        .camera-label { font-size: 9px; color: #888; padding: 3px; text-align: center; background: rgba(0,0,0,0.3); }
        .log-panel { height: 100%; display: flex; flex-direction: column; }
        .log-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        .log-header h2 { margin-bottom: 0; }
        .log-clear { background: #0f3460; border: none; color: #888; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 10px; }
        .log-clear:hover { color: #eee; }
        .log-container { flex: 1; background: #0a0a1a; border-radius: 6px; overflow-y: auto; max-height: 500px; font-family: monospace; font-size: 11px; }
        .log-entry { padding: 4px 8px; border-bottom: 1px solid #1a1a2e; word-break: break-all; }
        .log-entry.debug { color: #888; }
        .log-entry.info { color: #60a5fa; }
        .log-entry.warn { color: #fbbf24; }
        .log-entry.error { color: #f87171; background: rgba(248,113,113,0.1); }
        .log-entry.fatal { color: #ef4444; background: rgba(239,68,68,0.2); }
        .log-time { color: #666; margin-right: 6px; }
        .log-node { color: #888; margin-right: 6px; }
        .status-bar { text-align: center; padding: 6px; background: #0f3460; border-radius: 6px; font-size: 11px; margin-top: 12px; }
        .url-hint { text-align: center; font-size: 10px; color: #666; margin-top: 8px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>DAgger 控制面板</h1>
        <div class="mode-badge">
            <span class="badge idle" id="mode-badge">IDLE</span>
        </div>
        <div class="main-grid">
            <!-- 左列：Server 状态 + 节点状态 + 推理控制 + Home -->
            <div>
                <div class="card">
                    <h2>Server 状态</h2>
                    <div class="server-status">
                        <div class="server-row">
                            <div class="server-dot offline" id="server-dot"></div>
                            <div class="server-label offline" id="server-label">离线</div>
                        </div>
                        <div class="server-detail" id="server-policy-type">策略: --</div>
                        <div class="server-detail" id="server-model-path">模型: --</div>
                    </div>
                </div>
                <div class="card">
                    <h2>节点状态</h2>
                    <div class="node-grid" id="node-grid"></div>
                </div>
                <div class="card">
                    <h2>会话控制</h2>
                    <div class="session-status">
                        <div class="session-state idle" id="session-state">待机</div>
                        <div class="session-info" id="session-info">点击"开始"启动会话</div>
                        <div class="inference-info-line" id="inference-detail"></div>
                        <div class="recording-info-line" id="recording-detail"></div>
                    </div>
                    <button class="btn btn-start" id="btn-start-session" onclick="startSession(this)">开始会话</button>
                    <button class="btn btn-stop" id="btn-stop-session" onclick="stopSession(this)" disabled>停止会话</button>
                    <div class="row" style="gap:6px;margin-top:4px;">
                        <button class="btn btn-primary" id="btn-pause-session" onclick="pauseSession(this)" disabled style="flex:1;">暂停推理</button>
                        <button class="btn btn-success" id="btn-resume-session" onclick="resumeSession(this)" disabled style="flex:1;">恢复推理</button>
                    </div>
                    <div class="btn-hint" id="session-hint" style="font-size:10px;color:#888;margin-top:4px;text-align:center;">点击「开始会话」连接 PolicyServer 并启动推理</div>
                </div>
                <div class="card">
                    <h2>机械臂 Home</h2>
                    <button class="btn btn-primary" id="home-btn" onclick="callHomeApi(this)">回到 Home 位姿</button>
                    <div class="btn-hint" id="home-hint">点击后机械臂移动到预设 home 位置</div>
                </div>
                <div class="card">
                    <h2>录制控制</h2>
                    <div class="recording-status" id="rec-status-box" style="background:#0f3460;border-radius:6px;padding:10px;text-align:center;margin-bottom:8px;">
                        <div id="rec-status-text" style="font-size:12px;color:#888;">无活跃录制</div>
                    </div>
                    <button class="btn btn-danger" id="btn-stop-recording" onclick="stopRecording(this)" disabled>停止录制</button>
                    <button class="btn btn-success" id="btn-new-episode" onclick="newEpisode(this)" disabled>开始新 Episode</button>
                    <div class="btn-hint" id="rec-hint" style="font-size:10px;color:#888;margin-top:4px;text-align:center;"></div>
                </div>
            </div>
            <!-- 中列：RGB 相机预览 -->
            <div class="card">
                <h2>相机预览</h2>
                <div class="camera-grid" id="camera-grid"></div>
            </div>
            <!-- 右列：日志 -->
            <div class="card log-panel">
                <div class="log-header">
                    <h2>系统日志</h2>
                    <button class="log-clear" onclick="clearLogs()">清空</button>
                </div>
                <div class="log-container" id="log-container"></div>
            </div>
        </div>
        <div class="status-bar" id="status">就绪</div>
        <div class="url-hint" id="url-hint"></div>
        <!-- 停止录制确认对话框 -->
        <div class="modal-overlay" id="stop-confirm-modal">
            <div class="modal-box">
                <h3>停止录制</h3>
                <p id="stop-confirm-info">确认保存或丢弃当前录制？</p>
                <div class="row">
                    <button class="btn btn-success" onclick="confirmSave()">保存</button>
                    <button class="btn btn-danger" onclick="confirmDiscard()">丢弃</button>
                </div>
            </div>
        </div>
    </div>
    <script>
        const MODE_MAP = { 'IDLE': 'idle', 'HUMAN': 'human', 'POLICY': 'policy' };
        const MODE_LABEL = { 'IDLE': 'IDLE', 'HUMAN': 'HUMAN', 'POLICY': 'POLICY' };
        const LOG_LEVELS = ['debug', 'debug', 'info', 'warn', 'error', 'fatal'];
        let cameraElements = {};
        let lastLogId = 0;
        let currentMode = 'IDLE';
        let sessionActive = false;
        let sessionPaused = false;
        let hasActiveRecording = false;
        let pendingStopSession = false;

        async function startSession(btn) {
            const status = document.getElementById('status');
            btn.disabled = true;
            status.textContent = '正在启动推理会话...';
            try {
                const res = await fetch('/api/start_session', { method: 'POST' });
                const data = await res.json();
                status.textContent = data.message;
                if (!data.success) { btn.disabled = false; }
            } catch (e) {
                status.textContent = '请求失败: ' + e.message;
                btn.disabled = false;
            }
        }

        async function stopSession(btn) {
            const status = document.getElementById('status');
            btn.disabled = true;
            status.textContent = '正在停止推理会话...';
            // 直接停止会话（后端自动暂停录制），不在前端先暂停
            try {
                const res = await fetch('/api/stop_session', { method: 'POST' });
                const data = await res.json();
                status.textContent = data.message;
                // 根据后端返回的 message 判断是否有录制需要保存/丢弃
                // 如果 message 包含"录制已暂停"，说明有活跃录制，弹确认框
                if (data.success && data.message.includes('录制已暂停')) {
                    pendingStopSession = false;
                    // 等待一小段时间让 updateStatus() 更新一次，确保 UI 显示正确的帧数
                    await new Promise(resolve => setTimeout(resolve, 100));
                    document.getElementById('stop-confirm-modal').classList.add('show');
                }
            } catch (e) {
                status.textContent = '请求失败: ' + e.message;
            }
            btn.disabled = false;
        }

        async function pauseSession(btn) {
            const status = document.getElementById('status');
            btn.disabled = true;
            status.textContent = '正在暂停推理...';
            try {
                const res = await fetch('/api/pause_session', { method: 'POST' });
                const data = await res.json();
                status.textContent = data.message;
            } catch (e) {
                status.textContent = '请求失败: ' + e.message;
            }
            btn.disabled = false;
        }

        async function resumeSession(btn) {
            const status = document.getElementById('status');
            btn.disabled = true;
            status.textContent = '正在恢复推理...';
            try {
                const res = await fetch('/api/resume_session', { method: 'POST' });
                const data = await res.json();
                status.textContent = data.message;
            } catch (e) {
                status.textContent = '请求失败: ' + e.message;
            }
            btn.disabled = false;
        }

        async function confirmSave() {
            document.getElementById('stop-confirm-modal').classList.remove('show');
            const status = document.getElementById('status');
            try {
                const res = await fetch('/api/stop_episode', { method: 'POST' });
                const data = await res.json();
                status.textContent = data.message;
            } catch (e) { status.textContent = '保存失败: ' + e.message; }
        }

        async function confirmDiscard() {
            document.getElementById('stop-confirm-modal').classList.remove('show');
            const status = document.getElementById('status');
            try {
                const res = await fetch('/api/discard_episode', { method: 'POST' });
                const data = await res.json();
                status.textContent = data.message;
            } catch (e) { status.textContent = '丢弃失败: ' + e.message; }
        }

        async function stopRecording(btn) {
            const status = document.getElementById('status');
            btn.disabled = true;
            status.textContent = '正在暂停录制...';
            try {
                await fetch('/api/pause_episode', { method: 'POST' });
            } catch (e) { /* 忽略 */ }
            pendingStopSession = false;
            document.getElementById('stop-confirm-modal').classList.add('show');
            btn.disabled = false;
        }

        async function newEpisode(btn) {
            const status = document.getElementById('status');
            btn.disabled = true;
            status.textContent = '正在开始新 Episode...';
            try {
                const res = await fetch('/api/new_episode', { method: 'POST' });
                const data = await res.json();
                status.textContent = data.message;
            } catch (e) {
                status.textContent = '请求失败: ' + e.message;
            }
            btn.disabled = false;
        }

        async function callHomeApi(btn) {
            const status = document.getElementById('status');
            const hint = document.getElementById('home-hint');
            btn.disabled = true;
            status.textContent = '正在移动到 home 位姿...';
            hint.textContent = '机械臂移动中，请稍候...（最长 30 秒）';
            try {
                const res = await fetch('/api/move_to_home', { method: 'POST' });
                const data = await res.json();
                status.textContent = data.message;
                if (data.success) { hint.textContent = '已到达 home 位姿'; }
                else { hint.textContent = data.message; }
            } catch (e) {
                status.textContent = '请求失败: ' + e.message;
                hint.textContent = '请求失败';
            }
            setTimeout(() => { btn.disabled = false; hint.textContent = '点击后机械臂移动到预设 home 位置'; }, 2000);
        }

        function clearLogs() { document.getElementById('log-container').innerHTML = ''; }

        async function updateStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                const dagger = data.dagger || {};
                const mode = dagger.mode || 'IDLE';
                currentMode = mode;

                // 模式徽章
                const modeBadge = document.getElementById('mode-badge');
                modeBadge.textContent = MODE_LABEL[mode] || mode;
                modeBadge.className = 'badge ' + (MODE_MAP[mode] || 'idle');

                // Server 状态卡片
                const srv = dagger.server || {};
                const serverDot = document.getElementById('server-dot');
                const serverLabel = document.getElementById('server-label');
                if (srv.connected && srv.ready) {
                    serverDot.className = 'server-dot online';
                    serverLabel.className = 'server-label online';
                    serverLabel.textContent = '在线';
                } else if (srv.connected && !srv.ready) {
                    serverDot.className = 'server-dot connecting';
                    serverLabel.className = 'server-label connecting';
                    serverLabel.textContent = '连接中（Warmup）';
                } else {
                    serverDot.className = 'server-dot offline';
                    serverLabel.className = 'server-label offline';
                    serverLabel.textContent = '离线';
                }
                document.getElementById('server-policy-type').textContent = '策略: ' + (srv.policy_type || '--');
                const pathStr = srv.pretrained_path || '--';
                const pathEl = document.getElementById('server-model-path');
                pathEl.textContent = '模型: ' + (pathStr.length > 50 ? '...' + pathStr.slice(-47) : pathStr);
                pathEl.title = pathStr;

                // 节点状态
                const nodeGrid = document.getElementById('node-grid');
                let nodeHtml = '';
                for (const [name, info] of Object.entries(data.nodes)) {
                    const sc = info.active ? 'active' : 'inactive';
                    const st = info.active ? '在线' : '离线';
                    nodeHtml += '<div class="node-item"><div class="node-name">' + name + '</div><div class="node-hz">' + info.hz.toFixed(1) + '<span> Hz</span></div><div class="node-status ' + sc + '">' + st + '</div></div>';
                }
                // 控制频率监控（从 dagger status 读取）
                const ctrl = dagger.control || {};
                const ctrlHz = ctrl.actual_hz || 0;
                const ctrlTarget = ctrl.target_hz || 0;
                const ctrlSource = ctrl.source || 'IDLE';
                const ctrlActive = ctrlHz > 0.5;
                const ctrlSc = ctrlActive ? 'active' : 'inactive';
                // 状态文字：显示来源和目标频率
                let ctrlSt = '待机';
                if (ctrlSource === 'POLICY') { ctrlSt = 'Policy ' + ctrlTarget + 'Hz'; }
                else if (ctrlSource === 'HUMAN') { ctrlSt = 'VR ' + ctrlTarget + 'Hz'; }
                nodeHtml += '<div class="node-item"><div class="node-name">Control</div><div class="node-hz">' + ctrlHz.toFixed(1) + '<span> Hz</span></div><div class="node-status ' + ctrlSc + '">' + ctrlSt + '</div></div>';
                nodeGrid.innerHTML = nodeHtml;

                // 推理控制区
                const isRunning = mode === 'POLICY' || mode === 'HUMAN';
                sessionPaused = dagger.session_paused || false;
                const startBtn = document.getElementById('btn-start-session');
                const stopBtn = document.getElementById('btn-stop-session');
                const pauseBtn = document.getElementById('btn-pause-session');
                const resumeBtn = document.getElementById('btn-resume-session');
                const sessionState = document.getElementById('session-state');
                const sessionInfo = document.getElementById('session-info');
                const inferenceDetail = document.getElementById('inference-detail');
                const recordingDetail = document.getElementById('recording-detail');
                const sessionHint = document.getElementById('session-hint');

                if (isRunning && sessionPaused) {
                    sessionState.textContent = '已暂停';
                    sessionState.className = 'session-state paused';
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    pauseBtn.disabled = true;
                    resumeBtn.disabled = false;
                    sessionHint.textContent = '推理已暂停，可保存/丢弃录制、回 Home、开始新 Episode，点击「恢复推理」继续';
                } else if (isRunning) {
                    sessionState.textContent = mode === 'HUMAN' ? '运行中（VR 接管）' : '运行中';
                    sessionState.className = 'session-state running';
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    pauseBtn.disabled = (mode === 'HUMAN');  // VR 接管时不允许暂停
                    resumeBtn.disabled = true;
                    sessionHint.textContent = mode === 'HUMAN' ? 'VR 接管中，松开 trigger 切回策略' : '策略执行中，点击「暂停推理」可暂停';
                } else {
                    sessionState.textContent = '待机';
                    sessionState.className = 'session-state idle';
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                    pauseBtn.disabled = true;
                    resumeBtn.disabled = true;
                    sessionHint.textContent = '点击「开始会话」连接 PolicyServer 并启动推理';
                }

                // 推理详情
                const inf = dagger.inference || {};
                if (Object.keys(inf).length === 0) {
                    sessionInfo.textContent = isRunning ? '纯 VR 模式运行中' : '纯 VR 模式（点击"开始会话"启动）';
                    inferenceDetail.textContent = '';
                } else if (!inf.warmup_done) {
                    if (!srv.connected) {
                        sessionInfo.textContent = '连接 PolicyServer 中...';
                    } else {
                        sessionInfo.textContent = 'Warmup 中...';
                    }
                    inferenceDetail.textContent = '';
                } else if (mode === 'HUMAN' && !inf.paused) {
                    sessionInfo.textContent = 'VR 接管中（推理后台运行）';
                    inferenceDetail.textContent = 'Buffer: ' + (inf.buffer_size || 0) + ' | 推理次数: ' + (inf.infer_count || 0);
                } else if (inf.paused) {
                    sessionInfo.textContent = isRunning ? 'VR 接管中（推理已暂停）' : '推理已暂停';
                    inferenceDetail.textContent = 'Buffer: ' + (inf.buffer_size || 0) + ' | 推理次数: ' + (inf.infer_count || 0);
                } else {
                    sessionInfo.textContent = isRunning ? '策略推理运行中' : '点击"开始会话"启动';
                    inferenceDetail.textContent = 'Buffer: ' + (inf.buffer_size || 0) + ' | 推理次数: ' + (inf.infer_count || 0);
                }

                // 录制详情
                const rec = dagger.recording || {};
                hasActiveRecording = rec.episode_active || false;
                const sessionIsActive = dagger.session_active || false;
                const stopRecBtn = document.getElementById('btn-stop-recording');
                const newEpBtn = document.getElementById('btn-new-episode');
                const recStatusText = document.getElementById('rec-status-text');
                const recHint = document.getElementById('rec-hint');

                if (rec.episode_active && !rec.episode_paused) {
                    recordingDetail.textContent = '录制中 | Episodes: ' + (rec.num_episodes || 0) + ' | 帧: ' + (rec.frame_count || 0);
                    recStatusText.textContent = '录制中 — Episode ' + ((rec.num_episodes || 0) + 1) + ' | 帧: ' + (rec.frame_count || 0);
                    recStatusText.style.color = '#4ade80';
                    stopRecBtn.disabled = false;
                    newEpBtn.disabled = true;
                    recHint.textContent = '录制进行中，点击「停止录制」保存当前 Episode';
                } else if (rec.episode_paused) {
                    recordingDetail.textContent = '录制已暂停 | Episodes: ' + (rec.num_episodes || 0) + ' | 帧: ' + (rec.frame_count || 0);
                    recStatusText.textContent = '录制已暂停 | 帧: ' + (rec.frame_count || 0);
                    recStatusText.style.color = '#fbbf24';
                    // 暂停时允许保存/丢弃
                    stopRecBtn.disabled = !sessionPaused;
                    newEpBtn.disabled = true;
                    recHint.textContent = sessionPaused ? '可保存或丢弃当前录制' : '录制已暂停，等待恢复';
                } else if (sessionIsActive) {
                    recordingDetail.textContent = rec.num_episodes > 0 ? 'Episodes: ' + rec.num_episodes : '';
                    recStatusText.textContent = '无活跃录制' + (rec.num_episodes > 0 ? ' | 已完成 Episodes: ' + rec.num_episodes : '');
                    recStatusText.style.color = '#888';
                    stopRecBtn.disabled = true;
                    newEpBtn.disabled = false;
                    recHint.textContent = '点击「开始新 Episode」开始录制';
                } else {
                    recordingDetail.textContent = rec.num_episodes > 0 ? 'Episodes: ' + (rec.num_episodes || 0) : '';
                    recStatusText.textContent = '无活跃录制';
                    recStatusText.style.color = '#888';
                    stopRecBtn.disabled = true;
                    newEpBtn.disabled = true;
                    recHint.textContent = '启动会话后可开始录制';
                }

                // 确认对话框信息
                if (rec.episode_paused || rec.episode_active) {
                    document.getElementById('stop-confirm-info').textContent = 'Episodes: ' + (rec.num_episodes || 0) + ' | 帧: ' + (rec.frame_count || 0) + ' — 确认保存或丢弃？';
                }

                // 相机预览
                const cameraGrid = document.getElementById('camera-grid');
                const cameraNames = Object.keys(data.cameras).sort();
                if (Object.keys(cameraElements).sort().join(',') !== cameraNames.join(',')) {
                    cameraElements = {};
                    let camHtml = '';
                    for (const name of cameraNames) {
                        camHtml += '<div class="camera-item"><img id="cam-' + name + '" src="" alt="' + name + '"><div class="camera-label">' + name + '</div></div>';
                    }
                    if (!camHtml) { camHtml = '<div class="camera-label" style="grid-column:1/-1;padding:20px;">无相机数据</div>'; }
                    cameraGrid.innerHTML = camHtml;
                    for (const name of cameraNames) { cameraElements[name] = document.getElementById('cam-' + name); }
                }
                for (const [name, imgData] of Object.entries(data.cameras)) {
                    if (cameraElements[name] && imgData) { cameraElements[name].src = 'data:image/jpeg;base64,' + imgData; }
                }

                // 日志
                if (data.logs && data.logs.length > 0) {
                    const logContainer = document.getElementById('log-container');
                    for (const log of data.logs) {
                        if (log.id > lastLogId) {
                            lastLogId = log.id;
                            const levelClass = LOG_LEVELS[log.level] || 'info';
                            const entry = document.createElement('div');
                            entry.className = 'log-entry ' + levelClass;
                            entry.innerHTML = '<span class="log-time">' + log.time + '</span><span class="log-node">[' + log.node + ']</span>' + log.msg;
                            logContainer.appendChild(entry);
                        }
                    }
                    while (logContainer.children.length > 200) { logContainer.removeChild(logContainer.firstChild); }
                    logContainer.scrollTop = logContainer.scrollHeight;
                }
            } catch (e) { console.error('状态更新失败:', e); }
        }
        document.getElementById('url-hint').textContent = window.location.origin;
        setInterval(updateStatus, 33);
        updateStatus();
    </script>
</body>
</html>'''



class TopicMonitor:
    """话题监控器：统计消息频率。"""
    def __init__(self, window_sec: float = 2.0):
        self.window_sec = window_sec
        self._timestamps = []
        self._lock = threading.Lock()

    def on_msg(self, msg=None):
        now = time.time()
        with self._lock:
            self._timestamps.append(now)
            cutoff = now - self.window_sec
            self._timestamps = [t for t in self._timestamps if t > cutoff]

    def get_hz(self) -> float:
        now = time.time()
        with self._lock:
            cutoff = now - self.window_sec
            valid = [t for t in self._timestamps if t > cutoff]
            return len(valid) / self.window_sec if len(valid) >= 2 else 0.0

    def is_active(self, timeout: float = 1.0) -> bool:
        with self._lock:
            if not self._timestamps:
                return False
            return (time.time() - self._timestamps[-1]) < timeout


class DAggerControlPanelNode(Node):
    """ROS2 节点：DAgger 控制面板，监控话题状态，调用 Service，收集日志。"""

    def __init__(self):
        super().__init__("dagger_control_panel")
        self.bridge = CvBridge()

        # 从参数读取相机列表
        self.declare_parameter("camera_names", ["cam0", "cam1"])
        self.camera_names = self.get_parameter("camera_names").get_parameter_value().string_array_value
        if not self.camera_names:
            self.camera_names = ["cam0", "cam1"]
        # ROS2 话题名（camera_node 发布的名称，可能与推理 key 不同）
        self.declare_parameter("camera_ros_names", ["cam0", "cam1"])
        self.camera_ros_names = self.get_parameter("camera_ros_names").get_parameter_value().string_array_value
        if not self.camera_ros_names or len(self.camera_ros_names) != len(self.camera_names):
            self.camera_ros_names = list(self.camera_names)
        # 映射: ROS2 话题名 → 推理 key（用于显示和监控）
        self._ros_to_display = dict(zip(self.camera_ros_names, self.camera_names))
        self.get_logger().info(f"相机列表: {self.camera_names}, ROS话题名: {self.camera_ros_names}")

        # 从参数读取 Web UI 端口
        self.declare_parameter("web_ui_port", 5002)
        self.web_ui_port = self.get_parameter("web_ui_port").get_parameter_value().integer_value
        self.get_logger().info(f"Web UI 端口: {self.web_ui_port}")

        # 节点监控器
        self.monitors = {
            "DAgger": TopicMonitor(),
            "Driver": TopicMonitor(),
            "VR Input": TopicMonitor(),
        }
        # 添加相机监控
        for cam in self.camera_names:
            self.monitors[cam.capitalize()] = TopicMonitor()

        # Service 客户端 - 模式切换
        self.set_idle_client = self.create_client(Trigger, "/dagger/set_idle")
        self.set_human_client = self.create_client(Trigger, "/dagger/set_human")
        self.set_policy_client = self.create_client(Trigger, "/dagger/set_policy")

        # Service 客户端 - 录制控制
        self.start_episode_client = self.create_client(Trigger, "/dagger/start_episode")
        self.pause_episode_client = self.create_client(Trigger, "/dagger/pause_episode")
        self.stop_episode_client = self.create_client(Trigger, "/dagger/stop_episode")
        self.discard_episode_client = self.create_client(Trigger, "/dagger/discard_episode")
        self.new_episode_client = self.create_client(Trigger, "/dagger/new_episode")

        # Service 客户端 - Home
        self.move_to_home_client = self.create_client(Trigger, "/driver/move_to_home")

        # Service 客户端 - 统一会话控制
        self.start_session_client = self.create_client(Trigger, "/dagger/start_session")
        self.stop_session_client = self.create_client(Trigger, "/dagger/stop_session")
        self.pause_session_client = self.create_client(Trigger, "/dagger/pause_session")
        self.resume_session_client = self.create_client(Trigger, "/dagger/resume_session")

        # DAgger 状态缓存（来自 /dagger/status JSON）
        self._dagger_status = {
            "mode": "IDLE",
            "vr_active": False,
            "cmd_count": 0,
            "safety_rejects": 0,
        }
        self._dagger_lock = threading.Lock()

        # 相机图像缓存
        self._camera_images = {}
        self._camera_lock = threading.Lock()
        self._last_cam_update = {}

        # 日志缓存
        self._logs = deque(maxlen=100)
        self._log_lock = threading.Lock()
        self._log_id = 0

        # 订阅 /dagger/status（JSON String，2Hz）
        self.create_subscription(
            String, "/dagger/status", self._on_dagger_status, 10
        )

        # 订阅 Driver 状态
        from sensor_msgs.msg import JointState
        self.create_subscription(
            JointState, "/rm/state_joint_state",
            lambda m: self.monitors["Driver"].on_msg(), 10
        )

        # 订阅 VR Input
        from geometry_msgs.msg import PoseStamped
        self.create_subscription(
            PoseStamped, "/vr/right/pose",
            lambda m: self.monitors["VR Input"].on_msg(), 10
        )

        # 动态订阅相机话题（用 ROS2 话题名订阅，用推理 key 作为显示名）
        for ros_name, inference_key in self._ros_to_display.items():
            self.create_subscription(
                Image, f"/camera/{ros_name}/color/image_raw",
                lambda m, c=inference_key: self._on_color(c, m), 10
            )

        # 订阅 /rosout 获取日志
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Log, "/rosout", self._on_rosout, qos)

    def _on_dagger_status(self, msg: String):
        """处理 /dagger/status JSON 消息，缓存模式/推理/录制状态。"""
        self.monitors["DAgger"].on_msg()
        try:
            data = json.loads(msg.data)
            with self._dagger_lock:
                self._dagger_status = data
        except Exception:
            pass

    def _on_color(self, cam_name: str, msg: Image):
        """处理 RGB 图像，限制更新频率。"""
        display_name = cam_name.capitalize()
        if display_name in self.monitors:
            self.monitors[display_name].on_msg()

        # 限制每个相机流的更新频率（约 30 FPS）
        now = time.time()
        if now - self._last_cam_update.get(cam_name, 0) < 0.033:
            return
        self._last_cam_update[cam_name] = now

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self._encode_and_store(cam_name, cv_img)
        except Exception:
            pass

    def _encode_and_store(self, name: str, cv_img):
        """缩放、编码并存储图像。"""
        h, w = cv_img.shape[:2]
        scale = min(320 / w, 240 / h)
        new_size = (int(w * scale), int(h * scale))
        cv_img = cv2.resize(cv_img, new_size, interpolation=cv2.INTER_NEAREST)
        _, buf = cv2.imencode('.jpg', cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, 60])
        b64 = base64.b64encode(buf).decode('utf-8')
        with self._camera_lock:
            self._camera_images[name] = b64

    def _on_rosout(self, msg: Log):
        """处理 /rosout 日志消息。"""
        # 过滤掉自己的日志
        if msg.name == "dagger_control_panel":
            return

        # 只显示 INFO 及以上级别（level: 1=DEBUG, 2=INFO, 4=WARN, 8=ERROR, 16=FATAL）
        level_map = {1: 0, 2: 2, 4: 3, 8: 4, 16: 5}
        level = level_map.get(msg.level, 2)
        if level < 2:  # 跳过 DEBUG
            return

        # 格式化时间
        sec = msg.stamp.sec % 86400
        time_str = f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"

        # 简化节点名
        node_name = msg.name.split('/')[-1] if '/' in msg.name else msg.name

        with self._log_lock:
            self._log_id += 1
            self._logs.append({
                "id": self._log_id,
                "time": time_str,
                "node": node_name,
                "level": level,
                "msg": msg.msg[:200]
            })

    def call_service(self, client, timeout=3.0):
        """从 Flask 线程安全调用 Service。"""
        req = Trigger.Request()
        future = client.call_async(req)
        start = time.time()
        while not future.done():
            if time.time() - start > timeout:
                return False, "调用超时"
            time.sleep(0.05)
        try:
            result = future.result()
            return (result.success, result.message) if result else (False, "无响应")
        except Exception as e:
            return False, f"调用失败: {e}"

    def get_full_status(self) -> dict:
        """获取完整状态，返回给前端。"""
        nodes = {name: {"active": mon.is_active(), "hz": round(mon.get_hz(), 1)}
                 for name, mon in self.monitors.items()}
        with self._dagger_lock:
            dagger = self._dagger_status.copy()
        with self._camera_lock:
            cameras = self._camera_images.copy()
        with self._log_lock:
            logs = list(self._logs)
        return {
            "nodes": nodes,
            "dagger": dagger,
            "cameras": cameras,
            "logs": logs,
        }


# Flask 应用
app = Flask(__name__)
ros_node: DAggerControlPanelNode = None


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype='text/html')


@app.route("/api/status")
def get_status():
    return jsonify(ros_node.get_full_status())


@app.route("/api/set_idle", methods=["POST"])
def set_idle():
    success, msg = ros_node.call_service(ros_node.set_idle_client)
    return jsonify({"success": success, "message": msg})


@app.route("/api/set_human", methods=["POST"])
def set_human():
    success, msg = ros_node.call_service(ros_node.set_human_client)
    return jsonify({"success": success, "message": msg})


@app.route("/api/set_policy", methods=["POST"])
def set_policy():
    success, msg = ros_node.call_service(ros_node.set_policy_client)
    return jsonify({"success": success, "message": msg})


@app.route("/api/start_episode", methods=["POST"])
def start_episode():
    success, msg = ros_node.call_service(ros_node.start_episode_client)
    return jsonify({"success": success, "message": msg})


@app.route("/api/pause_episode", methods=["POST"])
def pause_episode():
    success, msg = ros_node.call_service(ros_node.pause_episode_client)
    return jsonify({"success": success, "message": msg})


@app.route("/api/stop_episode", methods=["POST"])
def stop_episode():
    success, msg = ros_node.call_service(ros_node.stop_episode_client)
    return jsonify({"success": success, "message": msg})


@app.route("/api/discard_episode", methods=["POST"])
def discard_episode():
    success, msg = ros_node.call_service(ros_node.discard_episode_client)
    return jsonify({"success": success, "message": msg})


@app.route("/api/new_episode", methods=["POST"])
def new_episode():
    success, msg = ros_node.call_service(ros_node.new_episode_client)
    return jsonify({"success": success, "message": msg})


@app.route("/api/move_to_home", methods=["POST"])
def move_to_home():
    # 回 home 是阻塞操作，设置较长超时
    success, msg = ros_node.call_service(ros_node.move_to_home_client, timeout=30.0)
    return jsonify({"success": success, "message": msg})


@app.route("/api/start_session", methods=["POST"])
def start_session():
    success, msg = ros_node.call_service(ros_node.start_session_client, timeout=10.0)
    return jsonify({"success": success, "message": msg})


@app.route("/api/stop_session", methods=["POST"])
def stop_session():
    success, msg = ros_node.call_service(ros_node.stop_session_client, timeout=10.0)
    return jsonify({"success": success, "message": msg})


@app.route("/api/pause_session", methods=["POST"])
def pause_session():
    success, msg = ros_node.call_service(ros_node.pause_session_client)
    return jsonify({"success": success, "message": msg})


@app.route("/api/resume_session", methods=["POST"])
def resume_session():
    success, msg = ros_node.call_service(ros_node.resume_session_client)
    return jsonify({"success": success, "message": msg})


def main():
    global ros_node

    # 屏蔽 Flask/werkzeug 的 HTTP 请求日志，避免刷屏
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    rclpy.init()
    ros_node = DAggerControlPanelNode()

    port = ros_node.web_ui_port
    flask_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, debug=False,
                               use_reloader=False, threaded=True),
        daemon=True
    )
    flask_thread.start()

    ros_node.get_logger().info(f"Web UI 已启动: http://0.0.0.0:{port} (浏览器访问 http://<本机IP>:{port})")

    try:
        rclpy.spin(ros_node)
    except KeyboardInterrupt:
        pass
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
