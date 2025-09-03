from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
import mimetypes
from datetime import datetime
import logging
import asyncio
from typing import Optional
import re
from pathlib import Path
import aiofiles
import json
from urllib.parse import unquote

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 初始化 FastAPI
app = FastAPI(
    title="高性能视频播放系统",
    description="支持智能预加载和HTTP Range请求的视频播放系统",
    version="2.0.1"
)

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 目录配置
BASE_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# 确保目录存在
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# 全局变量 - 用于管理预加载状态
current_loading_video = None
preload_tasks = {}

# 模板配置
templates = Jinja2Templates(directory=TEMPLATES_DIR) if os.path.exists(TEMPLATES_DIR) else None

# ==============================================================================
# 工具函数
# ==============================================================================

def is_valid_video_file(file_path: str) -> bool:
    """验证视频文件是否有效且可读取"""
    try:
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            return False
            
        if not os.path.isfile(file_path):
            logger.warning(f"不是文件: {file_path}")
            return False
            
        # 检查文件权限
        if not os.access(file_path, os.R_OK):
            logger.warning(f"文件无读取权限: {file_path}")
            return False
            
        # 检查文件大小（排除空文件）
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logger.warning(f"文件为空: {file_path}")
            return False
            
        # 检查文件是否过小（可能损坏）
        if file_size < 1024:  # 小于1KB
            logger.warning(f"文件过小，可能损坏: {file_path}")
            return False
            
        # 检查扩展名
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v', '.wmv', '.3gp', '.f4v'}
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in video_extensions:
            logger.warning(f"不支持的视频格式: {file_ext}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"验证视频文件失败 {file_path}: {e}")
        return False

def sanitize_filename(filename: str) -> str:
    """安全处理文件名，防止路径遍历攻击"""
    try:
        # URL解码
        filename = unquote(filename)
        
        # 移除危险字符
        filename = filename.replace('..', '').replace('/', '').replace('\\', '')
        filename = filename.strip()
        
        if not filename:
            raise ValueError("文件名为空")
            
        return filename
        
    except Exception as e:
        logger.error(f"文件名处理失败: {e}")
        raise HTTPException(status_code=400, detail=f"无效的文件名: {filename}")

def parse_range_header(range_header: str, file_size: int):
    """解析HTTP Range头"""
    try:
        range_match = re.match(r'bytes=(\d*)-(\d*)', range_header)
        if not range_match:
            return None, None
        
        start = range_match.group(1)
        end = range_match.group(2)
        
        start = int(start) if start else 0
        end = int(end) if end else file_size - 1
        
        # 确保范围有效
        start = max(0, min(start, file_size - 1))
        end = max(start, min(end, file_size - 1))
        
        return start, end
    except Exception as e:
        logger.error(f"解析Range头失败: {e}")
        return 0, file_size - 1

def get_optimal_chunk_size(file_size: int, bandwidth_estimate: str = "high") -> int:
    """根据文件大小和带宽估计获取最优分块大小"""
    if bandwidth_estimate == "low":
        base_chunk = 32 * 1024  # 32KB for low bandwidth
    elif bandwidth_estimate == "medium":
        base_chunk = 128 * 1024  # 128KB for medium bandwidth
    else:
        base_chunk = 512 * 1024  # 512KB for high bandwidth
    
    # 根据文件大小调整
    if file_size < 10 * 1024 * 1024:  # < 10MB
        return min(base_chunk, max(file_size // 20, 8192))
    elif file_size < 100 * 1024 * 1024:  # < 100MB
        return base_chunk
    else:  # > 100MB
        return min(base_chunk * 2, 1024 * 1024)  # 最大1MB

async def cancel_other_preloads(current_video: str):
    """取消其他视频的预加载任务，专注于当前视频"""
    global preload_tasks, current_loading_video
    
    current_loading_video = current_video
    
    # 取消所有非当前视频的预加载任务
    tasks_to_cancel = []
    for video_name, task in preload_tasks.items():
        if video_name != current_video and not task.done():
            tasks_to_cancel.append((video_name, task))
    
    for video_name, task in tasks_to_cancel:
        task.cancel()
        logger.info(f"取消预加载任务: {video_name}")
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            preload_tasks.pop(video_name, None)

# ==============================================================================
# 路由定义
# ==============================================================================

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    """视频中心主页"""
    # 如果有templates目录和文件，使用模板
    if templates and os.path.exists(os.path.join(TEMPLATES_DIR, "videos.html")):
        return templates.TemplateResponse("videos.html", {"request": request})
    
    # 否则返回内置HTML
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>视频中心</title>
        <meta charset="UTF-8">
    </head>
    <body>
        <h1>视频播放系统</h1>
        <p>请将完整的HTML文件放置在templates/videos.html中，或者访问API接口。</p>
        <h2>可用的API接口：</h2>
        <ul>
            <li><a href="/api/get_video_list">/api/get_video_list</a> - 获取视频列表</li>
            <li><a href="/api/video_quick_info">/api/video_quick_info</a> - 快速获取视频信息</li>
            <li><a href="/api/health">/api/health</a> - 系统健康检查</li>
        </ul>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/videos", response_class=HTMLResponse)
async def videos_page(request: Request):
    """视频中心页面"""
    if templates and os.path.exists(os.path.join(TEMPLATES_DIR, "videos.html")):
        return templates.TemplateResponse("videos.html", {
            "request": request,
            "active_page": "videos"
        })
    else:
        # 重定向到主页
        return await index(request)

# ==============================================================================
# 优化的API端点
# ==============================================================================

@app.get("/api/get_video_list")
async def get_video_list():
    """获取视频文件列表 - 增强版本"""
    try:
        video_files = []
        failed_files = []
        
        if not os.path.exists(VIDEO_DIR):
            os.makedirs(VIDEO_DIR, exist_ok=True)
            return {
                "status": "success", 
                "videos": [], 
                "message": "视频目录为空",
                "failed_files": []
            }
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v', '.wmv', '.3gp', '.f4v'}
        
        try:
            with os.scandir(VIDEO_DIR) as entries:
                for entry in entries:
                    if not entry.is_file():
                        continue
                        
                    file_ext = os.path.splitext(entry.name)[1].lower()
                    if file_ext not in video_extensions:
                        continue
                    
                    file_path = os.path.join(VIDEO_DIR, entry.name)
                    
                    # 验证文件有效性
                    if not is_valid_video_file(file_path):
                        failed_files.append({
                            "filename": entry.name,
                            "error": "文件无效或不可读",
                            "path": file_path
                        })
                        continue
                    
                    try:
                        stat_info = entry.stat()
                        video_info = {
                            "filename": entry.name,
                            "name": os.path.splitext(entry.name)[0],
                            "size": stat_info.st_size,
                            "modified": stat_info.st_mtime,
                            "extension": file_ext,
                            "url": f"/api/stream_video/{entry.name}",
                            "download_url": f"/api/download_video/{entry.name}",
                            "preload_url": f"/api/preload_video/{entry.name}",
                            "verify_url": f"/api/verify_video/{entry.name}",
                            "valid": True
                        }
                        video_files.append(video_info)
                        
                    except OSError as e:
                        logger.error(f"获取文件信息失败 {entry.name}: {e}")
                        failed_files.append({
                            "filename": entry.name,
                            "error": f"获取文件信息失败: {str(e)}",
                            "path": file_path
                        })
                        continue
                
        except OSError as e:
            logger.error(f"扫描视频目录失败: {e}")
            return {
                "status": "error",
                "message": f"扫描视频目录失败: {str(e)}",
                "videos": [],
                "failed_files": []
            }
        
        # 按修改时间排序并转换时间格式
        video_files.sort(key=lambda x: x["modified"], reverse=True)
        for video_file in video_files:
            video_file["modified"] = datetime.fromtimestamp(video_file["modified"]).isoformat()
        
        logger.info(f"成功获取 {len(video_files)} 个视频文件，{len(failed_files)} 个文件失败")
        
        return {
            "status": "success",
            "videos": video_files,
            "total_count": len(video_files),
            "failed_files": failed_files,
            "failed_count": len(failed_files),
            "optimization_enabled": True
        }
        
    except Exception as e:
        logger.error(f"获取视频列表失败: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error", 
                "message": f"获取视频列表失败: {str(e)}",
                "videos": [],
                "failed_files": []
            }
        )

@app.get("/api/verify_video/{filename}")
async def verify_video(filename: str):
    """验证单个视频文件是否可用"""
    try:
        safe_filename = sanitize_filename(filename)
        file_path = os.path.join(VIDEO_DIR, safe_filename)
        
        if not is_valid_video_file(file_path):
            return {
                "status": "failed",
                "filename": safe_filename,
                "valid": False,
                "error": "文件不存在、不可读或格式不支持",
                "path": file_path
            }
        
        file_size = os.path.getsize(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        stat_info = os.stat(file_path)
        
        return {
            "status": "success",
            "filename": safe_filename,
            "valid": True,
            "size": file_size,
            "mime_type": mime_type,
            "accessible": True,
            "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            "extension": os.path.splitext(safe_filename)[1].lower()
        }
        
    except Exception as e:
        logger.error(f"验证视频文件失败 {filename}: {e}")
        return {
            "status": "error",
            "filename": filename,
            "valid": False,
            "error": str(e)
        }

@app.get("/api/stream_video/{filename}")
async def stream_video(
    filename: str,
    request: Request,
    range: Optional[str] = Header(None),
    user_agent: Optional[str] = Header(None),
    connection_speed: Optional[str] = Header("high", alias="x-connection-speed")
):
    """优化的视频流式传输 - 增强错误处理"""
    try:
        # 安全处理文件名
        safe_filename = sanitize_filename(filename)
        file_path = os.path.join(VIDEO_DIR, safe_filename)
        
        # 验证文件
        if not is_valid_video_file(file_path):
            logger.error(f"视频文件验证失败: {safe_filename}")
            raise HTTPException(status_code=404, detail=f"视频文件不存在或无法访问: {safe_filename}")
        
        # 取消其他预加载，优先当前视频
        await cancel_other_preloads(safe_filename)
        
        file_size = os.path.getsize(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = 'video/mp4'
        
        # 获取最优分块大小
        chunk_size = get_optimal_chunk_size(file_size, connection_speed)
        
        logger.info(f"开始流传输视频: {safe_filename}, 大小: {file_size} bytes, 分块: {chunk_size} bytes")
        
        # 处理Range请求
        if range:
            try:
                start, end = parse_range_header(range, file_size)
                if start is None:
                    start, end = 0, file_size - 1
            except Exception as e:
                logger.error(f"解析Range头失败: {e}")
                start, end = 0, file_size - 1
            
            content_length = end - start + 1
            
            async def ranged_file_generator():
                try:
                    async with aiofiles.open(file_path, 'rb') as video_file:
                        await video_file.seek(start)
                        remaining = content_length
                        
                        while remaining > 0:
                            read_size = min(chunk_size, remaining)
                            chunk = await video_file.read(read_size)
                            if not chunk:
                                break
                            remaining -= len(chunk)
                            yield chunk
                            
                except Exception as e:
                    logger.error(f"流传输过程中出错 {safe_filename}: {e}")
                    raise
            
            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "Cache-Control": "public, max-age=3600",
                "X-Content-Type-Options": "nosniff",
                "Connection": "keep-alive",
                "X-File-Size": str(file_size)
            }
            
            return StreamingResponse(
                ranged_file_generator(),
                status_code=206,
                media_type=mime_type,
                headers=headers
            )
        else:
            # 完整文件传输
            async def full_file_generator():
                try:
                    async with aiofiles.open(file_path, 'rb') as video_file:
                        while True:
                            chunk = await video_file.read(chunk_size)
                            if not chunk:
                                break
                            yield chunk
                            
                except Exception as e:
                    logger.error(f"完整文件传输过程中出错 {safe_filename}: {e}")
                    raise
            
            headers = {
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
                "Cache-Control": "public, max-age=3600",
                "X-Content-Type-Options": "nosniff",
                "Connection": "keep-alive",
                "X-File-Size": str(file_size)
            }
            
            return StreamingResponse(
                full_file_generator(),
                media_type=mime_type,
                headers=headers
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"视频流传输失败 {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"视频流传输失败: {str(e)}")

@app.get("/api/preload_video/{filename}")
async def preload_video(
    filename: str,
    priority: bool = False,
    preload_size: int = 2097152  # 默认2MB
):
    """智能预加载视频 - 支持优先级控制"""
    try:
        global preload_tasks, current_loading_video
        
        # 安全检查
        safe_filename = sanitize_filename(filename)
        file_path = os.path.join(VIDEO_DIR, safe_filename)
        
        if not is_valid_video_file(file_path):
            raise HTTPException(status_code=404, detail=f"视频文件无效: {safe_filename}")
        
        file_size = os.path.getsize(file_path)
        actual_preload_size = min(preload_size, file_size)
        
        # 如果是优先视频，取消其他预加载
        if priority:
            await cancel_other_preloads(safe_filename)
        
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = 'video/mp4'
        
        async def preload_generator():
            try:
                async with aiofiles.open(file_path, 'rb') as video_file:
                    remaining = actual_preload_size
                    chunk_size = 65536  # 64KB chunks for preload
                    
                    while remaining > 0:
                        read_size = min(chunk_size, remaining)
                        chunk = await video_file.read(read_size)
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk
                        
            except Exception as e:
                logger.error(f"预加载读取失败 {safe_filename}: {e}")
                raise
        
        headers = {
            "Content-Range": f"bytes 0-{actual_preload_size-1}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(actual_preload_size),
            "Cache-Control": "public, max-age=7200",
            "X-Preload": "true",
            "X-Priority": "high" if priority else "normal",
            "X-File-Size": str(file_size)
        }
        
        logger.info(f"预加载视频 {safe_filename}: {actual_preload_size} bytes (优先级: {'高' if priority else '普通'})")
        
        return StreamingResponse(
            preload_generator(),
            status_code=206,
            media_type=mime_type,
            headers=headers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预加载失败 {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"预加载失败: {str(e)}")

@app.get("/api/preload_chunk/{filename}")
async def preload_chunk(
    filename: str,
    chunk_size: int = 524288,  # 512KB默认
    offset: int = 0
):
    """分块预加载视频 - 支持渐进式加载"""
    try:
        safe_filename = sanitize_filename(filename)
        file_path = os.path.join(VIDEO_DIR, safe_filename)
        
        if not is_valid_video_file(file_path):
            raise HTTPException(status_code=404, detail=f"视频文件无效: {safe_filename}")
        
        file_size = os.path.getsize(file_path)
        
        # 验证偏移量
        if offset >= file_size:
            raise HTTPException(status_code=416, detail="偏移量超出文件大小")
        
        # 计算实际读取大小
        actual_chunk_size = min(chunk_size, file_size - offset)
        
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = 'video/mp4'
        
        async def chunk_generator():
            try:
                async with aiofiles.open(file_path, 'rb') as video_file:
                    await video_file.seek(offset)
                    chunk = await video_file.read(actual_chunk_size)
                    if chunk:
                        yield chunk
                        
            except Exception as e:
                logger.error(f"分块读取失败 {safe_filename}: {e}")
                raise
        
        headers = {
            "Content-Range": f"bytes {offset}-{offset + actual_chunk_size - 1}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(actual_chunk_size),
            "Cache-Control": "public, max-age=7200",
            "X-Preload-Chunk": "true",
            "X-File-Size": str(file_size),
            "X-Chunk-Offset": str(offset)
        }
        
        logger.info(f"分块预加载 {safe_filename}: offset={offset}, size={actual_chunk_size}")
        
        return StreamingResponse(
            chunk_generator(),
            status_code=206,
            media_type=mime_type,
            headers=headers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分块预加载失败 {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"分块预加载失败: {str(e)}")

@app.get("/api/download_video/{filename}")
async def download_video(filename: str):
    """视频文件下载"""
    try:
        safe_filename = sanitize_filename(filename)
        file_path = os.path.join(VIDEO_DIR, safe_filename)
        
        if not is_valid_video_file(file_path):
            raise HTTPException(status_code=404, detail=f"视频文件不存在: {safe_filename}")
        
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = 'video/mp4'
        
        async def download_generator():
            try:
                async with aiofiles.open(file_path, 'rb') as video_file:
                    while True:
                        chunk = await video_file.read(8192)
                        if not chunk:
                            break
                        yield chunk
            except Exception as e:
                logger.error(f"下载过程中出错 {safe_filename}: {e}")
                raise
        
        return StreamingResponse(
            download_generator(),
            media_type=mime_type,
            headers={
                "Content-Disposition": f"attachment; filename={safe_filename}",
                "Accept-Ranges": "bytes"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"下载失败 {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"下载失败: {str(e)}")

@app.get("/api/video_quick_info")
async def video_quick_info(limit: int = 10):
    """快速获取视频基本信息 - 增强版本"""
    try:
        if not os.path.exists(VIDEO_DIR):
            return {"status": "success", "videos": []}
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v', '.wmv', '.3gp', '.f4v'}
        videos = []
        failed_count = 0
        
        try:
            filenames = []
            for filename in os.listdir(VIDEO_DIR):
                if len(filenames) >= limit:
                    break
                    
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in video_extensions:
                    file_path = os.path.join(VIDEO_DIR, filename)
                    
                    # 快速验证（不读取完整文件信息）
                    if os.path.isfile(file_path) and os.access(file_path, os.R_OK):
                        try:
                            # 检查文件不为空且大小合理
                            file_size = os.path.getsize(file_path)
                            if file_size > 1024:  # 大于1KB
                                filenames.append(filename)
                            else:
                                failed_count += 1
                                logger.warning(f"跳过过小文件: {filename} ({file_size} bytes)")
                        except OSError:
                            failed_count += 1
                            continue
                    else:
                        failed_count += 1
            
            # 构造基本信息
            for filename in filenames:
                videos.append({
                    "filename": filename,
                    "name": os.path.splitext(filename)[0],
                    "extension": os.path.splitext(filename)[1].lower(),
                    "url": f"/api/stream_video/{filename}",
                    "preload_url": f"/api/preload_video/{filename}",
                    "download_url": f"/api/download_video/{filename}",
                    "verify_url": f"/api/verify_video/{filename}"
                })
                
        except OSError as e:
            logger.error(f"快速扫描失败: {e}")
        
        return {
            "status": "success", 
            "videos": videos,
            "fast_mode": True,
            "message": f"快速获取{len(videos)}个视频，跳过{failed_count}个无效文件" if failed_count > 0 else f"快速获取{len(videos)}个视频"
        }
        
    except Exception as e:
        logger.error(f"快速获取视频信息失败: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/set_priority_video/{filename}")
async def set_priority_video(filename: str):
    """设置优先加载的视频"""
    try:
        global current_loading_video
        
        # 验证文件存在
        safe_filename = sanitize_filename(filename)
        file_path = os.path.join(VIDEO_DIR, safe_filename)
        
        if not is_valid_video_file(file_path):
            raise HTTPException(status_code=404, detail="视频文件不存在或无效")
        
        # 取消其他预加载任务
        await cancel_other_preloads(safe_filename)
        
        logger.info(f"设置优先视频: {safe_filename}")
        
        return {
            "status": "success",
            "priority_video": safe_filename,
            "message": f"已设置 {safe_filename} 为优先加载视频"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"设置优先视频失败: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/preload_status")
async def preload_status():
    """获取当前预加载状态"""
    global current_loading_video, preload_tasks
    
    active_tasks = {
        video: "running" if not task.done() else "completed"
        for video, task in preload_tasks.items()
    }
    
    return {
        "status": "success",
        "current_priority_video": current_loading_video,
        "active_preload_tasks": active_tasks,
        "total_active_tasks": len([t for t in preload_tasks.values() if not t.done()]),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/bandwidth_test")
async def bandwidth_test():
    """简单的带宽测试接口"""
    test_data = b"0" * (100 * 1024)  # 100KB测试数据
    
    return Response(
        content=test_data,
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Bandwidth-Test": "true"
        }
    )

# ==============================================================================
# 系统管理接口
# ==============================================================================

@app.get("/api/health")
async def health_check():
    """系统健康检查 - 增强版本"""
    global preload_tasks, current_loading_video
    
    try:
        # 检查视频目录
        video_dir_status = "healthy"
        video_count = 0
        invalid_files = 0
        total_size = 0
        
        if os.path.exists(VIDEO_DIR):
            try:
                video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v', '.wmv', '.3gp', '.f4v'}
                for filename in os.listdir(VIDEO_DIR):
                    file_ext = os.path.splitext(filename)[1].lower()
                    if file_ext in video_extensions:
                        file_path = os.path.join(VIDEO_DIR, filename)
                        if is_valid_video_file(file_path):
                            video_count += 1
                            try:
                                total_size += os.path.getsize(file_path)
                            except OSError:
                                pass
                        else:
                            invalid_files += 1
            except Exception as e:
                video_dir_status = f"error: {str(e)}"
        else:
            video_dir_status = "directory_missing"
        
        active_preloads = len([t for t in preload_tasks.values() if not t.done()])
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "高性能视频播放系统",
            "version": "2.0.1",
            "video_directory": {
                "status": video_dir_status,
                "path": VIDEO_DIR,
                "valid_videos": video_count,
                "invalid_files": invalid_files,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "accessible": os.access(VIDEO_DIR, os.R_OK) if os.path.exists(VIDEO_DIR) else False
            },
            "features": [
                "HTTP Range 请求支持",
                "智能预加载管理",
                "优先级控制",
                "自适应分块传输",
                "文件验证",
                "错误恢复",
                "异步文件操作"
            ],
            "active_preload_tasks": active_preloads,
            "current_priority_video": current_loading_video
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.post("/api/recover_system")
async def recover_system():
    """系统错误恢复"""
    try:
        global preload_tasks, current_loading_video
        
        # 清理预加载任务
        cancelled_tasks = 0
        for task in preload_tasks.values():
            if not task.done():
                task.cancel()
                cancelled_tasks += 1
        preload_tasks.clear()
        current_loading_video = None
        
        # 检查视频目录
        recovery_actions = []
        
        if not os.path.exists(VIDEO_DIR):
            os.makedirs(VIDEO_DIR, exist_ok=True)
            recovery_actions.append("创建视频目录")
        
        # 验证视频文件
        valid_videos = 0
        invalid_files = []
        
        if os.path.exists(VIDEO_DIR):
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v', '.wmv', '.3gp', '.f4v'}
            try:
                for filename in os.listdir(VIDEO_DIR):
                    file_ext = os.path.splitext(filename)[1].lower()
                    if file_ext in video_extensions:
                        file_path = os.path.join(VIDEO_DIR, filename)
                        if is_valid_video_file(file_path):
                            valid_videos += 1
                        else:
                            invalid_files.append(filename)
                            logger.warning(f"发现无效视频文件: {filename}")
            except Exception as e:
                recovery_actions.append(f"扫描文件时出错: {str(e)}")
        
        recovery_actions.extend([
            f"取消了 {cancelled_tasks} 个预加载任务",
            f"验证了 {valid_videos} 个有效视频",
            f"发现 {len(invalid_files)} 个无效文件"
        ])
        
        logger.info("系统恢复完成")
        
        return {
            "status": "success",
            "message": "系统恢复完成",
            "recovery_actions": recovery_actions,
            "cancelled_tasks": cancelled_tasks,
            "valid_videos": valid_videos,
            "invalid_files": invalid_files[:10],  # 只返回前10个无效文件
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"系统恢复失败: {e}")
        return {
            "status": "error",
            "message": f"系统恢复失败: {str(e)}"
        }

@app.get("/api/performance_info")
async def performance_info():
    """性能信息接口"""
    try:
        video_count = 0
        total_size = 0
        supported_formats = []
        
        if os.path.exists(VIDEO_DIR):
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v', '.wmv', '.3gp', '.f4v'}
            format_counts = {}
            
            for filename in os.listdir(VIDEO_DIR):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in video_extensions:
                    file_path = os.path.join(VIDEO_DIR, filename)
                    if is_valid_video_file(file_path):
                        try:
                            file_size = os.path.getsize(file_path)
                            total_size += file_size
                            video_count += 1
                            
                            format_counts[file_ext] = format_counts.get(file_ext, 0) + 1
                        except OSError:
                            continue
            
            supported_formats = [
                {"format": fmt, "count": count} 
                for fmt, count in format_counts.items()
            ]
        
        return {
            "status": "success",
            "video_count": video_count,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "supported_formats": supported_formats,
            "optimization_features": [
                "Range请求支持",
                "智能预加载",
                "优先级管理", 
                "自适应分块",
                "缓存优化",
                "异步I/O",
                "文件验证"
            ],
            "performance_tips": [
                "使用Range请求支持断点续传",
                "预加载功能提高播放流畅度",
                "优先级控制确保当前视频最高质量",
                "自适应分块根据网络状况调整",
                "文件验证确保视频完整性"
            ]
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==============================================================================
# 错误处理
# ==============================================================================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """404错误处理"""
    return JSONResponse(
        status_code=404,
        content={
            "status": "error",
            "message": "请求的资源不存在",
            "path": str(request.url.path),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    """500错误处理"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "服务器内部错误",
            "timestamp": datetime.now().isoformat()
        }
    )

# ==============================================================================
# 启动应用
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("启动高性能视频播放系统 v2.0.1")
    logger.info("修复内容:")
    logger.info("  - 完善文件验证机制")
    logger.info("  - 增强错误处理")
    logger.info("  - 优化异步文件操作")
    logger.info("  - 添加系统恢复功能")
    logger.info("  - 改进Range请求支持")
    logger.info("访问地址: http://localhost:8000")
    logger.info(f"视频目录: {VIDEO_DIR}")
    
    if not os.path.exists(VIDEO_DIR):
        logger.warning(f"视频目录不存在，已创建: {VIDEO_DIR}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        access_log=True,
        loop="asyncio"
    )