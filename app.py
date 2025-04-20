from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
from tasks import task1, task2
from fastapi import UploadFile, File, Form
import os
from datetime import datetime

app = FastAPI()

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件和模板配置
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/run_task/{task_id}")
async def run_task(task_id: int):
    try:
        if task_id == 1:
            result = task1.run_task()
        elif task_id == 2:
            result = task2.run_task()
        else:
            raise ValueError("无效的任务ID")
        return JSONResponse({"message": f"任务 {task_id} 执行成功: {result}"})
    except Exception as e:
        return JSONResponse({"message": f"执行失败: {str(e)}"}, status_code=500)

@app.get("/run_bash")
async def run_bash():
    try:
        script_path = os.path.join(os.path.dirname(__file__), "tasks", "run_bash.sh")
        result = subprocess.run(
            ["bash", script_path],
            capture_output=True,
            text=True,
            encoding='utf-8',  # 指定编码
            errors='ignore',   # 忽略解码错误
            check=True
        )
        return JSONResponse({
            "message": f"标准输出: {result.stdout.strip()}\n错误输出: {result.stderr.strip()}"
        })
    except subprocess.CalledProcessError as e:
        return JSONResponse({"message": f"脚本执行失败: {e.stderr}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"message": f"系统错误: {str(e)}"}, status_code=500)




# 在已有代码后添加以下路由
@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    filename: str = Form(...)
):
    try:
        # 创建图片目录
        save_dir = "pictures"
        os.makedirs(save_dir, exist_ok=True)
        
        # 验证文件类型
        if not file.content_type.startswith("image/"):
            return JSONResponse(
                {"detail": "仅支持图片文件"},
                status_code=400
            )
        
        # 生成安全路径
        file_ext = file.filename.split('.')[-1]
        safe_filename = f"{filename.split('.')[0]}.{file_ext}"
        save_path = os.path.join(save_dir, safe_filename)
        
        # 防止覆盖已有文件
        if os.path.exists(save_path):
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            safe_filename = f"{filename.split('.')[0]}_{timestamp}.{file_ext}"
            save_path = os.path.join(save_dir, safe_filename)
        
        # 保存文件
        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)
            
        return {"filename": safe_filename}
    
    except Exception as e:
        return JSONResponse(
            {"detail": f"上传失败: {str(e)}"},
            status_code=500
        )


