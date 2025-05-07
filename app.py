from datetime import datetime
import os
import subprocess
from fastapi import (
    FastAPI, Request,
    UploadFile, File, Form,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    JSONResponse, HTMLResponse, RedirectResponse
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from tasks import task1, task2   # 你的自定义任务模块

# ------------------------------------------------------------------------------
# FastAPI 基础
# ------------------------------------------------------------------------------
app = FastAPI()

# ---- CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 目录确保存在
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
PICTURE_DIR = os.path.join(BASE_DIR, "pictures")

for d in (STATIC_DIR, TEMPLATE_DIR, PICTURE_DIR):
    os.makedirs(d, exist_ok=True)

# ---- Mount Static & Templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)


# ------------------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------------------
def render(request: Request, template: str, active: str):
    """给模板统一注入 nav 高亮字段"""
    return templates.TemplateResponse(
        template,
        {"request": request, "active_page": active},
    )


def _script_path(name: str) -> str:
    """
    生成 Bash 脚本绝对路径，支持
    - <项目根>/tasks/<name>
    - <项目根>/<name>
    """
    cand1 = os.path.join(BASE_DIR, "tasks", name)
    cand2 = os.path.join(BASE_DIR, name)
    return cand1 if os.path.isfile(cand1) else cand2


# ------------------------------------------------------------------------------
# 页面路由
# ------------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def index():
    # 你也可以改成 render(..., "index.html", "tasks")
    return RedirectResponse(url="/tasks")

@app.get("/tasks", response_class=HTMLResponse)
async def task_page(request: Request):
    return render(request, "tasks.html", "tasks")

@app.get("/files", response_class=HTMLResponse)
async def files_page(request: Request):
    return render(request, "files.html", "files")

@app.get("/docs", response_class=HTMLResponse)
async def docs_page(request: Request):
    return render(request, "docs.html", "docs")


# ------------------------------------------------------------------------------
# 业务 API
# ------------------------------------------------------------------------------
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
        return JSONResponse({"message": f"执行失败: {e}"}, status_code=500)


@app.get("/run_bash")
async def run_bash():
    try:
        script_path = _script_path("run_bash.sh")
        result = subprocess.run(
            ["bash", script_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=True,
        )
        return {
            "message": (
                f"标准输出:\n{result.stdout.strip()}"
                f"\n\n错误输出:\n{result.stderr.strip() or 'None'}"
            )
        }
    except subprocess.CalledProcessError as e:
        return JSONResponse({"message": f"脚本执行失败: {e.stderr}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"message": f"系统错误: {e}"}, status_code=500)


@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    filename: str = Form(...),
):
    try:
        # ---- MIME 校验
        if not file.content_type.startswith("image/"):
            return JSONResponse({"detail": "仅支持图片文件"}, status_code=400)

        # ---- 生成安全文件名
        ext = file.filename.rsplit(".", 1)[-1].lower()
        stem = filename.rsplit(".", 1)[0] or "image"
        safe_name = f"{stem}.{ext}"
        save_path = os.path.join(PICTURE_DIR, safe_name)

        # ---- 重名加时间戳
        if os.path.exists(save_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = f"{stem}_{timestamp}.{ext}"
            save_path = os.path.join(PICTURE_DIR, safe_name)

        # ---- 保存文件
        with open(save_path, "wb") as fp:
            fp.write(await file.read())

        return {"filename": safe_name}
    except Exception as e:
        return JSONResponse({"detail": f"上传失败: {e}"}, status_code=500)
