from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os, subprocess
from tasks import task1, task2
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import pandas as pd
import json
import os
from typing import List

app = FastAPI(docs_url="/swagger", redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = "uploaded_csv"
os.makedirs(UPLOAD_DIR, exist_ok=True)
BASE_DIR = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
PICTURE_DIR = os.path.join(BASE_DIR, "pictures")
os.makedirs(PICTURE_DIR, exist_ok=True)

@app.get("/", include_in_schema=False)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "active_page": "tasks"})

@app.get("/tasks", response_class=HTMLResponse)
def tasks_page(request: Request):
    return templates.TemplateResponse("tasks.html", {"request": request, "active_page": "tasks"})

@app.get("/files", response_class=HTMLResponse)
def files_page(request: Request):
    return templates.TemplateResponse("files.html", {"request": request, "active_page": "files"})

@app.get("/doss", response_class=HTMLResponse)
def files_page(request: Request):
    return templates.TemplateResponse("doss.html", {"request": request, "active_page": "doss"})

@app.get("/search", response_class=HTMLResponse)
def search_page(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})

@app.get("/email", response_class=HTMLResponse)
def email_form(request: Request):
    return templates.TemplateResponse("email.html", {"request": request})

@app.get("/plot", response_class=HTMLResponse)
def plot_page(request: Request):
    return templates.TemplateResponse("plot.html", {"request": request})

@app.post("/plot", response_class=HTMLResponse)
async def plot_csv_upload(request: Request, file: UploadFile = File(...)):
    try:
        # 读取并保存文件
        df = pd.read_csv(file.file)
        filename = f"{UPLOAD_DIR}/{file.filename}"
        df.to_csv(filename, index=False)
        
        # 初始化选择（默认第一列为X轴）
        columns = df.columns.tolist()
        return templates.TemplateResponse("plot.html", {
            "request": request,
            "columns": columns,
            "x_col": columns[0],
            "selected_columns": columns[1:2] if len(columns) >1 else [],
            "filename": filename
        })
        
    except Exception as e:
        return templates.TemplateResponse("plot.html", {
            "request": request,
            "error": f"文件处理错误: {str(e)}"
        })

@app.post("/plot_select", response_class=HTMLResponse)
async def update_plot(
    request: Request,
    filename: str = Form(...),
    x_col: str = Form(...),
    columns: List[str] = Form(...)
):
    try:
        df = pd.read_csv(filename)
        all_columns = df.columns.tolist()
        
        # 验证列有效性
        if x_col not in all_columns:
            raise ValueError("无效的X轴列")
            
        invalid_cols = [col for col in columns if col not in all_columns]
        if invalid_cols:
            raise ValueError(f"无效的列: {', '.join(invalid_cols)}")
        
        # 生成绘图数据（包含X轴数据）
        plot_data = [{
            "name": x_col,
            "x": df.index.tolist(),
            "y": df[x_col].tolist()
        }]
        
        for col in columns:
            if col != x_col:
                plot_data.append({
                    "name": col,
                    "x": df[x_col].tolist(),  # X轴数据
                    "y": df[col].tolist()     # Y轴数据
                })
                
        return templates.TemplateResponse("plot.html", {
            "request": request,
            "plot_data": plot_data,
            "columns": all_columns,
            "x_col": x_col,
            "selected_columns": columns,
            "filename": filename
        })
        
    except Exception as e:
        return templates.TemplateResponse("plot.html", {
            "request": request,
            "error": str(e),
            "filename": filename,
            "columns": pd.read_csv(filename).columns.tolist() if os.path.exists(filename) else []
        })

@app.post("/send_email", response_class=HTMLResponse)
def send_email(request: Request, message: str = Form(...)):
    try:
        sender = "1748476648@qq.com"
        recipient = "kanghuibin@outlook.com"

        from email.header import Header

        msg = MIMEText(message, "plain", "utf-8")
        msg["Subject"] = Header("来自平台用户的建议", "utf-8")
        msg["From"] = sender
        msg["To"] = recipient

        server = smtplib.SMTP_SSL("smtp.qq.com", 465)
        server.login(sender, "gosqqzivffcrejbh")
        server.send_message(msg)
        server.quit()

        return templates.TemplateResponse("email.html", {
            "request": request,
            "success": True
        })
    except Exception as e:
        return templates.TemplateResponse("email.html", {
            "request": request,
            "success": False,
            "error": str(e)
        })
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), filename: str = Form(...)):
    ext = file.filename.split(".")[-1].lower()
    base_name = filename.strip().replace(" ", "_") or "image"
    save_path = os.path.join(PICTURE_DIR, f"{base_name}.{ext}")
    if os.path.exists(save_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(PICTURE_DIR, f"{base_name}_{timestamp}.{ext}")
    with open(save_path, "wb") as f:
        f.write(await file.read())
    return {"filename": os.path.basename(save_path)}

@app.get("/run_task/{task_id}")
def run_task(task_id: int):
    try:
        if task_id == 1:
            return {"message": task1.run_task()}
        elif task_id == 2:
            return {"message": task2.run_task()}
        else:
            raise ValueError("无效任务 ID")
    except Exception as e:
        return JSONResponse({"message": f"执行失败: {e}"}, status_code=500)

@app.get("/run_bash")
def run_bash():
    try:
        result = subprocess.run(["bash", "run_bash.sh"], capture_output=True, text=True)
        return {"message": result.stdout.strip()}
    except Exception as e:
        return JSONResponse({"message": str(e)}, status_code=500)
