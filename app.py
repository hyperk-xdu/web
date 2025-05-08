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
    df = pd.read_csv(file.file)
    filename = f"{UPLOAD_DIR}/last_uploaded.csv"
    df.to_csv(filename, index=False)
    x_col = df.columns[0]
    columns = df.columns[1:].tolist()
    return templates.TemplateResponse("plot.html", {
        "request": request,
        "columns": columns,
        "filename": filename,
        "x_label": x_col
    })


@app.post("/plot_select", response_class=HTMLResponse)
def plot_selected(
    request: Request,
    filename: str = Form(...),
    columns: List[str] = Form(...)
):
    df = pd.read_csv(filename)
    x_col = df.columns[0]

    if not columns:
        return templates.TemplateResponse("plot.html", {
            "request": request,
            "plot_data": None,
            "columns": df.columns[1:].tolist(),
            "filename": filename,
            "error": "请选择至少一个通道列"
        })

    data = []
    for col in columns:
        trace = {
            "x": df[x_col].tolist(),
            "y": df[col].tolist(),
            "type": "scatter",
            "mode": "lines+markers",
            "name": col
        }
        data.append(trace)
    print("传入的列：", columns)
    print("生成数据：", data)

    return templates.TemplateResponse("plot.html", {
        "request": request,
        "plot_data": json.dumps(data),
        "x_label": x_col,
        "columns": df.columns[1:].tolist(),
        "filename": filename
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