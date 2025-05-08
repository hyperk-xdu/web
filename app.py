from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os, subprocess
from tasks import task1, task2
from datetime import datetime

app = FastAPI(docs_url="/swagger", redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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