
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import subprocess
from tasks import task1, task2

app = FastAPI()

# ✅ 静态目录挂载，必须加 name="static"
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ 使用模板引擎
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # ✅ 传入 request 给模板引擎
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/run_task/{task_id}")
async def run_task(task_id: int):
    if task_id == 1:
        result = task1.run_task()
        return {"message": f"任务 1 执行结果: {result}"}
    elif task_id == 2:
        result = task2.run_task()
        return {"message": f"任务 2 执行结果: {result}"}
    else:
        return {"message": "无效任务 ID"}
