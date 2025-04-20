// 任务执行功能
async function handleTask(taskId) {
    const statusElement = document.getElementById("status");
    try {
        const endpoint = taskId === "bash" ? "/run_bash" : `/run_task/${taskId}`;
        const response = await fetch(endpoint);
        
        if (!response.ok) {
            throw new Error(`HTTP错误: ${response.status}`);
        }
        
        const data = await response.json();
        statusElement.textContent = data.message;
        statusElement.style.color = "#28a745"; // 绿色表示成功
    } catch (error) {
        console.error("请求失败:", error);
        statusElement.textContent = `错误: ${error.message}`;
        statusElement.style.color = "#dc3545"; // 红色表示错误
    }
}

// 事件绑定（原有任务按钮）
document.getElementById("task1").addEventListener("click", () => handleTask(1));
document.getElementById("task2").addEventListener("click", () => handleTask(2));
document.getElementById("task_bash").addEventListener("click", () => handleTask("bash"));

// 图片拖拽上传功能 -------------------------------------------------
function dragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    e.target.classList.add('dragover');
}

function dragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    e.target.classList.remove('dragover');
}

async function drop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.target.classList.remove('dragover');

    const statusElement = document.getElementById('upload-status');
    const fileInput = document.getElementById('filename');
    statusElement.style.color = ""; // 重置颜色
    
    // 验证文件
    const files = e.dataTransfer.files;
    if (files.length === 0) {
        statusElement.textContent = "错误：未检测到文件";
        statusElement.style.color = "red";
        return;
    }

    const file = files[0];
    if (!file.type.startsWith('image/')) {
        statusElement.textContent = "错误：仅支持图片文件 (JPEG, PNG, GIF)";
        statusElement.style.color = "red";
        return;
    }

    // 处理文件名
    let customName = fileInput.value.trim();
    if (!customName) {
        customName = `img_${Date.now()}`;
    } else {
        customName = cleanFileName(customName);
    }

    // 保留原始扩展名
    const ext = file.name.split('.').pop().toLowerCase();
    const filename = `${customName}.${ext}`;

    // 构建表单数据
    const formData = new FormData();
    formData.append('file', file);
    formData.append('filename', filename);

    try {
        statusElement.textContent = "上传中...";
        
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        if (response.ok) {
            statusElement.textContent = `上传成功 ➔ ${data.filename}`;
            statusElement.style.color = "green";
            
            // 3秒后清除状态
            setTimeout(() => {
                statusElement.textContent = "";
            }, 3000);
        } else {
            throw new Error(data.detail || "服务器返回未知错误");
        }
    } catch (error) {
        console.error("上传失败:", error);
        statusElement.textContent = `错误: ${error.message}`;
        statusElement.style.color = "red";
    }
}

// 辅助函数：清理非法字符
function cleanFileName(name) {
    return name
        .replace(/[\\/:"*?<>|]/g, "") // 移除非法字符
        .replace(/\s+/g, "_")        // 空格转下划线
        .substring(0, 50);           // 限制最大长度
}

// 可选：点击拖拽区域触发文件选择
document.getElementById('drop-zone').addEventListener('click', () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = (e) => {
        if (e.target.files.length > 0) {
            const fakeEvent = {
                dataTransfer: { files: e.target.files },
                preventDefault: () => {},
                stopPropagation: () => {}
            };
            drop(fakeEvent);
        }
    };
    input.click();
});