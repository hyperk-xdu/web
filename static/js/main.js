function dragOver(e) {
  e.preventDefault();
  e.currentTarget.classList.add("dragover");
}

function dragLeave(e) {
  e.preventDefault();
  e.currentTarget.classList.remove("dragover");
}

async function drop(e) {
  e.preventDefault();
  e.currentTarget.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  const name = document.getElementById("filename").value.trim();
  const ext = file.name.split('.').pop();
  const filename = (name || "upload") + "." + ext;

  const formData = new FormData();
  formData.append("file", file);
  formData.append("filename", filename);

  const status = document.getElementById("upload-status");
  status.textContent = "上传中...";

  try {
    const res = await fetch("/upload", { method: "POST", body: formData });
    const json = await res.json();
    status.textContent = res.ok ? `上传成功: ${json.filename}` : `错误: ${json.detail}`;
  } catch (err) {
    status.textContent = "上传失败";
  }
}

function callTask(taskId) {
  const status = document.getElementById("status");
  const url = taskId === "bash" ? "/run_bash" : `/run_task/${taskId}`;

  fetch(url)
    .then(res => res.json())
    .then(data => status.textContent = data.message)
    .catch(() => status.textContent = "任务执行失败");
}

document.addEventListener("DOMContentLoaded", () => {
  const task1 = document.getElementById("task1");
  const task2 = document.getElementById("task2");
  const bash  = document.getElementById("task_bash");
  const zone  = document.getElementById("drop-zone");

  if (task1) task1.onclick = () => callTask(1);
  if (task2) task2.onclick = () => callTask(2);
  if (bash)  bash.onclick  = () => callTask("bash");

  if (zone) {
    zone.addEventListener("dragover", dragOver);
    zone.addEventListener("dragleave", dragLeave);
    zone.addEventListener("drop", drop);
  }
});