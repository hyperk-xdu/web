#!/bin/bash
set -e

# ---------- 配置区 -----------
USER_NAME="ubuntu"
PROJECT_DIR="/home/${USER_NAME}/web"
CONDA_PATH="/home/${USER_NAME}/miniconda3"
CONDA_ENV="web"
PORT=8000
DOMAIN="62.234.138.62"  # 改为你的IP地址
REPO="git@github.com:hyperk-xdu/web.git"
NGINX_USER="www-data"
# ----------------------------

echo "✅ 开始自动化部署..."
echo "--------------------------------------------------"

# 步骤0：预检系统依赖
echo "🔍 检查系统依赖..."
sudo apt-get update
sudo apt-get install -y git nginx

# 步骤1：初始化 Conda 环境
echo "🔄 初始化 Conda 路径..."
source "${CONDA_PATH}/etc/profile.d/conda.sh"
export PATH="${CONDA_PATH}/bin:$PATH"

# 步骤2：清理旧环境
echo "🧹 清理旧环境和残留文件..."
{
    sudo systemctl stop web || true
    conda deactivate || true
    #rm -rf "${PROJECT_DIR}" || true
    #sudo rm -f /etc/nginx/sites-enabled/default /etc/nginx/sites-enabled/web || true
} &> /dev/null

# 步骤3：克隆代码
echo "⬇️ 克隆代码仓库到 ${PROJECT_DIR}..."
#git clone ${REPO} ${PROJECT_DIR}

# 步骤4：修复项目所有权
echo "🔒 配置项目目录权限..."
sudo chown -R ${USER_NAME}:${USER_NAME} ${PROJECT_DIR}
sudo find ${PROJECT_DIR} -type d -exec chmod 755 {} \;

# 步骤5：创建 Conda 环境
echo "🐍 创建 Conda 环境 [${CONDA_ENV}]..."


# 步骤6：安装依赖
echo "📦 安装 Python 依赖..."
conda activate ${CONDA_ENV}

# 步骤7：精确配置静态文件权限
echo "🔧 配置静态文件权限..."
sudo mkdir -p "${PROJECT_DIR}/static"
sudo chown -R ${USER_NAME}:${NGINX_USER} "${PROJECT_DIR}/static"
sudo chmod -R 775 "${PROJECT_DIR}/static"
sudo find "${PROJECT_DIR}/static" -type d -exec chmod g+s {} \;

# 步骤8：配置 systemd 服务
echo "🛠️ 配置 systemd 服务..."
sudo tee /etc/systemd/system/web.service > /dev/null <<EOF
[Unit]
Description=Web Service
After=network.target

[Service]
User=${USER_NAME}
WorkingDirectory=${PROJECT_DIR}
Environment="PATH=${CONDA_PATH}/envs/${CONDA_ENV}/bin"
ExecStart=${CONDA_PATH}/envs/${CONDA_ENV}/bin/uvicorn app:app --host 0.0.0.0 --port ${PORT}

# 安全限制
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full

[Install]
WantedBy=multi-user.target
EOF

# 步骤9：配置 Nginx（适配IP访问）
echo "🌐 配置 Nginx 反向代理..."
sudo tee /etc/nginx/sites-available/web > /dev/null <<EOF
server {
    listen 80;
    server_name ${DOMAIN} _;  # 添加 _ 通配符，允许任何域名/IP访问
    client_max_body_size 20M;

    location ~ ^/static/(.*)$ {
        alias ${PROJECT_DIR}/static/\$1;
        expires 30d;
        access_log off;
        add_header Cache-Control "public";
    }

    location / {
        proxy_pass http://127.0.0.1:${PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

# 强制启用配置并禁用默认配置
sudo ln -sf /etc/nginx/sites-available/web /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# 步骤10：应用配置
echo "🔧 应用系统配置..."
sudo systemctl daemon-reload
sudo systemctl enable nginx web
sudo systemctl restart nginx web

# 步骤11：跳过HTTPS配置（IP地址不支持Let's Encrypt）
echo "ℹ️  IP地址访问暂不配置HTTPS，如需HTTPS请使用域名"

# 最终验证
echo "🔍 运行最终验证..."
curl -Is http://localhost | head -n 1 || echo "HTTP 检查失败"
curl -Is http://127.0.0.1:${PORT} | head -n 1 || echo "FastAPI 检查失败"

echo "--------------------------------------------------"
echo "✅ 部署完成！访问地址：http://${DOMAIN}"
echo "🔍 查看实时日志："
echo "Nginx 错误日志: sudo tail -f /var/log/nginx/error.log"
echo "应用服务日志: journalctl -u web -f"