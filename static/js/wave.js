// wave.js - 波形分析页面的JavaScript功能
let currentScale = 1;
let waveChart = null;

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
  // 初始化图表
  initChart();
  
  // 设置菜单功能
  setupMenuOptions();
  
  // 设置移动端控制功能
  setupMobileControls();
  
  // 设置菜单激活状态
  setActiveMenuOptions();
  
  // 初始化缩放控制
  setupZoomControls();
  
  // 监听窗口调整
  window.addEventListener('resize', handleResize);
  
  // 设置表单提交事件
  setupFormEvents();
});

// 初始化波形图表
function initChart() {
  // 模拟数据生成
  function generateWaveData() {
    const data = [];
    let value = 0;
    for (let i = 0; i < 240; i++) {
      value = Math.sin(i * 0.2) + Math.sin(i * 0.05) + Math.sin(i * 0.01) * 0.2 + Math.random() * 0.3;
      data.push({x: i, y: value});
    }
    return data;
  }
  
  // 获取图表上下文
  const ctx = document.getElementById('waveChart').getContext('2d');
  
  // 创建图表
  waveChart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [{
        label: '电压信号 (V)',
        data: generateWaveData(),
        borderColor: '#3498db',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
        fill: false
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          type: 'linear',
          title: {
            display: true,
                text: '时间 (ms)',
                font: {
                  size: 13,
                  weight: 'bold'
                }
          },
          grid: {
            color: 'rgba(0, 0, 0, 0.05)'
          }
        },
        y: {
          title: {
            display: true,
            text: '电压 (V)',
            font: {
              size: 13,
              weight: 'bold'
            }
          },
          grid: {
            color: 'rgba(0, 0, 0, 0.05)'
          }
        }
      },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            font: {
              size: 13
            }
          }
        },
        tooltip: {
          mode: 'index',
          intersect: false
        }
      },
      animation: {
        duration: 1000
      }
    }
  });
}

// 设置菜单选项功能
function setupMenuOptions() {
  const menuOptions = document.querySelectorAll('.menu-option');
  menuOptions.forEach(option => {
    option.addEventListener('click', function() {
      // 移除所有激活状态
      menuOptions.forEach(opt => opt.classList.remove('active'));
      // 设置当前为激活状态
      this.classList.add('active');
    });
  });
}

// 移动端控制功能
function setupMobileControls() {
  const paramBar = document.querySelector('.param-bar');
  const paramsBtn = document.getElementById('paramsBtn');
  
  // 移动端按钮控制
  if (paramsBtn) {
    paramsBtn.addEventListener('click', function() {
      paramBar.classList.toggle('active');
    });
  }
  
  // 在屏幕上检测移动设备
  if (window.innerWidth <= 1200) {
    document.querySelector('.mobile-controls').style.display = 'flex';
    document.querySelector('.param-toggle').style.display = 'flex';
  }
}

// 设置缩放控制
function setupZoomControls() {
  const zoomInBtn = document.getElementById('zoomIn');
  const zoomOutBtn = document.getElementById('zoomOut');
  
  zoomInBtn.addEventListener('click', function() {
    zoomChart(0.1);
  });
  
  zoomOutBtn.addEventListener('click', function() {
    zoomChart(-0.1);
  });
}

// 缩放控制函数
function zoomChart(change) {
  const newScale = Math.max(0.5, Math.min(1.5, currentScale + change));
  
  // 更新body的缩放
  document.body.style.transform = `scale(${newScale})`;
  document.body.style.width = `${100 / newScale}%`;
  
  currentScale = newScale;
  
  // 更新图表以适应缩放变化
  if (waveChart) {
    waveChart.resize();
  }
}

// 处理窗口调整
function handleResize() {
  // 移动端参数控制
  if (window.innerWidth <= 1200) {
    document.querySelector('.mobile-controls').style.display = 'flex';
    document.querySelector('.param-toggle').style.display = 'flex';
  } else {
    document.querySelector('.mobile-controls').style.display = 'none';
    document.querySelector('.param-toggle').style.display = 'none';
    document.querySelector('.param-bar').classList.remove('active');
  }
  
  // 调整图表大小
  if (waveChart) {
    waveChart.resize();
  }
}

// 设置菜单激活状态
function setActiveMenuOptions() {
  const menuOptions = document.querySelectorAll('.menu-option');
  menuOptions.forEach(option => {
    option.addEventListener('click', function() {
      // 移除所有激活状态
      menuOptions.forEach(opt => opt.classList.remove('active'));
      // 设置当前为激活状态
      this.classList.add('active');
    });
  });
}

// 设置表单事件
function setupFormEvents() {
  const fileInput = document.getElementById('fileInput');
  const hiddenUploadBtn = document.getElementById('hiddenUploadBtn');
  
  if (fileInput && hiddenUploadBtn) {
    fileInput.addEventListener('change', function() {
      hiddenUploadBtn.click();
    });
  }
  
  const resetBtn = document.getElementById('resetBtn');
  if (resetBtn) {
    resetBtn.addEventListener('click', function() {
      document.querySelector('form').reset();
      if (waveChart) {
        waveChart.destroy();
        initChart();
      }
    });
  }
}

// 更新图表数据
function updateChart(data) {
  if (waveChart) {
    waveChart.data.datasets[0].data = data;
    waveChart.update();
  }
}

// 加载数据并更新图表
function loadDataAndUpdateChart(filename, column) {
  fetch(`/api/get_wave_data?filename=${encodeURIComponent(filename)}&column=${encodeURIComponent(column)}`)
    .then(response => response.json())
    .then(data => {
      updateChart(data);
    })
    .catch(error => {
      console.error('Error loading wave data:', error);
    });
}