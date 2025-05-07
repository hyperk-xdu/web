/* 侧边栏 & 高亮逻辑  */
document.addEventListener('DOMContentLoaded', () => {
    const toggleBtn = document.getElementById('toggle-menu');
    const sidebar   = document.getElementById('sidebar');
    const mask      = document.getElementById('menu-mask');
  
    const openSidebar  = () => sidebar.classList.add('open');
    const closeSidebar = () => sidebar.classList.remove('open');
  
    toggleBtn.addEventListener('click', openSidebar);
    mask.addEventListener('click',  closeSidebar);
  
    /* 自动在桌面端关闭侧边栏（防止窗口拉伸错位） */
    window.addEventListener('resize', () => {
      if (window.innerWidth > 768) closeSidebar();
    });
  });
  