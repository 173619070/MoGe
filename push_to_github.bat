@echo off
cd /d "E:\MoGe"
echo 正在推送到 GitHub: https://github.com/173619070/MoGe.git ...
git push -u origin main
if %ERRORLEVEL% equ 0 (
    echo.
    echo [成功] 代码已成功推送到 https://github.com/173619070/MoGe !
) else (
    echo.
    echo [提示] 如果提示权限不足或需要登录，请按提示完成浏览器授权或输入 GitHub Personal Access Token。
)
pause
