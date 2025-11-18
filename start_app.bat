@echo off
echo ============================================================
echo 🚀 STARTING IMAGE SIMILARITY SEARCH APPLICATION
echo ============================================================

echo.
echo 📋 What would you like to do?
echo 1. Initialize Admin System (run once)
echo 2. Start Backend Server
echo 3. Both - Initialize + Start Server
echo 4. Open Admin Panel
echo 5. Open Main App
echo 6. Exit

set /p choice=Enter your choice (1-6): 

if "%choice%"=="1" goto :init
if "%choice%"=="2" goto :start
if "%choice%"=="3" goto :both
if "%choice%"=="4" goto :admin
if "%choice%"=="5" goto :main
if "%choice%"=="6" goto :exit
goto :invalid

:init
echo.
echo 🔧 Initializing Admin System...
python init_admin_system.py
pause
goto :menu

:start
echo.
echo 🚀 Starting Backend Server...
echo 📌 Server will run at: http://localhost:8000
echo 📌 API Documentation: http://localhost:8000/docs
echo 📌 Admin Panel: frontend/admin.html
echo 📌 Main App: frontend/index.html
echo.
echo Press Ctrl+C to stop the server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
goto :exit

:both
echo.
echo 🔧 Step 1: Initializing Admin System...
python init_admin_system.py
if errorlevel 1 (
    echo ❌ Initialization failed! Check the errors above.
    pause
    goto :menu
)
echo.
echo ✅ Admin system initialized successfully!
echo.
echo 🚀 Step 2: Starting Backend Server...
echo 📌 Server will run at: http://localhost:8000
echo 📌 Admin Panel: frontend/admin.html (login: admin/admin123)
echo 📌 Main App: frontend/index.html
echo.
echo Press Ctrl+C to stop the server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
goto :exit

:admin
echo.
echo 🔓 Opening Admin Panel...
echo 📌 Default Login: admin / admin123
start frontend/admin.html
goto :menu

:main
echo.
echo 🖼️ Opening Main Application...
start frontend/index.html
goto :menu

:invalid
echo.
echo ❌ Invalid choice. Please enter 1-6.
echo.

:menu
goto :start_menu

:start_menu
echo.
echo ============================================================
goto :start

:exit
echo.
echo 👋 Goodbye! Your Image Similarity Search App is ready!
echo.
echo 📋 Quick Reference:
echo - Admin Panel: frontend/admin.html (admin/admin123)
echo - Main App: frontend/index.html  
echo - Backend: python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
echo - Docs: http://localhost:8000/docs
echo.
pause
exit