@echo off
REM Image Similarity Search System Setup Script for Windows
REM This script helps set up and run the complete image similarity search application

echo === Image Similarity Search System Setup ===
echo This script will set up your image similarity search application.
echo.

REM Check if Python is installed
echo [INFO] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    python3 --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Python is not installed. Please install Python 3.8 or higher.
        pause
        exit /b 1
    ) else (
        set PYTHON_CMD=python3
    )
) else (
    set PYTHON_CMD=python
)

for /f "tokens=2" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [SUCCESS] Python %PYTHON_VERSION% found

:menu
echo.
echo Please select an option:
echo 1. Complete setup (recommended for first time)
echo 2. Install dependencies only
echo 3. Setup database schema only
echo 4. Populate with CIFAR-10 data only
echo 5. Initialize system (vectors, prototypes, FAISS index)
echo 6. Start API server
echo 7. Run system health check
echo 8. Exit
echo.

set /p choice="Enter your choice (1-8): "

if "%choice%"=="1" goto complete_setup
if "%choice%"=="2" goto install_deps
if "%choice%"=="3" goto setup_db
if "%choice%"=="4" goto populate_data
if "%choice%"=="5" goto init_system
if "%choice%"=="6" goto start_server
if "%choice%"=="7" goto health_check
if "%choice%"=="8" goto exit_script
echo [WARNING] Invalid option. Please try again.
goto menu

:complete_setup
echo [INFO] Starting complete setup...
call :install_dependencies
if %errorlevel% neq 0 exit /b 1
call :check_database
call :setup_database
if %errorlevel% neq 0 exit /b 1
call :populate_database
if %errorlevel% neq 0 exit /b 1
call :initialize_system
if %errorlevel% neq 0 exit /b 1
echo [SUCCESS] Complete setup finished!
echo [INFO] You can now start the server with option 6
goto menu

:install_deps
call :install_dependencies
goto menu

:setup_db
call :check_database
call :setup_database
goto menu

:populate_data
call :check_database
call :populate_database
goto menu

:init_system
call :initialize_system
goto menu

:start_server
call :start_api_server
goto menu

:health_check
call :health_check_system
goto menu

:exit_script
echo [INFO] Goodbye!
exit /b 0

REM Functions

:install_dependencies
echo [INFO] Installing Python dependencies...
%PYTHON_CMD% -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pip is not installed. Please install pip first.
    exit /b 1
)

if exist "requirements.txt" (
    echo [INFO] Installing from requirements.txt...
    %PYTHON_CMD% -m pip install -r requirements.txt
    if %errorlevel% eq 0 (
        echo [SUCCESS] Dependencies installed successfully
    ) else (
        echo [ERROR] Failed to install dependencies
        exit /b 1
    )
) else (
    echo [ERROR] requirements.txt not found
    exit /b 1
)
exit /b 0

:check_database
echo [INFO] Checking database connection...
%PYTHON_CMD% -c "import psycopg2; conn = psycopg2.connect('host=localhost port=5432 dbname=imsrc user=postgres password=14789'); conn.close(); print('Database connection successful')" 2>nul
if %errorlevel% eq 0 (
    echo [SUCCESS] Database connection verified
) else (
    echo [WARNING] Database connection failed. Please ensure PostgreSQL is running and configured correctly.
    echo [WARNING] Default connection: host=localhost port=5432 dbname=imsrc user=postgres password=14789
    set /p continue="Continue anyway? (y/N): "
    if /i not "%continue%"=="y" exit /b 1
)
exit /b 0

:setup_database
echo [INFO] Setting up database schema...
if exist "db\create_table.sql" (
    %PYTHON_CMD% db\init_system.py --setup-db --schema-path db\create_table.sql
    if %errorlevel% eq 0 (
        echo [SUCCESS] Database schema setup completed
    ) else (
        echo [ERROR] Database schema setup failed
        exit /b 1
    )
) else (
    echo [ERROR] Database schema file not found: db\create_table.sql
    exit /b 1
)
exit /b 0

:populate_database
echo [INFO] Populating database with CIFAR-10 images...
set /p num_images="How many images would you like to populate? (default: 1000, max: 50000): "
if "%num_images%"=="" set num_images=1000

if exist "db\populate_cifar10.py" (
    %PYTHON_CMD% db\populate_cifar10.py --num-images %num_images%
    if %errorlevel% eq 0 (
        echo [SUCCESS] Data population completed
    ) else (
        echo [ERROR] Data population failed
        exit /b 1
    )
) else (
    echo [ERROR] Population script not found: db\populate_cifar10.py
    exit /b 1
)
exit /b 0

:initialize_system
echo [INFO] Initializing vector search system...
if exist "db\init_system.py" (
    %PYTHON_CMD% db\init_system.py --all
    if %errorlevel% eq 0 (
        echo [SUCCESS] System initialization completed
    ) else (
        echo [ERROR] System initialization failed
        exit /b 1
    )
) else (
    echo [ERROR] Initialization script not found: db\init_system.py
    exit /b 1
)
exit /b 0

:start_api_server
echo [INFO] Starting API server...
if exist "backend\main.py" (
    echo [INFO] Server will start on http://localhost:8000
    echo [INFO] Frontend will be available at http://localhost:8000/app
    echo [INFO] API documentation at http://localhost:8000/docs
    echo.
    echo [INFO] Press Ctrl+C to stop the server
    cd backend
    %PYTHON_CMD% -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    cd ..
) else (
    echo [ERROR] Main application not found: backend\main.py
    exit /b 1
)
exit /b 0

:health_check_system
echo [INFO] Running system health check...
if exist "db\init_system.py" (
    %PYTHON_CMD% db\init_system.py --check
) else (
    echo [ERROR] Health check script not found
    exit /b 1
)
exit /b 0