#!/bin/bash

# Image Similarity Search System Setup Script
# This script helps set up and run the complete image similarity search application

set -e  # Exit on any error

echo "=== Image Similarity Search System Setup ==="
echo "This script will set up your image similarity search application."
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if ! command -v python &> /dev/null; then
        if ! command -v python3 &> /dev/null; then
            print_error "Python is not installed. Please install Python 3.8 or higher."
            exit 1
        else
            PYTHON_CMD="python3"
        fi
    else
        PYTHON_CMD="python"
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    print_success "Python $PYTHON_VERSION found"
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Check if pip is available
    if ! $PYTHON_CMD -m pip --version &> /dev/null; then
        print_error "pip is not installed. Please install pip first."
        exit 1
    fi
    
    # Install dependencies
    if [ -f "requirements.txt" ]; then
        print_status "Installing from requirements.txt..."
        $PYTHON_CMD -m pip install -r requirements.txt
        print_success "Dependencies installed successfully"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Check PostgreSQL connection
check_database() {
    print_status "Checking database connection..."
    
    # Try to connect using Python
    $PYTHON_CMD -c "
import psycopg2
try:
    conn = psycopg2.connect('host=localhost port=5432 dbname=imsrc user=postgres password=14789')
    conn.close()
    print('Database connection successful')
except Exception as e:
    print(f'Database connection failed: {e}')
    exit(1)
" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        print_success "Database connection verified"
    else
        print_warning "Database connection failed. Please ensure PostgreSQL is running and configured correctly."
        print_warning "Default connection: host=localhost port=5432 dbname=imsrc user=postgres password=14789"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Setup database schema
setup_database() {
    print_status "Setting up database schema..."
    
    if [ -f "db/create_table.sql" ]; then
        $PYTHON_CMD db/init_system.py --setup-db --schema-path db/create_table.sql
        if [ $? -eq 0 ]; then
            print_success "Database schema setup completed"
        else
            print_error "Database schema setup failed"
            exit 1
        fi
    else
        print_error "Database schema file not found: db/create_table.sql"
        exit 1
    fi
}

# Populate with CIFAR-10 data
populate_data() {
    print_status "Populating database with CIFAR-10 images..."
    
    read -p "How many images would you like to populate? (default: 1000, max: 50000): " num_images
    num_images=${num_images:-1000}
    
    if [ -f "db/populate_cifar10.py" ]; then
        $PYTHON_CMD db/populate_cifar10.py --num-images $num_images
        if [ $? -eq 0 ]; then
            print_success "Data population completed"
        else
            print_error "Data population failed"
            exit 1
        fi
    else
        print_error "Population script not found: db/populate_cifar10.py"
        exit 1
    fi
}

# Initialize the system
initialize_system() {
    print_status "Initializing vector search system..."
    
    if [ -f "db/init_system.py" ]; then
        $PYTHON_CMD db/init_system.py --all
        if [ $? -eq 0 ]; then
            print_success "System initialization completed"
        else
            print_error "System initialization failed"
            exit 1
        fi
    else
        print_error "Initialization script not found: db/init_system.py"
        exit 1
    fi
}

# Start the API server
start_server() {
    print_status "Starting API server..."
    
    if [ -f "backend/main.py" ]; then
        print_status "Server will start on http://localhost:8000"
        print_status "Frontend will be available at http://localhost:8000/app"
        print_status "API documentation at http://localhost:8000/docs"
        print_status ""
        print_status "Press Ctrl+C to stop the server"
        
        # Start the server
        cd backend
        $PYTHON_CMD -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    else
        print_error "Main application not found: backend/main.py"
        exit 1
    fi
}

# Main menu
show_menu() {
    echo ""
    echo "Please select an option:"
    echo "1. Complete setup (recommended for first time)"
    echo "2. Install dependencies only"
    echo "3. Setup database schema only"
    echo "4. Populate with CIFAR-10 data only"
    echo "5. Initialize system (vectors, prototypes, FAISS index)"
    echo "6. Start API server"
    echo "7. Run system health check"
    echo "8. Exit"
    echo ""
}

# System health check
health_check() {
    print_status "Running system health check..."
    
    if [ -f "db/init_system.py" ]; then
        $PYTHON_CMD db/init_system.py --check
    else
        print_error "Health check script not found"
        exit 1
    fi
}

# Main execution
main() {
    check_python
    
    while true; do
        show_menu
        read -p "Enter your choice (1-8): " choice
        
        case $choice in
            1)
                print_status "Starting complete setup..."
                install_dependencies
                check_database
                setup_database
                populate_data
                initialize_system
                print_success "Complete setup finished!"
                print_status "You can now start the server with option 6"
                ;;
            2)
                install_dependencies
                ;;
            3)
                check_database
                setup_database
                ;;
            4)
                check_database
                populate_data
                ;;
            5)
                initialize_system
                ;;
            6)
                start_server
                ;;
            7)
                health_check
                ;;
            8)
                print_status "Goodbye!"
                exit 0
                ;;
            *)
                print_warning "Invalid option. Please try again."
                ;;
        esac
    done
}

# Run main function
main