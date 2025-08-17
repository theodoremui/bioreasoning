#!/bin/bash

# BioReasoning Infrastructure Startup Script
# This script starts all required infrastructure services using Docker Compose

set -e  # Exit on any error

# Colors for output
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

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to check if Docker Compose is available
check_docker_compose() {
    if ! command -v docker-compose > /dev/null 2>&1; then
        print_error "Docker Compose is not installed. Please install Docker Compose and try again."
        exit 1
    fi
    print_success "Docker Compose is available"
}

# Function to check for port conflicts
check_ports() {
    local ports=("5432" "16686" "4317" "4318" "8080")
    local conflicts=()
    
    for port in "${ports[@]}"; do
        if netstat -ano | grep -q ":$port "; then
            conflicts+=("$port")
        fi
    done
    
    if [ ${#conflicts[@]} -ne 0 ]; then
        print_warning "Port conflicts detected on: ${conflicts[*]}"
        print_warning "Some services may fail to start. Consider stopping conflicting services."
    else
        print_success "No port conflicts detected"
    fi
}

# Function to start services
start_services() {
    print_status "Starting infrastructure services..."
    
    # Start services in background
    docker-compose up -d
    
    if [ $? -eq 0 ]; then
        print_success "Services started successfully"
    else
        print_error "Failed to start services"
        exit 1
    fi
}

# Function to wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for PostgreSQL
    print_status "Waiting for PostgreSQL..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose exec postgres pg_isready -U llama -d notebookllama > /dev/null 2>&1; then
            print_success "PostgreSQL is ready"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            print_error "PostgreSQL failed to start within timeout"
            exit 1
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    # Wait for Jaeger
    print_status "Waiting for Jaeger..."
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:16686/api/services > /dev/null 2>&1; then
            print_success "Jaeger is ready"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            print_error "Jaeger failed to start within timeout"
            exit 1
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    # Wait for Adminer
    print_status "Waiting for Adminer..."
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8080 > /dev/null 2>&1; then
            print_success "Adminer is ready"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            print_error "Adminer failed to start within timeout"
            exit 1
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
}

# Function to display service information
show_service_info() {
    echo
    print_success "Infrastructure services are ready!"
    echo
    echo "Service Information:"
    echo "==================="
    echo "📊 Jaeger Tracing UI:     http://localhost:16686"
    echo "🗄️  Adminer (Database):   http://localhost:8080"
    echo "🔌 PostgreSQL:            localhost:5432"
    echo "📡 OpenTelemetry HTTP:    localhost:4318"
    echo "📡 OpenTelemetry gRPC:    localhost:4317"
    echo
    echo "Database Credentials:"
    echo "===================="
    echo "Username: llama"
    echo "Password: S*********1"
    echo "Database: notebookllama"
    echo
    echo "Next Steps:"
    echo "==========="
    echo "1. Set up your .env file with API keys"
    echo "2. Start the MCP server: ./scripts/run-docs-server.sh"
    echo "3. Start the Streamlit client: ./scripts/run-docs-client.sh"
    echo
}

# Function to check service status
check_service_status() {
    print_status "Checking service status..."
    
    echo
    echo "Service Status:"
    echo "==============="
    docker-compose ps
    
    echo
    echo "Recent Logs:"
    echo "============"
    docker-compose logs --tail=10
}

# Main script
main() {
    echo "🚀 BioReasoning Infrastructure Startup"
    echo "====================================="
    echo
    
    # Check prerequisites
    check_docker
    check_docker_compose
    check_ports
    
    # Start services
    start_services
    
    # Wait for services to be ready
    wait_for_services
    
    # Show service information
    show_service_info
    
    # Check final status
    check_service_status
}

# Handle command line arguments
case "${1:-}" in
    "status")
        check_service_status
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "stop")
        print_status "Stopping infrastructure services..."
        docker-compose down
        print_success "Services stopped"
        ;;
    "restart")
        print_status "Restarting infrastructure services..."
        docker-compose down
        docker-compose up -d
        print_success "Services restarted"
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo
        echo "Commands:"
        echo "  (no args)  Start infrastructure services"
        echo "  status     Check service status"
        echo "  logs       Show service logs"
        echo "  stop       Stop all services"
        echo "  restart    Restart all services"
        echo "  help       Show this help message"
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac 