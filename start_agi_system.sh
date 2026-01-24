#!/bin/bash
# H2Q-Evo AGI系统启动脚本
# 用于快速启动完整的AGI训练和进化系统

set -e  # 遇到错误立即退出

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
CONFIG_FILE="$PROJECT_ROOT/agi_training_config.ini"
LOG_DIR="$PROJECT_ROOT/agi_persistent_training/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN: $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# 检查依赖
check_dependencies() {
    log_info "检查系统依赖..."

    # 检查Python
    if ! command -v python3 &> /dev/null; then
        log_error "未找到python3，请安装Python 3.8+"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $(echo "$PYTHON_VERSION < 3.8" | bc -l) -eq 1 ]]; then
        log_error "Python版本过低，需要3.8+，当前版本: $PYTHON_VERSION"
        exit 1
    fi
    log_info "Python版本: $PYTHON_VERSION ✓"

    # 检查pip
    if ! command -v pip3 &> /dev/null; then
        log_error "未找到pip3，请安装pip"
        exit 1
    fi
    log_info "pip ✓"

    # 检查必要的Python包
    REQUIRED_PACKAGES=("torch" "transformers" "datasets" "accelerate" "wandb")
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if ! python3 -c "import $package" &> /dev/null; then
            log_warn "缺少Python包: $package，尝试安装..."
            pip3 install "$package" || {
                log_error "安装 $package 失败"
                exit 1
            }
        fi
    done
    log_info "Python依赖检查完成 ✓"
}

# 检查配置文件
check_config() {
    log_info "检查配置文件..."

    if [ ! -f "$CONFIG_FILE" ]; then
        log_warn "配置文件不存在: $CONFIG_FILE，创建默认配置..."
        cat > "$CONFIG_FILE" << EOF
[system]
auto_restart = true
max_restarts = 3
health_check_interval = 30

[training]
enabled = true
model_name = microsoft/DialoGPT-medium
batch_size = 8
learning_rate = 0.001
max_epochs = 100
save_steps = 500

[manifold_encoding]
resolution = 0.01
compression_target = 0.85

[evolution]
enabled = true
population_size = 10
mutation_rate = 0.1
crossover_rate = 0.8

[monitoring]
enabled = true
update_interval = 5
alert_threshold_loss = 10.0
alert_threshold_memory = 90.0

[data_generation]
enabled = true
generation_interval = 3600
samples_per_generation = 1000

[logging]
level = INFO
max_log_files = 10
log_rotation = daily
EOF
        log_info "默认配置文件已创建"
    else
        log_info "配置文件存在 ✓"
    fi
}

# 创建目录结构
create_directories() {
    log_info "创建目录结构..."

    DIRECTORIES=(
        "$PROJECT_ROOT/agi_persistent_training"
        "$PROJECT_ROOT/agi_persistent_training/models"
        "$PROJECT_ROOT/agi_persistent_training/data"
        "$PROJECT_ROOT/agi_persistent_training/metrics"
        "$PROJECT_ROOT/agi_persistent_training/logs"
        "$PROJECT_ROOT/agi_persistent_training/reports"
        "$PROJECT_ROOT/agi_persistent_training/checkpoints"
    )

    for dir in "${DIRECTORIES[@]}"; do
        mkdir -p "$dir"
        log_info "创建目录: $dir ✓"
    done
}

# 启动系统
start_system() {
    log_info "启动H2Q-Evo AGI系统..."

    # 检查是否已在运行
    if pgrep -f "agi_system_manager.py" > /dev/null; then
        log_warn "AGI系统似乎已在运行"
        read -p "是否要重启系统? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "取消启动"
            exit 0
        fi

        # 停止现有系统
        stop_system
        sleep 3
    fi

    # 启动系统管理器
    log_info "启动系统管理器..."
    nohup python3 "$PROJECT_ROOT/agi_system_manager.py" start --background \
        --config "$CONFIG_FILE" > "$LOG_DIR/system_startup_$TIMESTAMP.log" 2>&1 &

    SYSTEM_PID=$!
    echo $SYSTEM_PID > "$PROJECT_ROOT/agi_system.pid"

    # 等待系统启动
    sleep 5

    # 检查是否成功启动
    if kill -0 $SYSTEM_PID 2>/dev/null; then
        log_info "✅ AGI系统启动成功 (PID: $SYSTEM_PID)"

        # 显示状态
        show_status

        log_info "系统日志: $LOG_DIR/system_startup_$TIMESTAMP.log"
        log_info "使用以下命令查看状态:"
        log_info "  python3 agi_system_manager.py status"
        log_info "  python3 agi_evolution_monitor.py --mode status"
        log_info "使用以下命令停止系统:"
        log_info "  python3 agi_system_manager.py stop"
        log_info "  或 ./start_agi_system.sh stop"

    else
        log_error "❌ AGI系统启动失败，请检查日志: $LOG_DIR/system_startup_$TIMESTAMP.log"
        exit 1
    fi
}

# 停止系统
stop_system() {
    log_info "停止AGI系统..."

    # 读取PID文件
    if [ -f "$PROJECT_ROOT/agi_system.pid" ]; then
        SYSTEM_PID=$(cat "$PROJECT_ROOT/agi_system.pid")

        if kill -0 $SYSTEM_PID 2>/dev/null; then
            log_info "终止系统进程 (PID: $SYSTEM_PID)..."
            kill $SYSTEM_PID

            # 等待进程结束
            for i in {1..10}; do
                if ! kill -0 $SYSTEM_PID 2>/dev/null; then
                    break
                fi
                sleep 1
            done

            if kill -0 $SYSTEM_PID 2>/dev/null; then
                log_warn "强制终止进程..."
                kill -9 $SYSTEM_PID
            fi
        else
            log_warn "系统进程已不存在"
        fi

        rm -f "$PROJECT_ROOT/agi_system.pid"
    fi

    # 停止所有相关进程
    log_info "清理相关进程..."
    pkill -f "agi_persistent_evolution.py" || true
    pkill -f "agi_training_monitor.py" || true
    pkill -f "agi_evolution_monitor.py" || true
    pkill -f "agi_data_generator.py" || true

    log_info "✅ AGI系统已停止"
}

# 显示状态
show_status() {
    log_info "获取系统状态..."

    if [ -f "$PROJECT_ROOT/agi_system.pid" ]; then
        SYSTEM_PID=$(cat "$PROJECT_ROOT/agi_system.pid")
        if kill -0 $SYSTEM_PID 2>/dev/null; then
            log_info "系统管理器运行中 (PID: $SYSTEM_PID) ✓"
        else
            log_error "系统管理器进程不存在 ✗"
            rm -f "$PROJECT_ROOT/agi_system.pid"
        fi
    else
        log_warn "未找到系统PID文件"
    fi

    # 显示详细状态
    python3 "$PROJECT_ROOT/agi_system_manager.py" status 2>/dev/null || {
        log_warn "无法获取详细状态"
    }
}

# 生成报告
generate_report() {
    log_info "生成系统报告..."

    python3 "$PROJECT_ROOT/agi_system_manager.py" report 2>/dev/null && {
        log_info "✅ 系统报告生成完成"
    } || {
        log_error "❌ 系统报告生成失败"
    }
}

# 验证核心算法使用情况
verify_core_algorithm() {
    log_info "验证核心算法使用情况..."

    python3 "$PROJECT_ROOT/verify_agi_algorithm.py" --quiet
    if [ $? -eq 0 ]; then
        log_info "✅ 核心算法验证通过 - 诚实的AGI实验"
    else
        log_error "❌ 核心算法验证失败 - 请检查算法集成"
        exit 1
    fi
}

# 显示帮助
show_help() {
    cat << EOF
H2Q-Evo AGI系统启动脚本

用法: $0 <命令> [选项]

命令:
    start       启动AGI系统
    stop        停止AGI系统
    status      显示系统状态
    report      生成系统报告
    cleanup     清理旧日志和数据
    help        显示此帮助信息

选项:
    --background    后台运行 (仅用于start命令)

示例:
    $0 start              # 启动系统
    $0 start --background # 后台启动系统
    $0 stop               # 停止系统
    $0 status             # 查看状态
    $0 report             # 生成报告
    $0 cleanup            # 清理文件

系统文件位置:
    配置: $CONFIG_FILE
    日志: $LOG_DIR/
    数据: $PROJECT_ROOT/agi_persistent_training/
    PID:  $PROJECT_ROOT/agi_system.pid

EOF
}

# 主函数
main() {
    # 检查参数
    if [ $# -eq 0 ]; then
        show_help
        exit 1
    fi

    COMMAND=$1
    shift

    case $COMMAND in
        start)
            BACKGROUND=false
            while [[ $# -gt 0 ]]; do
                case $1 in
                    --background)
                        BACKGROUND=true
                        shift
                        ;;
                    *)
                        log_error "未知选项: $1"
                        show_help
                        exit 1
                        ;;
                esac
            done

            check_dependencies
            check_config
            create_directories
            verify_core_algorithm  # 验证核心算法使用
            start_system

            if [ "$BACKGROUND" = false ]; then
                log_info "系统正在前台运行，按 Ctrl+C 停止..."
                trap stop_system INT
                wait
            fi
            ;;
        stop)
            stop_system
            ;;
        status)
            show_status
            ;;
        report)
            generate_report
            ;;
        cleanup)
            cleanup_logs
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "未知命令: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"