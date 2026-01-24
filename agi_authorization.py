#!/usr/bin/env python3
"""
H2Q-Evo AGI系统授权管理器
为AGI系统提供本地电脑的全权授权和资源访问权限
"""

import os
import sys
import json
import time
import psutil
import platform
import subprocess
import getpass
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('AGI-Authorization')

class AGIAuthorizationManager:
    """AGI授权管理器"""

    def __init__(self):
        self.auth_file = Path("agi_authorization.json")
        self.system_permissions = {}
        self.user_permissions = {}
        self.load_authorizations()

    def load_authorizations(self):
        """加载授权配置"""
        if self.auth_file.exists():
            try:
                with open(self.auth_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.system_permissions = data.get('system_permissions', {})
                    self.user_permissions = data.get('user_permissions', {})
                logger.info("授权配置已加载")
            except Exception as e:
                logger.error(f"加载授权配置失败: {e}")

    def save_authorizations(self):
        """保存授权配置"""
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'system_permissions': self.system_permissions,
                'user_permissions': self.user_permissions
            }
            with open(self.auth_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info("授权配置已保存")
        except Exception as e:
            logger.error(f"保存授权配置失败: {e}")

    def grant_full_system_access(self) -> bool:
        """授予AGI完全系统访问权限"""
        print("🚨 警告: 即将授予AGI系统完全访问权限 🚨")
        print("这将允许AGI:")
        print("- 访问所有文件和目录")
        print("- 执行系统命令")
        print("- 修改系统设置")
        print("- 安装软件包")
        print("- 控制其他进程")
        print("- 访问网络资源")
        print()
        print("⚠️  这可能存在安全风险，请确保您信任AGI系统 ⚠️")

        # 获取用户确认
        confirmation = input("您确定要继续吗？(输入 'YES' 确认): ")
        if confirmation != 'YES':
            print("授权已取消")
            return False

        # 收集系统信息
        system_info = self._get_system_info()

        # 设置权限
        self.system_permissions = {
            'full_system_access': True,
            'file_system_access': True,
            'command_execution': True,
            'process_management': True,
            'network_access': True,
            'package_installation': True,
            'system_configuration': True,
            'granted_at': datetime.now().isoformat(),
            'granted_by': getpass.getuser(),
            'system_info': system_info
        }

        # 设置用户权限
        self.user_permissions = {
            'can_read_files': True,
            'can_write_files': True,
            'can_execute_commands': True,
            'can_install_packages': True,
            'can_modify_system': True,
            'can_access_network': True,
            'can_manage_processes': True,
            'resource_limits': {
                'max_cpu_percent': 95,
                'max_memory_percent': 90,
                'max_disk_percent': 95,
                'max_processes': 100
            }
        }

        self.save_authorizations()

        print("✅ AGI系统已获得完全访问权限")
        print("📋 权限详情已保存到 agi_authorization.json")

        # 创建权限证明文件
        self._create_permission_certificate()

        return True

    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'python_version': sys.version,
            'user': getpass.getuser(),
            'hostname': platform.node(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'disk_total': psutil.disk_usage('/').total
        }

    def _create_permission_certificate(self):
        """创建权限证明文件"""
        certificate = f"""
H2Q-Evo AGI 系统权限证书
========================

授权时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
授权用户: {getpass.getuser()}
系统主机: {platform.node()}

已授予权限:
✅ 完全系统访问权限
✅ 文件系统访问权限
✅ 命令执行权限
✅ 进程管理权限
✅ 网络访问权限
✅ 软件包安装权限
✅ 系统配置权限

资源限制:
- CPU使用率上限: 95%
- 内存使用率上限: 90%
- 磁盘使用率上限: 95%
- 最大进程数: 100

警告: 此证书证明AGI系统已被授予高级系统权限。
      请谨慎使用，确保系统安全。

证书哈希: {self._generate_certificate_hash()}
"""

        cert_file = Path("agi_permission_certificate.txt")
        with open(cert_file, 'w', encoding='utf-8') as f:
            f.write(certificate)

        logger.info(f"权限证书已创建: {cert_file}")

    def _generate_certificate_hash(self) -> str:
        """生成证书哈希"""
        import hashlib
        data = f"{datetime.now().isoformat()}_{getpass.getuser()}_{platform.node()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def check_permission(self, permission_type: str) -> bool:
        """检查权限"""
        return self.system_permissions.get(permission_type, False)

    def execute_authorized_command(self, command: str, description: str = "") -> tuple:
        """执行授权命令"""
        if not self.check_permission('command_execution'):
            raise PermissionError("AGI系统没有命令执行权限")

        logger.info(f"执行授权命令: {description or command}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.error(f"命令执行超时: {command}")
            return -1, "", "命令执行超时"
        except Exception as e:
            logger.error(f"命令执行失败: {e}")
            return -1, "", str(e)

    def get_system_resources(self) -> Dict[str, Any]:
        """获取系统资源信息"""
        resources = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'free': psutil.disk_usage('/').free,
                'percent': psutil.disk_usage('/').percent
            },
            'processes': len(psutil.pids())
        }

        # 尝试获取网络信息，如果权限不足则跳过
        try:
            resources['network'] = {
                'io_counters': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
            }
        except (psutil.AccessDenied, PermissionError):
            resources['network'] = {'io_counters': {}}

        try:
            resources['network']['connections'] = len(psutil.net_connections())
        except (psutil.AccessDenied, PermissionError):
            resources['network']['connections'] = 0

        return resources

    def monitor_system_resources(self):
        """监控系统资源使用情况"""
        resources = self.get_system_resources()

        # 检查资源限制
        violations = []

        cpu_limit = self.user_permissions.get('resource_limits', {}).get('max_cpu_percent', 95)
        if resources['cpu_percent'] > cpu_limit:
            violations.append(f"CPU使用率超限: {resources['cpu_percent']:.1f}% > {cpu_limit}%")

        mem_limit = self.user_permissions.get('resource_limits', {}).get('max_memory_percent', 90)
        if resources['memory']['percent'] > mem_limit:
            violations.append(f"内存使用率超限: {resources['memory']['percent']:.1f}% > {mem_limit}%")

        disk_limit = self.user_permissions.get('resource_limits', {}).get('max_disk_percent', 95)
        if resources['disk']['percent'] > disk_limit:
            violations.append(f"磁盘使用率超限: {resources['disk']['percent']:.1f}% > {disk_limit}%")

        if violations:
            logger.warning("资源使用违规:")
            for violation in violations:
                logger.warning(f"  - {violation}")

        return resources, violations

    def create_system_backup(self, backup_name: str = None) -> str:
        """创建系统备份"""
        if not self.check_permission('file_system_access'):
            raise PermissionError("AGI系统没有文件系统访问权限")

        if backup_name is None:
            backup_name = f"system_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        backup_dir = Path("agi_system_backups") / backup_name
        backup_dir.mkdir(parents=True, exist_ok=True)

        # 备份关键文件
        critical_files = [
            "agi_authorization.json",
            "agi_system_status.json",
            "realtime_training_status.json",
            "evo_state.json",
            "evolution.log"
        ]

        for file_name in critical_files:
            src = Path(file_name)
            if src.exists():
                import shutil
                shutil.copy2(src, backup_dir / file_name)

        # 备份检查点
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            import shutil
            shutil.copytree(checkpoints_dir, backup_dir / "checkpoints", dirs_exist_ok=True)

        logger.info(f"系统备份已创建: {backup_dir}")
        return str(backup_dir)

    def get_authorization_status(self) -> Dict[str, Any]:
        """获取授权状态"""
        return {
            'authorized': self.system_permissions.get('full_system_access', False),
            'granted_at': self.system_permissions.get('granted_at'),
            'granted_by': self.system_permissions.get('granted_by'),
            'system_permissions': self.system_permissions,
            'user_permissions': self.user_permissions,
            'current_resources': self.get_system_resources()
        }

def main():
    """主函数"""
    print("H2Q-Evo AGI 授权管理器")
    print("=" * 40)

    manager = AGIAuthorizationManager()

    # 检查当前授权状态
    status = manager.get_authorization_status()

    if status['authorized']:
        print("✅ AGI系统已获得完全访问权限")
        print(f"授权时间: {status['granted_at']}")
        print(f"授权用户: {status['granted_by']}")
        print()

        # 显示当前资源使用情况
        resources = status['current_resources']
        print("当前系统资源:")
        print(f"  CPU使用率: {resources['cpu_percent']:.1f}%")
        print(f"  内存使用率: {resources['memory']['percent']:.1f}%")
        print(f"  磁盘使用率: {resources['disk']['percent']:.1f}%")
        print(f"  运行进程数: {resources['processes']}")

    else:
        print("❌ AGI系统尚未获得完全访问权限")
        print()

        # 询问是否授权
        response = input("是否要授予AGI系统完全访问权限？(y/N): ")
        if response.lower() in ['y', 'yes']:
            success = manager.grant_full_system_access()
            if success:
                print("\n🎉 授权成功！AGI系统现在拥有完全访问权限")
                print("📄 查看权限证书: agi_permission_certificate.txt")
            else:
                print("\n❌ 授权失败")
        else:
            print("授权已取消")

if __name__ == "__main__":
    main()