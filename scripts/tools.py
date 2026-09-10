"""
辅助工具
数据转换、日志可视化、模型导出等
"""

import numpy as np
import json
import time
import os
import sys
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import argparse
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataConverter:
    """数据转换器"""
    
    @staticmethod
    def numpy_to_json(data: np.ndarray) -> List:
        """numpy数组转JSON"""
        return data.tolist()
    
    @staticmethod
    def json_to_numpy(data: List) -> np.ndarray:
        """JSON转numpy数组"""
        return np.array(data)
    
    @staticmethod
    def save_array(data: np.ndarray, filepath: str):
        """保存numpy数组"""
        np.save(filepath, data)
    
    @staticmethod
    def load_array(filepath: str) -> np.ndarray:
        """加载numpy数组"""
        return np.load(filepath)


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str = 'output/logs'):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.entries = []
    
    def log(self, iteration: int, stats: Dict):
        """
        记录训练数据
        
        Args:
            iteration: 迭代次数
            stats: 统计信息
        """
        entry = {
            'iteration': iteration,
            'timestamp': time.time(),
            'stats': stats
        }
        
        self.entries.append(entry)
        
        # 实时写入
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def load_log(self, log_file: str):
        """加载日志文件"""
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    self.entries.append(json.loads(line.strip()))
    
    def get_plot_data(self) -> Dict:
        """获取绘图数据"""
        if not self.entries:
            return {}
        
        plot_data = {
            'iterations': [e['iteration'] for e in self.entries],
            'policy_loss': [e['stats'].get('policy_loss', 0) for e in self.entries],
            'value_loss': [e['stats'].get('value_loss', 0) for e in self.entries],
            'entropy': [e['stats'].get('entropy', 0) for e in self.entries],
            'approx_kl': [e['stats'].get('approx_kl', 0) for e in self.entries]
        }
        
        return plot_data
    
    def get_summary(self) -> Dict:
        """获取日志摘要"""
        if not self.entries:
            return {}
        
        last_stats = self.entries[-1]['stats']
        
        return {
            'num_entries': len(self.entries),
            'first_iteration': self.entries[0]['iteration'],
            'last_iteration': self.entries[-1]['iteration'],
            'policy_loss_range': (min(e['stats'].get('policy_loss', 0) for e in self.entries),
                                  max(e['stats'].get('policy_loss', 0) for e in self.entries)),
            'final_stats': last_stats
        }


class ModelExporter:
    """模型导出器"""
    
    def __init__(self, model_path: str):
        """
        初始化模型导出器
        
        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
    
    def export_to_onnx(self, output_path: str, input_dim: int = 132):
        """
        导出为ONNX格式
        
        Args:
            output_path: 输出路径
            input_dim: 输入维度
        """
        try:
            import torch
            from mjai.ppo import PPO
            
            # 加载模型
            ppo = PPO.load(self.model_path)
            model = ppo.policy_net
            model.eval()
            
            # 创建示例输入
            dummy_input = torch.randn(1, input_dim)
            
            # 导出ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=['input'],
                output_names=['action_probs', 'value'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'action_probs': {0: 'batch_size'},
                    'value': {0: 'batch_size'}
                }
            )
            
            logger.info(f"模型已导出为ONNX格式: {output_path}")
            return True
            
        except ImportError:
            logger.error("需要安装torch才能导出ONNX")
            return False
        except Exception as e:
            logger.error(f"导出失败: {e}")
            return False
    
    def export_to_torchscript(self, output_path: str, input_dim: int = 132):
        """
        导出为TorchScript格式
        
        Args:
            output_path: 输出路径
            input_dim: 输入维度
        """
        try:
            import torch
            from mjai.ppo import PPO
            
            # 加载模型
            ppo = PPO.load(self.model_path)
            model = ppo.policy_net
            model.eval()
            
            # 创建示例输入
            dummy_input = torch.randn(1, input_dim)
            
            # 转换为TorchScript
            traced_script_module = torch.jit.trace(model, dummy_input)
            
            # 保存
            traced_script_module.save(output_path)
            
            logger.info(f"模型已导出为TorchScript格式: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出失败: {e}")
            return False
    
    def export_architecture_info(self, output_path: str):
        """
        导出模型架构信息
        
        Args:
            output_path: 输出路径
        """
        try:
            from mjai.ppo import PPO
            import torchinfo
            
            # 加载模型
            ppo = PPO.load(self.model_path)
            
            # 获取模型信息
            model_info = {
                'model_path': self.model_path,
                'input_dim': 132,
                'action_dim': 34,
                'total_params': sum(p.numel() for p in ppo.policy_net.parameters()),
                'trainable_params': sum(p.numel() for p in ppo.policy_net.parameters() if p.requires_grad),
                'layers': []
            }
            
            # 遍历网络层
            for name, module in ppo.policy_net.named_modules():
                if name:
                    model_info['layers'].append({
                        'name': name,
                        'type': type(module).__name__,
                        'params': sum(p.numel() for p in module.parameters())
                    })
            
            # 保存
            with open(output_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            logger.info(f"模型架构信息已导出: {output_path}")
            return model_info
            
        except Exception as e:
            logger.error(f"导出失败: {e}")
            return None


class ProjectManager:
    """项目管理器"""
    
    @staticmethod
    def check_dependencies() -> Dict:
        """检查依赖"""
        results = {}
        
        # 检查Python版本
        results['python_version'] = sys.version
        
        # 检查关键依赖
        packages = ['numpy', 'torch', 'tqdm']
        
        for pkg in packages:
            try:
                __import__(pkg)
                results[pkg] = 'installed'
            except ImportError:
                results[pkg] = 'not installed'
        
        return results
    
    @staticmethod
    def print_project_info():
        """打印项目信息"""
        print("\n" + "=" * 60)
        print("四川麻将自博弈强化学习项目")
        print("=" * 60)
        
        # 检查依赖
        deps = ProjectManager.check_dependencies()
        
        print("\n依赖检查:")
        for name, status in deps.items():
            status_icon = "✓" if status == 'installed' else "✗"
            print(f"  {status_icon} {name}: {status}")
        
        print("\n项目结构:")
        print("  mjai/           - Python组件")
        print("  engine/         - Rust引擎")
        print("  configs/        - 配置文件")
        print("  scripts/        - 脚本")
        print("  tests/          - 测试")
        print("  output/         - 输出")
        
        print("\n可用脚本:")
        print("  python scripts/train.py          - 训练模型")
        print("  python scripts/eval.py           - 评估模型")
        print("  python scripts/replay.py         - 分析回放")
        print("  python tests/run_tests.py        - 运行测试")
        
        print("=" * 60 + "\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='辅助工具集')
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 项目信息
    subparsers.add_parser('info', help='显示项目信息')
    
    # 依赖检查
    subparsers.add_parser('deps', help='检查依赖')
    
    # 导出模型
    export_parser = subparsers.add_parser('export', help='导出模型')
    export_parser.add_argument('--model', required=True, help='模型路径')
    export_parser.add_argument('--format', choices=['onnx', 'torchscript', 'info'],
                               default='info', help='导出格式')
    export_parser.add_argument('--output', help='输出路径')
    
    # 日志分析
    log_parser = subparsers.add_parser('analyze', help='分析训练日志')
    log_parser.add_argument('--log', required=True, help='日志文件路径')
    
    args = parser.parse_args()
    
    if args.command == 'info':
        ProjectManager.print_project_info()
    
    elif args.command == 'deps':
        deps = ProjectManager.check_dependencies()
        print("依赖状态:")
        for name, status in deps.items():
            status_icon = "✓" if status == 'installed' else "✗"
            print(f"  {status_icon} {name}: {status}")
    
    elif args.command == 'export':
        if args.format == 'info':
            exporter = ModelExporter(args.model)
            info = exporter.export_architecture_info(args.output or 'model_info.json')
            if info:
                print(f"\n模型参数总数: {info['total_params']:,}")
                print(f"可训练参数: {info['trainable_params']:,}")
                print(f"网络层数: {len(info['layers'])}")
        elif args.format == 'onnx':
            exporter = ModelExporter(args.model)
            exporter.export_to_onnx(args.output or 'model.onnx')
        elif args.format == 'torchscript':
            exporter = ModelExporter(args.model)
            exporter.export_to_torchscript(args.output or 'model.pt')
    
    elif args.command == 'analyze':
        logger = TrainingLogger()
        logger.load_log(args.log)
        summary = logger.get_summary()
        
        print(f"\n日志摘要:")
        print(f"  条目数: {summary['num_entries']}")
        print(f"  迭代范围: {summary['first_iteration']} - {summary['last_iteration']}")
        print(f"  Policy Loss范围: {summary['policy_loss_range']}")
        print(f"  最终统计: {summary['final_stats']}")


if __name__ == "__main__":
    main()