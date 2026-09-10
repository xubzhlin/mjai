#!/usr/bin/env python3
"""
测试运行脚本
运行所有测试并生成报告
"""

import sys
import os
import subprocess
import time
import json
from datetime import datetime

def run_test_command(command, description):
    """运行测试命令"""
    print(f"\n{'='*60}")
    print(f"运行测试: {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    print(f"执行时间: {execution_time:.2f}秒")
    print(f"返回码: {result.returncode}")
    
    if result.stdout:
        print("输出:")
        print(result.stdout)
    
    if result.stderr:
        print("错误:")
        print(result.stderr)
    
    return {
        'command': command,
        'description': description,
        'returncode': result.returncode,
        'execution_time': execution_time,
        'stdout': result.stdout,
        'stderr': result.stderr
    }

def main():
    """主函数"""
    print("=" * 60)
    print("四川麻将自博弈强化学习项目 - 测试套件")
    print("=" * 60)
    
    # 测试结果
    test_results = []
    
    # 1. 运行单元测试
    print("\n1. 运行单元测试...")
    unit_test_results = []
    
    unit_tests = [
        ('python -m pytest tests/test_tile.py -v', "牌具模块测试"),
        ('python -m pytest tests/test_algorithms.py -v', "核心算法测试"),
        ('python -m pytest tests/test_features.py -v', "特征处理测试"),
        ('python -m pytest tests/test_models.py -v', "神经网络模型测试"),
        ('python -m pytest tests/test_config.py -v', "配置管理测试")
    ]
    
    for command, description in unit_tests:
        result = run_test_command(command, description)
        unit_test_results.append(result)
        test_results.append(result)
    
    # 2. 运行集成测试
    print("\n2. 运行集成测试...")
    integration_test_result = run_test_command(
        'python -m pytest tests/test_integration.py -v',
        "集成测试"
    )
    test_results.append(integration_test_result)
    
    # 3. 运行完整测试套件
    print("\n3. 运行完整测试套件...")
    full_test_result = run_test_command(
        'python tests/run_tests.py',
        "完整测试套件"
    )
    test_results.append(full_test_result)
    
    # 4. 运行性能测试
    print("\n4. 运行性能测试...")
    performance_test_result = run_test_command(
        'python -m pytest tests/test_integration.py::TestPerformance -v',
        "性能测试"
    )
    test_results.append(performance_test_result)
    
    # 5. 生成测试报告
    print("\n5. 生成测试报告...")
    
    # 计算统计信息
    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results if r['returncode'] == 0)
    failed_tests = total_tests - passed_tests
    success_rate = (passed_tests / total_tests) * 100
    total_execution_time = sum(r['execution_time'] for r in test_results)
    
    # 生成测试报告
    test_report = {
        'timestamp': datetime.now().isoformat(),
        'project': '四川麻将自博弈强化学习项目',
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
        'success_rate': success_rate,
        'total_execution_time': total_execution_time,
        'test_results': test_results,
        'summary': {
            'unit_tests': len([r for r in test_results if 'unit' in r['description'].lower()]),
            'integration_tests': len([r for r in test_results if 'integration' in r['description'].lower()]),
            'performance_tests': len([r for r in test_results if 'performance' in r['description'].lower()])
        }
    }
    
    # 保存测试报告
    report_path = os.path.join(os.path.dirname(__file__), '..', 'test_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(test_report, f, indent=2, ensure_ascii=False)
    
    print(f"测试报告已保存到: {report_path}")
    
    # 6. 显示测试结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {failed_tests}")
    print(f"成功率: {success_rate:.2f}%")
    print(f"总执行时间: {total_execution_time:.2f}秒")
    
    # 显示详细结果
    print("\n详细结果:")
    for i, result in enumerate(test_results, 1):
        status = "✓ 通过" if result['returncode'] == 0 else "✗ 失败"
        print(f"  {i}. {result['description']}: {status}")
        print(f"     执行时间: {result['execution_time']:.2f}秒")
    
    # 7. 给出建议
    print("\n" + "=" * 60)
    print("建议")
    print("=" * 60)
    
    if success_rate == 100:
        print("✓ 所有测试通过！项目状态良好。")
        print("建议:")
        print("  - 继续保持代码质量")
        print("  - 定期运行测试套件")
        print("  - 考虑添加更多边缘情况测试")
    elif success_rate >= 80:
        print("✓ 大部分测试通过，但有少量失败。")
        print("建议:")
        print("  - 检查失败的测试用例")
        print("  - 修复相关问题")
        print("  - 重新运行测试验证修复")
    else:
        print("✗ 测试失败率较高，需要关注。")
        print("建议:")
        print("  - 立即检查失败的测试用例")
        print("  - 修复关键问题")
        print("  - 重新运行测试直到通过率提升")
    
    # 8. 返回退出码
    if failed_tests > 0:
        print(f"\n⚠️  有 {failed_tests} 个测试失败，请检查相关代码。")
        sys.exit(1)
    else:
        print("\n🎉 所有测试通过！")
        sys.exit(0)

if __name__ == '__main__':
    main()