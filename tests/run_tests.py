#!/usr/bin/env python3
"""
测试运行脚本
运行所有测试并生成报告
"""

import unittest
import sys
import os
import time
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("四川麻将自博弈强化学习项目 - 测试套件")
    print("=" * 60)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试用例
    test_modules = [
        'test_tile',
        'test_algorithms', 
        'test_features',
        'test_models',
        'test_integration'
    ]
    
    for module in test_modules:
        try:
            tests = loader.loadTestsFromName(f'{module}')
            suite.addTests(tests)
            print(f"✓ 已添加 {module} 测试模块")
        except Exception as e:
            print(f"✗ 无法加载 {module} 测试模块: {e}")
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # 生成测试报告
    test_report = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped),
        'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100,
        'execution_time': end_time - start_time,
        'failures_details': result.failures,
        'errors_details': result.errors
    }
    
    # 保存测试报告
    report_path = os.path.join(os.path.dirname(__file__), '..', 'test_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(test_report, f, indent=2, ensure_ascii=False)
    
    # 打印测试结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    print(f"总测试数: {test_report['total_tests']}")
    print(f"失败数: {test_report['failures']}")
    print(f"错误数: {test_report['errors']}")
    print(f"跳过数: {test_report['skipped']}")
    print(f"成功率: {test_report['success_rate']:.2f}%")
    print(f"执行时间: {test_report['execution_time']:.2f}秒")
    
    # 显示失败和错误详情
    if test_report['failures'] > 0:
        print("\n失败的测试:")
        for failure in result.failures:
            print(f"  - {failure[0]}: {failure[1]}")
    
    if test_report['errors'] > 0:
        print("\n错误的测试:")
        for error in result.errors:
            print(f"  - {error[0]}: {error[1]}")
    
    # 返回测试结果
    return test_report

def run_specific_test(test_name):
    """运行特定测试"""
    print(f"运行特定测试: {test_name}")
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    try:
        tests = loader.loadTestsFromName(test_name)
        suite.addTests(tests)
        
        # 运行测试
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result
    except Exception as e:
        print(f"无法运行测试 {test_name}: {e}")
        return None

def main():
    """主函数"""
    if len(sys.argv) > 1:
        # 运行特定测试
        test_name = sys.argv[1]
        result = run_specific_test(test_name)
        if result:
            print(f"测试 {test_name} 完成")
        else:
            print(f"测试 {test_name} 失败")
    else:
        # 运行所有测试
        report = run_tests()
        
        # 根据测试结果返回退出码
        if report['failures'] > 0 or report['errors'] > 0:
            sys.exit(1)
        else:
            print("\n✓ 所有测试通过！")
            sys.exit(0)

if __name__ == '__main__':
    main()