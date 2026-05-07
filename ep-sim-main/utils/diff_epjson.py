#!/usr/bin/env python3
"""
比较两个 epJSON 文件的区别，不考虑顺序
"""

import json
import sys
from pathlib import Path


def load_epjson(filepath):
    """加载 epJSON 文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def deep_diff(obj1, obj2, path=""):
    """递归比较两个对象的差异"""
    differences = []

    if type(obj1) != type(obj2):
        differences.append(f"{path}: 类型不同 - {type(obj1).__name__} vs {type(obj2).__name__}")
        return differences

    if isinstance(obj1, dict):
        # 比较字典的键
        keys1 = set(obj1.keys())
        keys2 = set(obj2.keys())

        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        common = keys1 & keys2

        for key in only_in_1:
            differences.append(f"{path}.{key}: 仅在文件1中存在")

        for key in only_in_2:
            differences.append(f"{path}.{key}: 仅在文件2中存在")

        # 递归比较共同的键
        for key in sorted(common):
            new_path = f"{path}.{key}" if path else key
            differences.extend(deep_diff(obj1[key], obj2[key], new_path))

    elif isinstance(obj1, list):
        if len(obj1) != len(obj2):
            differences.append(f"{path}: 列表长度不同 - {len(obj1)} vs {len(obj2)}")
        else:
            for i, (item1, item2) in enumerate(zip(obj1, obj2)):
                differences.extend(deep_diff(item1, item2, f"{path}[{i}]"))

    else:
        # 基本类型比较
        if obj1 != obj2:
            differences.append(f"{path}: {obj1} -> {obj2}")

    return differences


def compare_epjson_summary(file1, file2):
    """比较两个 epJSON 文件并显示摘要"""
    print(f"比较文件:")
    print(f"  文件1: {file1}")
    print(f"  文件2: {file2}")
    print("=" * 80)

    epjson1 = load_epjson(file1)
    epjson2 = load_epjson(file2)

    # 获取顶层对象类型
    keys1 = set(epjson1.keys())
    keys2 = set(epjson2.keys())

    print("\n📋 顶层对象类型比较:")
    print(f"  文件1有 {len(keys1)} 个对象类型")
    print(f"  文件2有 {len(keys2)} 个对象类型")

    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    common = keys1 & keys2

    if only_in_1:
        print(f"\n  ⚠️  仅在文件1中的对象类型 ({len(only_in_1)}):")
        for key in sorted(only_in_1):
            print(f"    - {key}")

    if only_in_2:
        print(f"\n  ⚠️  仅在文件2中的对象类型 ({len(only_in_2)}):")
        for key in sorted(only_in_2):
            print(f"    - {key}")

    print(f"\n  ✓ 共同的对象类型: {len(common)}")

    # 比较共同对象类型中的实例数量
    print("\n📊 对象实例数量比较:")
    diff_count = 0
    for obj_type in sorted(common):
        if isinstance(epjson1[obj_type], dict) and isinstance(epjson2[obj_type], dict):
            count1 = len(epjson1[obj_type])
            count2 = len(epjson2[obj_type])
            if count1 != count2:
                print(f"  {obj_type}: {count1} -> {count2} ({count2 - count1:+d})")
                diff_count += 1

    if diff_count == 0:
        print("  ✓ 所有共同对象类型的实例数量相同")

    # 详细差异分析
    print("\n🔍 详细差异分析:")
    differences = deep_diff(epjson1, epjson2)

    if not differences:
        print("  ✓ 两个文件完全相同！")
    else:
        print(f"  发现 {len(differences)} 处差异")

        # 按对象类型分组差异
        diff_by_type = {}
        for diff in differences:
            obj_type = diff.split('.')[0] if '.' in diff else diff.split(':')[0]
            if obj_type not in diff_by_type:
                diff_by_type[obj_type] = []
            diff_by_type[obj_type].append(diff)

        print(f"\n  差异涉及 {len(diff_by_type)} 个对象类型:")
        for obj_type in sorted(diff_by_type.keys()):
            diffs = diff_by_type[obj_type]
            print(f"\n  📌 {obj_type} ({len(diffs)} 处差异):")
            # 只显示前5个差异
            for diff in diffs[:5]:
                print(f"    • {diff}")
            if len(diffs) > 5:
                print(f"    ... 还有 {len(diffs) - 5} 处差异")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python diff_epjson.py <文件1.epJSON> <文件2.epJSON>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    if not Path(file1).exists():
        print(f"错误: 文件不存在 - {file1}")
        sys.exit(1)

    if not Path(file2).exists():
        print(f"错误: 文件不存在 - {file2}")
        sys.exit(1)

    compare_epjson_summary(file1, file2)

