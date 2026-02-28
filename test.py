from dataclasses import dataclass, field

# 危险的写法
@dataclass
class SharedProblem:
    bad_list: list = []        # 类定义时创建一次
    bad_dict: dict = {}        # 类定义时创建一次

# 安全的写法
@dataclass
class SafeSolution:
    good_list: list = field(default_factory=list)    # 每次实例化时调用
    good_dict: dict = field(default_factory=dict)    # 每次实例化时调用

# 验证问题
print("=== 问题演示 ===")
problem1 = SharedProblem()
problem2 = SharedProblem()

problem1.bad_list.append("item1")
problem1.bad_dict["key1"] = "value1"

print(f"problem1.list: {problem1.bad_list}")  # ['item1']
print(f"problem2.list: {problem2.bad_list}")  # ['item1'] !!! 共享了！
print(f"problem1.dict: {problem1.bad_dict}")  # {'key1': 'value1'}
print(f"problem2.dict: {problem2.bad_dict}")  # {'key1': 'value1'} !!! 共享了！

# 验证解决方案
print("\n=== 解决方案演示 ===")
safe1 = SafeSolution()
safe2 = SafeSolution()

safe1.good_list.append("item1")
safe1.good_dict["key1"] = "value1"

print(f"safe1.list: {safe1.good_list}")  # ['item1']
print(f"safe2.list: {safe2.good_list}")  # [] 独立！
print(f"safe1.dict: {safe1.good_dict}")  # {'key1': 'value1'}
print(f"safe2.dict: {safe2.good_dict}")  # {} 独立！
