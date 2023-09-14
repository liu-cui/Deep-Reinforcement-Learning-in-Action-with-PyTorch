# 添加自定义模块路径
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/agents/")
sys.path.append("/common/")

# 导入自定义模块
from agents.hello_agent import run_hello_agent


if __name__ == '__main__':
    run_hello_agent()
