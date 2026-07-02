import os
import sys

# 테스트를 어디서(pytest/python -m pytest, IDE 등) 실행하든 리포 루트가
# sys.path 에 있도록 보장(benchmark/optim 패키지 import 용).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
