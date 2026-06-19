# scripts/verify.py
import numpy as np
import time
from maraboupy import Marabou
from torchvision import datasets, transforms

# 1. 모델 로드
print("모델 로드 중...")
network = Marabou.read_onnx("models/mnist_fc.onnx")

# 2. 테스트 이미지 
datasets.MNIST.mirrors = ["https://storage.googleapis.com/cvdf-datasets/mnist/"]
test_data = datasets.MNIST('./data', train=False, download=False,
                           transform=transforms.ToTensor())
x, true_label = test_data[0]
x_flat = x.numpy().flatten()
print(f"테스트 이미지 레이블: {true_label}")

# 3. 검증 (각 클래스 j vs true_label)
epsilon = 0.01
print(f"ε={epsilon} 로 robustness 검증 시작...\n")

all_unsat = True
for j in range(10):
    if j == true_label:
        continue

    net = Marabou.read_onnx("models/mnist_fc.onnx")
    inp = net.inputVars[0].flatten()
    out = net.outputVars[0].flatten()

    # 입력 제약: ℓ∞-ball
    for i, var in enumerate(inp):
        lb = float(np.clip(x_flat[i] - epsilon, 0.0, 1.0))
        ub = float(np.clip(x_flat[i] + epsilon, 0.0, 1.0))
        net.setLowerBound(var, lb)
        net.setUpperBound(var, ub)

    # 출력 제약: output[j] >= output[true_label]
    net.addInequality([out[j], out[true_label]], [1, -1], 0)

    options = Marabou.createOptions(verbosity=0, timeoutInSeconds=120)
    t0 = time.time()
    result = net.solve(options=options)
    elapsed = time.time() - t0

    status = result[0]
    print(f"class {true_label} vs {j}: {status.upper()} ({elapsed:.1f}s)")

    if status == 'sat':
        all_unsat = False

# 4. 최종 결과
print()
if all_unsat:
    print(f"✅ UNSAT: ε={epsilon} 범위 내 모든 입력이 '{true_label}'로 분류됨 (검증 성공)")
else:
    print(f"❌ SAT: 반례 발견 - adversarial example 존재")