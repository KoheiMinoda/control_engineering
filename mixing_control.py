import numpy as np
import cvxpy as cp

# システムパラメータ
A = np.array([[0, 1],
              [-2, -3]])
B = np.array([[0],
              [1]])
C = np.array([[1, 0]])
D = np.array([[0]])

# 重み行列
Q = np.eye(2)
R = np.eye(1)
gamma = 1.0  # H_inf ノルム制約

# 変数の定義
X = cp.Variable((2, 2), symmetric=True)
Z = cp.Variable((1, 2))

# LQR制御のLMI
LQR_LMI = [
    X >> 0,
    cp.bmat([
        [A @ X + X @ A.T + B @ Z + Z.T @ B.T, X @ C.T + Z.T @ D.T],
        [C @ X + D @ Z, -gamma * np.eye(1)]
    ]) << 0
]

# 制約のリスト
constraints = LQR_LMI

# 目的関数
objective = cp.Minimize(cp.trace(Q @ X) + cp.trace(R @ (Z.T @ Z)))

# 最適化問題の定義
prob = cp.Problem(objective, constraints)

# 最適化の実行
prob.solve()

# 最適フィードバックゲインの計算
K = Z.value @ np.linalg.inv(X.value)
print("最適フィードバックゲインK:", K)