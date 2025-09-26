import numpy as np
from scipy.optimize import minimize

def optimize_pruning_ratio(scores, rho_bar=0.2, rho_min=0.1, rho_max=0.3):
    """
    スコアに基づいて、各要素にclip制約付きかつ平均がrho_barになるように
    Pruning率を最適化する。
    
    Parameters:
        scores (np.ndarray): スコア列（例: 残差ノルムなど）
        rho_bar (float): 目標とする平均Pruning率
        rho_min (float): Pruning率の下限
        rho_max (float): Pruning率の上限

    Returns:
        rho_opt (np.ndarray): 最適化されたPruning率（長さn）
    """

    scores = np.array(scores)
    n = len(scores)

    # min-max正規化
    s = (scores - scores.min()) / (scores.max() - scores.min())

    # 初期値（clip済みのスコア）
    x0 = np.clip(s.copy(), rho_min, rho_max)

    # 目的関数：スコアとの距離を最小に
    def objective(rho):
        return np.sum((rho - s)**2)

    # 平均制約
    def constraint_mean(rho):
        return np.mean(rho) - rho_bar

    constraints = {'type': 'eq', 'fun': constraint_mean}
    bounds = [(rho_min, rho_max)] * n

    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return result.x


def allocate_keep_counts(q_o, q_d, H=4096, M=11008, rho=0.5):
    """
    γなし・平均正規化なし。
    まずモジュール間に予算を比例配分し、次に各層へ比例配分。
    端数調整は簡単な貪欲で合わせる。
    """
    import math

    L = len(q_o)
    assert len(q_d) == L
    C_o, C_d = 4, 3  # 連動コスト（o:4H, mlp:3H のHで割った簡略形）

    B_tot = (1.0 - rho) * (L * (H*C_o + M*C_d))

    # モジュール重み（ハイパラ無し）
    W_o = (sum(q_o) * H) / C_o
    W_d = (sum(q_d) * M) / C_d
    if W_o + W_d == 0:
        # どちらもゼロなら均等
        W_o = W_d = 1.0

    B_o = B_tot * (W_o / (W_o + W_d))
    B_d = B_tot * (W_d / (W_o + W_d))

    # 層内配分（比例配分→丸め→cap）
    so, sd = sum(q_o), sum(q_d)
    so = so if so > 0 else 1.0
    sd = sd if sd > 0 else 1.0

    # まず丸め前ターゲット本数
    target_o = (B_o / C_o) * (1.0 / so)
    target_d = (B_d / C_d) * (1.0 / sd)

    K_o = [min(H, int(round(target_o * q_o[l]))) for l in range(L)]
    K_d = [min(M, int(round(target_d * q_d[l]))) for l in range(L)]

    # 予算に合わせて微調整（過不足分を±1で調整）
    def total_cost(Ko, Kd):
        return sum(Ko) * C_o + sum(Kd) * C_d

    need = int(round(B_tot))
    cur = total_cost(K_o, K_d)

    # 調整用のスコア（価値/コストが小さい順に削る、大きい順に追加）
    attn_scores = [q_o[l] / C_o for l in range(L)]
    mlp_scores  = [q_d[l] / C_d for l in range(L)]

    while cur != need:
        if cur > need:
            # 予算オーバー → 一番“価値/コスト”の低いユニットを1本減らす
            cand = []
            for l in range(L):
                if K_o[l] > 0:
                    cand.append(("o", l, attn_scores[l]))
                if K_d[l] > 0:
                    cand.append(("d", l, mlp_scores[l]))
            if not cand:
                break
            kind, l, _ = min(cand, key=lambda x: x[2])
            if kind == "o":
                K_o[l] -= 1; cur -= C_o
            else:
                K_d[l] -= 1; cur -= C_d
        else:
            # 予算不足 → 一番“価値/コスト”の高いユニットを1本増やす
            cand = []
            for l in range(L):
                if K_o[l] < H: cand.append(("o", l, attn_scores[l]))
                if K_d[l] < M: cand.append(("d", l, mlp_scores[l]))
            if not cand:
                break
            kind, l, _ = max(cand, key=lambda x: x[2])
            if kind == "o":
                K_o[l] += 1; cur += C_o
            else:
                K_d[l] += 1; cur += C_d

    pr_o = [1.0 - Ko / float(H) for Ko in K_o]
    pr_d = [1.0 - Kd / float(M) for Kd in K_d]
    return K_o, K_d, pr_o, pr_d




# ======= 使用例 =======
if __name__ == "__main__":
    # 任意のスコア列（ここでは o を例に）
    o2 = np.array([
        164.16, 53.89, 171.70, 66.02, 64.09, 100.59, 63.99, 226.68,
        397.49, 784.23, 382.78, 422.87, 722.85, 1058.71, 587.60, 396.77,
        594.68, 216.89, 151.81, 105.85, 473.02, 364.50, 1086.93, 903.07,
        282.26, 1817.61, 622.35, 458.26, 1248.40, 826.05, 3037.79, 2261.98
    ])
    d2 = np.array([
        5.50, 4.76, 8.28, 12.51, 20.02, 31.46, 49.65, 64.40,
        84.13, 84.91, 104.69, 146.36, 226.18, 257.77, 227.66, 299.63,
        410.53, 463.81, 450.16, 494.00, 520.48, 576.49, 564.12, 614.65,
        650.97, 692.60, 723.30, 739.40, 833.89, 950.21, 2876.08, 4308.87
    ])
    o5 = np.array([
        1080.8387, 414.9301, 271.3277, 699.2816, 191.9243, 467.6941, 
        371.1183, 679.4035, 1686.9888, 1522.0968, 1099.7529, 1325.3856, 
        2876.5466, 3214.8438, 2675.4995, 2131.2131, 3484.5281, 1337.7899, 
        1072.0746, 1051.3743, 4109.7339, 3295.6113, 18745.8984, 4369.6865, 
        2049.9624, 4960.0747, 4505.3179, 6535.0918, 12794.3037, 4639.9966, 
        18005.0566, 83804.6406
    ])
    d5 = np.array([
        22.2312, 17.4953, 26.0230, 42.5517, 74.9224, 121.8142, 
        195.2499, 245.6230, 338.1105, 365.3853, 442.0906, 630.6843, 
        986.8322, 1169.2766, 1106.8429, 1395.9443, 1769.3878, 1952.0522, 
        1755.0018, 1724.3341, 1805.8481, 1854.5426, 1855.3607, 1965.0187, 
        2028.4087, 2149.9419, 2159.9778, 2275.5098, 2570.1582, 3121.8750, 
        67301.7031, 14351.5391
    ])


    rho_o = optimize_pruning_ratio(o2, rho_bar=0.2, rho_min=0.1, rho_max=0.3)
    rho_d = optimize_pruning_ratio(d2, rho_bar=0.2, rho_min=0.1, rho_max=0.3)
    print("Pruning ratios (rho_o, rho_d):")
    print(list(np.round(rho_o, 3)))
    print(list(np.round(rho_d, 3)))
    print("Mean pruning rate o:", np.round(np.mean(rho_o), 4))
    print("Mean pruning rate d:", np.round(np.mean(rho_d), 4))

    rho_o = optimize_pruning_ratio(o5, rho_bar=0.5, rho_min=0.3, rho_max=0.7)
    rho_d = optimize_pruning_ratio(d5, rho_bar=0.5, rho_min=0.3, rho_max=0.7)
    print("Pruning ratios (rho_o, rho_d):")
    print(list(np.round(rho_o, 3)))
    print(list(np.round(rho_d, 3)))
    print("Mean pruning rate o:", np.round(np.mean(rho_o), 4))
    print("Mean pruning rate d:", np.round(np.mean(rho_d), 4))
