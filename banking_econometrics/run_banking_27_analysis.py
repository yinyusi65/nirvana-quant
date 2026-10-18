"""
BANKING ECONOMETRICS: 27 COMMERCIAL BANKS SYSTEMIC RISK RUNNER
==============================================================
Comprehensive execution pipeline applying the newly updated Nirvana Engine
to all 27 Vietnamese commercial banks (2017-2025).

Features:
1. Balanced panel connectedness (2021-2025)
2. Full historical synthesized panel (2017-2025)
3. Asset-weighting & Bootstrap apoptosis edge pruning (p < 0.05)
4. SVD Market Orthogonalization (separating market co-movement from idiosyncratic risk)
5. 4 Historical Macro Regimes (Pre-Covid, Covid Crisis, Bond/SCB Shock, Post-Crisis Recovery)
6. Complex Network Centrality (Hubs, Authorities, Bridges, Flight-to-Safety)
7. Cross-Sectional Econometric Phase 2 Integration with Stata Diagnostics

Author: yinyusi65
"""

import os
import sys
from typing import Dict, Any
import numpy as np
import pandas as pd

# Windows encoding safety
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nirvana import (
    NirvanaConnectedness,
    NirvanaTwoStageRegressor,
    stata_regress,
    parse_formula,
)
from nirvana.domains.banking import DEFAULT_ASSET_WEIGHTS, DEFAULT_EXCHANGE_BANDS, classify_sifi_status


def find_data_file(filename: str) -> str:
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "data", filename),
        os.path.join(os.path.dirname(__file__), "data", filename),
        os.path.join("data", filename),
        filename,
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"Cannot locate {filename}. Looked in: {candidates}")


def run_banking_systemic_risk_pipeline(output_dir: str = None) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 82)
    print("🏦 NIRVANA QUANT: BÁO CÁO MẠNG LƯỚI RỦI RO HỆ THỐNG 27 NGÂN HÀNG THƯƠNG MẠI")
    print("   Khung Phân Tích Chuỗi Thời Gian & Kinh Tế Lượng Mở Rộng Phase 2 (2017 - 2025)")
    print("   Tác giả / Author: yinyusi65")
    print("=" * 82)

    # 1. Load Data
    path_balanced = find_data_file("all27_panel_data_floor_2021_2025.csv")
    path_full = find_data_file("all27_panel_multiverse_synthesized_2017_2025.csv")

    df_balanced = pd.read_csv(path_balanced, index_col=0, parse_dates=True)
    df_full = pd.read_csv(path_full, index_col=0, parse_dates=True)

    print(f"\n[1/6] Nạp dữ liệu thành công:")
    print(f"  * Mẫu cân bằng (2021-2025): {df_balanced.shape[0]} phiên x {df_balanced.shape[1]} ngân hàng")
    print(f"  * Toàn kỳ Đa vũ trụ (2017-2025): {df_full.shape[0]} phiên x {df_full.shape[1]} ngân hàng (0 ô khuyết)")

    # 2. Run Engine on Balanced Sample
    engine = NirvanaConnectedness(kalman_q=1e-3, kalman_r=1e-2, apoptosis_threshold=0.015)
    rep_balanced = engine.fit(df_balanced)

    print(f"\n[2/6] Kết quả Mẫu Cân Bằng 27 Ngân Hàng (2021 - 2025):")
    print(f"  * Chỉ số TCI thô (Raw TCI):               {rep_balanced.total_connectedness_index:.2f}%")
    print(f"  * Chỉ số TCI chuẩn hóa (Normalized TCI):   {rep_balanced.normalized_tci:.2f}%")
    print(f"  * Top 3 Nguồn phát tán: {', '.join([f'{b} (+{rep_balanced.net_spillovers[b]:.1f}%)' for b in rep_balanced.top_transmitters[:3]])}")
    print(f"  * Top 3 Bên hấp thụ sốc: {', '.join([f'{b} ({rep_balanced.net_spillovers[b]:.1f}%)' for b in rep_balanced.top_receivers[:3]])}")

    # 3. Run Engine on Full Historical Period with Asset-Weighting & Bootstrap (p < 0.05)
    print(f"\n[3/6] Phân tích Toàn kỳ 2017-2025 (Trọng số Quy mô Tài sản & Bootstrap p < 0.05)...")
    rep_macro = engine.fit(df_full, asset_weights=DEFAULT_ASSET_WEIGHTS, n_boot=50)

    print(f"  * Chỉ số TCI thô Toàn kỳ (2017-2025):      {rep_macro.total_connectedness_index:.2f}%")
    print(f"  * Chỉ số TCI chuẩn hóa (Normalized TCI):   {rep_macro.normalized_tci:.2f}%")

    # 4. Multi-Factor SVD Market Orthogonalization
    print(f"\n[4/6] Bóc tách Nhân tố Thị trường Chung (Truncated SVD Orthogonalization)...")
    rep_ortho = engine.fit(df_full, asset_weights=DEFAULT_ASSET_WEIGHTS, orthogonalize_market=True, n_boot=50)

    print(f"  * Tỷ trọng phương sai VN-Index giải thích: {rep_ortho.market_variance_share:.2f}%")
    print(f"  * Chỉ số TCI Đặc thù (Idiosyncratic TCI):  {rep_ortho.total_connectedness_index:.2f}% (Chuẩn hóa: {rep_ortho.normalized_tci:.2f}%)")
    print(f"  -> Nhận định: Khoảng {100.0 - (rep_ortho.total_connectedness_index / rep_macro.total_connectedness_index * 100):.1f}% mức gắn kết bắt nguồn từ tâm lý thị trường chung.")

    # 5. Macro-Historical Regimes Analysis
    print(f"\n[5/6] Đánh giá 4 Thời kỳ Lịch sử Vĩ mô (Macro Regimes 2017 - 2025)...")
    regimes_reports = engine.fit_regimes(df_full, asset_weights=DEFAULT_ASSET_WEIGHTS)
    regime_records = []
    for reg_name, rep_reg in regimes_reports.items():
        t1 = rep_reg.top_transmitters[0]
        r1 = rep_reg.top_receivers[0]
        regime_records.append({
            "Thời kỳ Lịch sử": reg_name,
            "Raw TCI (%)": round(rep_reg.total_connectedness_index, 2),
            "Norm TCI (%)": round(rep_reg.normalized_tci, 2),
            "Top 1 Phát tán Rủi ro": f"{t1} (+{rep_reg.net_spillovers[t1]:.1f}%)",
            "Top 1 Hấp thụ Tổn thương": f"{r1} ({rep_reg.net_spillovers[r1]:.1f}%)",
        })
    df_regimes = pd.DataFrame(regime_records)
    print(df_regimes.to_string(index=False))

    # 6. Build Comprehensive Cross-Sectional Table for 27 Banks
    banks_list = list(df_full.columns)
    cross_records = []
    for b in banks_list:
        asset = DEFAULT_ASSET_WEIGHTS.get(b, 100.0)
        band = DEFAULT_EXCHANGE_BANDS.get(b, 0.07)
        net_spill = rep_macro.net_spillovers.get(b, 0.0)
        hub = rep_macro.eigenvector_hub.get(b, 0.0) if rep_macro.eigenvector_hub else 0.0
        auth = rep_macro.eigenvector_authority.get(b, 0.0) if rep_macro.eigenvector_authority else 0.0
        bc = rep_macro.betweenness_centrality.get(b, 0.0) if rep_macro.betweenness_centrality else 0.0
        fts = rep_macro.flight_to_safety_index.get(b, 0.0) if rep_macro.flight_to_safety_index else 0.0
        beta_mkt = rep_ortho.market_beta.get(b, 0.15) if rep_ortho.market_beta else 0.15

        cross_records.append({
            "Bank": b,
            "Exchange_Band": band,
            "Total_Assets_Trillion": asset,
            "Log_Assets": np.log(asset),
            "Net_Spillover": net_spill,
            "Hub_Centrality": hub,
            "Authority_Centrality": auth,
            "Betweenness_Bridge": bc,
            "Flight_To_Safety": fts,
            "Market_Beta": beta_mkt,
        })

    df_cross = pd.DataFrame(cross_records).sort_values(by="Net_Spillover", ascending=False).reset_index(drop=True)

    # 7. PHASE 2 ECONOMETRIC DISCOVERY ON BANKING CROSS-SECTION
    print(f"\n[6/6] Khảo sát Kinh Tế Lượng Phase 2 & Chẩn Đoán Stata (Cắt Lớp Ngân Hàng)...")
    print("  * Mục tiêu: Giải thích mức phát tán rủi ro thuần (Net_Spillover) bằng các biến cơ sở:")
    print("    Net_Spillover ~ Log_Assets + Exchange_Band + Market_Beta")

    X_bank, y_bank, f_names = parse_formula(
        "Net_Spillover ~ Log_Assets + Exchange_Band + Market_Beta", df_cross
    )

    # Run Stata Base Regression
    df_stata_base = df_cross[["Log_Assets", "Exchange_Band", "Market_Beta", "Net_Spillover"]].copy()
    base_stata_res = stata_regress(df_stata_base, "Net_Spillover", ["Log_Assets", "Exchange_Band", "Market_Beta"])

    print("\n--- BẢNG HỒI QUY CƠ SỞ CHUẨN STATA (PHASE 1 OLS BASELINE) ---")
    print(base_stata_res.summary())

    # Fit Nirvana TwoStageRegressor (Phase 2 Apoptosis Tree)
    two_stage = NirvanaTwoStageRegressor(base_estimator="ols", max_depth=5, apoptosis_start_depth=2)
    two_stage.fit(X_bank, y_bank)

    # Prove Phase 2 Discovered Variable
    proof_rep = two_stage.prove_phase2_multiverse(
        X_bank, y_bank, n_universes=100, seed=42
    )

    print("\n--- BÁO CÁO CHỨNG MINH KINH TẾ LƯỢNG BIẾN MỚI TÌM ĐƯỢC (PHASE 2 MULTIVERSE) ---")
    print(proof_rep.summary())

    # Rank top variables
    ranked_rep = two_stage.rank_discovered_variables(X_bank, y_bank, top_k=3, n_universes=100, seed=42)
    print("\n--- BẢNG XẾP HẠNG TOP BIẾN SỐ TIỀM ẨN BẤT BIẾN (TOP 3 CHAMPIONS) ---")
    print(ranked_rep.summary())

    # Save Outputs
    csv_systemic_path = os.path.join(output_dir, "banking_systemic_risk_table_2017_2025.csv")
    csv_regimes_path = os.path.join(output_dir, "banking_macro_regimes_summary.csv")
    txt_report_path = os.path.join(output_dir, "banking_execution_report.txt")

    df_cross.to_csv(csv_systemic_path, index=False, encoding="utf-8-sig")
    df_regimes.to_csv(csv_regimes_path, index=False, encoding="utf-8-sig")

    with open(txt_report_path, "w", encoding="utf-8") as f:
        f.write("NIRVANA BANKING SYSTEMIC RISK & PHASE 2 REPORT\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Raw TCI (Balanced 2021-2025): {rep_balanced.total_connectedness_index:.2f}%\n")
        f.write(f"Raw TCI (Full 2017-2025):     {rep_macro.total_connectedness_index:.2f}%\n")
        f.write(f"Idiosyncratic TCI:             {rep_ortho.total_connectedness_index:.2f}%\n")
        f.write(f"Market Factor Variance Share:  {rep_ortho.market_variance_share:.2f}%\n\n")
        f.write("--- REGIMES ---\n")
        f.write(df_regimes.to_string(index=False) + "\n\n")
        f.write("--- STATA BASE REGRESSION ---\n")
        f.write(base_stata_res.summary() + "\n\n")
        f.write("--- PHASE 2 MULTIVERSE PROOF ---\n")
        f.write(proof_rep.summary() + "\n\n")
        f.write("--- TOP DISCOVERED VARIABLES ---\n")
        f.write(ranked_rep.summary() + "\n")

    print("\n" + "=" * 82)
    print(f"✓ ĐÃ LƯU KẾT QUẢ THÀNH CÔNG VÀO THƯ MỤC: {output_dir}")
    print(f"  * [CSV] {csv_systemic_path}")
    print(f"  * [CSV] {csv_regimes_path}")
    print(f"  * [TXT] {txt_report_path}")
    print("=" * 82)

    return {
        "rep_balanced": rep_balanced,
        "rep_macro": rep_macro,
        "rep_ortho": rep_ortho,
        "regimes_reports": regimes_reports,
        "df_cross": df_cross,
        "df_regimes": df_regimes,
        "base_stata_res": base_stata_res,
        "proof_rep": proof_rep,
        "ranked_rep": ranked_rep,
    }


if __name__ == "__main__":
    run_banking_systemic_risk_pipeline()
