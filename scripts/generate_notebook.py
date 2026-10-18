import json

def generate_clean_colab_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🏦 NIRVANA QUANT: PHÂN TÍCH RỦI RO HỆ THỐNG MẠNG LƯỚI TOÀN BỘ 27 NGÂN HÀNG VIỆT NAM (2017 - 2025)\n",
                "### Tác giả: yinyusi65\n",
                "---\n",
                "Nghiên cứu cấu trúc mạng lưới lan truyền rủi ro biến động cho **toàn bộ 27 ngân hàng thương mại** Việt Nam (HOSE, HNX, UPCoM):\n",
                "`['ABB', 'ACB', 'BAB', 'BID', 'BVB', 'CTG', 'EIB', 'HDB', 'KLB', 'LPB', 'MBB', 'MSB', 'NAB', 'NVB', 'OCB', 'PGB', 'SGB', 'SHB', 'SSB', 'STB', 'TCB', 'TPB', 'VAB', 'VBB', 'VCB', 'VIB', 'VPB']`\n",
                "\n",
                "Ứng dụng khung định lượng tối tân giải quyết triệt để 7 hạt sạn học thuật:\n",
                "1. **Chuẩn hóa Biên độ Sàn (Band Normalization)**: Triệt tiêu thiên lệch phương sai nhân tạo 4.6x giữa UPCoM (15%), HNX (10%) và HOSE (7%).\n",
                "2. **Chu kỳ Thanh toán T+2.5**: Nhân tích chập đa kỳ trễ (Lags 1, 2, 3) với trọng số suy giảm [0.5, 0.3, 0.2].\n",
                "3. **Độ Co Giãn Quy Mô Phi Tuyến (Sub-linear Elasticity)**: Trọng số căn bậc hai tài sản bảo toàn cấu trúc mạng lưới thực nghiệm.\n",
                "4. **Bóc Tách Thị Trường Chung (Market Orthogonalization)**: Tách nhân tố VN-Index / First Principal Component để làm lộ diện mạng lưới liên ngân hàng đặc thù.\n",
                "5. **Kiểm Định Bootstrap Ý Nghĩa Thống Kê (p < 0.05)**: Tự động apoptosis cắt tỉa liên kết nhiễu vô căn cứ.\n",
                "6. **Lý Thuyết Đồ Thị Phức Hợp (Graph Centrality)**: Phân tách rõ ràng Hub Centrality, Authority Centrality, Betweenness Bridge Banks và Flight-to-Safety Index.\n",
                "7. **Đa Vũ Trụ Tái Tạo Tiền Niêm Yết (Multiverse Reconstructor)**: Bổ khuyết chuỗi dữ liệu với nhiễu hiệu chuẩn và chuyển tiếp Hermite trơn mượt."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Cài đặt và Nạp Thư viện Nirvana Quant"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "from nirvana import (\n",
                "    NirvanaConnectedness, \n",
                "    MultiverseReconstructor, \n",
                "    MultiplexNetworkEngine,\n",
                "    compute_parkinson,\n",
                "    compute_yang_zhang\n",
                ")\n",
                "print(\"Nirvana Quantitative Engine loaded successfully!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Phần 1: Phân tích Mẫu Cân Bằng Chuẩn 27 Ngân Hàng (2021 - 2025)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "try:\n",
                "    df_27_balanced = pd.read_csv('all27_panel_data_floor_2021_2025.csv', index_col=0, parse_dates=True)\n",
                "    print(f\"[OK] Đã tải 27 ngân hàng cân bằng (2021-2025): {df_27_balanced.shape[0]} phiên x {df_27_balanced.shape[1]} banks\")\n",
                "except Exception as e:\n",
                "    print(\"Vui lòng upload file all27_panel_data_floor_2021_2025.csv vào Colab!\")\n",
                "\n",
                "engine = NirvanaConnectedness(kalman_q=1e-3, kalman_r=1e-2, apoptosis_threshold=0.015)\n",
                "report_27 = engine.fit(df_27_balanced)\n",
                "\n",
                "print(f\"=== TỔNG CHỈ SỐ KẾT NỐI HỆ THỐNG (Raw TCI): {report_27.total_connectedness_index:.2f}% ===\")\n",
                "print(f\"=== CHỈ SỐ CHUẨN HÓA KÍCH THƯỚC (Normalized TCI): {report_27.normalized_tci:.2f}%\\n\")\n",
                "print(\"Top 5 Ngân hàng phát tán rủi ro mạnh nhất (Systemic Transmitters, NET > 0):\")\n",
                "for b in report_27.top_transmitters[:5]:\n",
                "    print(f\"  [+] {b}: +{report_27.net_spillovers[b]:.2f}%\")\n",
                "\n",
                "print(\"\\nTop 5 Ngân hàng hấp thụ rủi ro / dễ bị tổn thương nhất (Systemic Receivers, NET < 0):\")\n",
                "for b in report_27.top_receivers[:5]:\n",
                "    print(f\"  [-] {b}: {report_27.net_spillovers[b]:.2f}%\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Hiển thị Heatmap Ma trận Rủi ro 27 Ngân Hàng (2021 - 2025)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "banks = [c for c in df_27_balanced.columns]\n",
                "matrix_core = report_27.spillover_table.loc[banks, banks].values\n",
                "\n",
                "plt.figure(figsize=(14, 11), dpi=150)\n",
                "plt.imshow(matrix_core, cmap='viridis', interpolation='nearest')\n",
                "plt.title('Ma Trận Lan Truyền Rủi Ro Hệ Thống 27 Ngân Hàng Việt Nam (Nirvana Engine 2021 - 2025)', fontsize=14, fontweight='bold')\n",
                "plt.colorbar(label='Tỷ trọng Lan truyền (%)')\n",
                "plt.xticks(range(len(banks)), banks, rotation=90, fontsize=9)\n",
                "plt.yticks(range(len(banks)), banks, fontsize=9)\n",
                "plt.xlabel('Ngân hàng Nguồn phát (Source / Transmitter - Col j)', fontsize=11)\n",
                "plt.ylabel('Ngân hàng Tiếp nhận (Target / Receiver - Row i)', fontsize=11)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Phần 2: Nấu Dữ Liệu Đa Vũ Trụ Tái Cấu Trúc Toàn Kỳ (2017 - 2025)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "try:\n",
                "    raw_full = pd.read_csv('all27_panel_full_2017_2025.csv', index_col=0, parse_dates=True)\n",
                "    reconstructor = MultiverseReconstructor(n_universes=3, boundary_window=30)\n",
                "    syn_full = reconstructor.reconstruct(raw_full)\n",
                "    print(f\"[OK] Đã hoàn tất tái cấu trúc Đa vũ trụ cho 27 ngân hàng (2017-2025): {syn_full.shape[0]} phiên, 0 ô khuyết thiếu!\")\n",
                "    \n",
                "    rep_full = engine.fit(syn_full)\n",
                "    print(f\"\\n=== TỔNG CHỈ SỐ KẾT NỐI TOÀN KỲ (2017 - 2025 TCI): {rep_full.total_connectedness_index:.2f}% | Norm TCI: {rep_full.normalized_tci:.2f}% ===\")\n",
                "except Exception as e:\n",
                "    print(\"Lỗi hoặc chưa upload all27_panel_full_2017_2025.csv:\", e)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Chuỗi Thời Gian Biến Động Lan Truyền Rủi Ro Toàn Kỳ (Rolling Dynamic TCI 2017 - 2025)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "rolling_tci = engine.rolling_tci(syn_full, window=200, step=20)\n",
                "\n",
                "plt.figure(figsize=(13, 5), dpi=150)\n",
                "plt.plot(pd.to_datetime(rolling_tci.index), rolling_tci.values, color='#0284c7', lw=2, label='Nirvana Multiverse Dynamic TCI (%)')\n",
                "plt.title('Chuỗi Thời Gian Chỉ Số Lan Truyền Rủi Ro Hệ Thống 27 Ngân Hàng Việt Nam (2017 - 2025)', fontsize=13, fontweight='bold')\n",
                "plt.ylabel('Total Connectedness Index (%)')\n",
                "plt.grid(True, linestyle='--', alpha=0.5)\n",
                "plt.legend()\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Mô Hình Trọng Số Quy Mô Tài Sản (Asset-Weighted Macro Footprint) & Bootstrap (p < 0.05)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "ASSETS_TRILLION = {\n",
                "    'BID': 2300, 'VCB': 1850, 'CTG': 2000, 'MBB': 950, 'TCB': 850, 'VPB': 820,\n",
                "    'ACB': 720, 'SHB': 630, 'HDB': 600, 'STB': 670, 'VIB': 410, 'LPB': 380,\n",
                "    'TPB': 350, 'MSB': 260, 'OCB': 240, 'SSB': 270, 'EIB': 200, 'NAB': 210,\n",
                "    'BAB': 150, 'NVB': 95, 'ABB': 160, 'VBB': 140, 'BVB': 88, 'KLB': 87,\n",
                "    'PGB': 56, 'VAB': 110, 'SGB': 32\n",
                "}\n",
                "\n",
                "rep_macro = engine.fit(syn_full, asset_weights=ASSETS_TRILLION, n_boot=50)\n",
                "\n",
                "print(f\"=== KẾT QUẢ KINH TẾ LƯỢNG VĨ MÔ (ASSET-WEIGHTED & BOOTSTRAP p < 0.05) ===\")\n",
                "print(f\"Raw TCI: {rep_macro.total_connectedness_index:.2f}% | Dimension-Normalized TCI: {rep_macro.normalized_tci:.2f}%\")\n",
                "\n",
                "print(\"\\nTop 10 Nguồn phát tán rủi ro hệ thống vĩ mô hàng đầu (NET > 0):\")\n",
                "for b in rep_macro.top_transmitters[:10]:\n",
                "    print(f\"  [+] {b:5s}: NET = {rep_macro.net_spillovers[b]:+7.2f}% | Tài sản: {ASSETS_TRILLION.get(b):,d} nghìn tỷ VND\")\n",
                "\n",
                "print(\"\\nTop 5 Ngân hàng hấp thụ rủi ro / tổn thương nhất (NET < 0):\")\n",
                "for b in rep_macro.top_receivers[:5]:\n",
                "    print(f\"  [-] {b:5s}: NET = {rep_macro.net_spillovers[b]:+7.2f}% | Tài sản: {ASSETS_TRILLION.get(b):,d} nghìn tỷ VND\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Bóc Tách Tác Động Thị Trường Chung (Market Orthogonalization)\n",
                "Phân lập cú sốc biến động toàn thị trường (VN-Index / First Principal Component) ra khỏi chuỗi đổi mới để đo lường mạng lưới lan truyền đặc thù liên ngân hàng (Idiosyncratic Interbank Network)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "rep_ortho = engine.fit(syn_full, asset_weights=ASSETS_TRILLION, orthogonalize_market=True, n_boot=50)\n",
                "\n",
                "print(\"=== MẠNG LƯỚI LAN TRUYỀN ĐẶC THÙ SAU KHI BÓC TÁCH VN-INDEX ===\")\n",
                "print(f\"Tỷ trọng phương sai Nhân tố Thị trường chung giải thích: {rep_ortho.market_variance_share:.2f}%\")\n",
                "print(f\"Chỉ số TCI Đặc thù (Idiosyncratic TCI): {rep_ortho.total_connectedness_index:.2f}% (Norm: {rep_ortho.normalized_tci:.2f}%)\\n\")\n",
                "\n",
                "print(\"Top 5 Ngân hàng Phát tán rủi ro thanh khoản đặc thù cao nhất:\")\n",
                "for b in rep_ortho.top_transmitters[:5]:\n",
                "    print(f\"  [+] {b:5s}: NET = {rep_ortho.net_spillovers[b]:+7.2f}% (Beta thị trường: {rep_ortho.market_beta.get(b, 0.0):.2f})\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Phân tích Theo 4 Thời kỳ Biến động Vĩ mô (Macro Regimes: 2017 - 2025)\n",
                "Nirvana Engine tự động phân rã 2,227 phiên giao dịch thành 4 thời kỳ lịch sử then chốt:\n",
                "1. **Pre-Covid (2017 - 2019)**: Giai đoạn tăng trưởng tín dụng bình thường.\n",
                "2. **Covid-19 Crisis (2020 - 2021)**: Cú sốc đại dịch toàn cầu & hạ lãi suất kỷ lục.\n",
                "3. **Bond & SCB Shock (2022 - 2023)**: Khủng hoảng thanh khoản trái phiếu doanh nghiệp & sự kiện SCB / Vạn Thịnh Phát.\n",
                "4. **Post-Crisis Recovery (2024 - 2025)**: Phục hồi sau khủng hoảng, phân hóa chất lượng tài sản."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "regimes_reports = engine.fit_regimes(syn_full, asset_weights=ASSETS_TRILLION)\n",
                "\n",
                "regime_summary = []\n",
                "for name, rep in regimes_reports.items():\n",
                "    t1 = rep.top_transmitters[0]\n",
                "    t2 = rep.top_transmitters[1]\n",
                "    t3 = rep.top_transmitters[2]\n",
                "    r1 = rep.top_receivers[0]\n",
                "    regime_summary.append({\n",
                "        \"Thời kỳ\": name,\n",
                "        \"Raw TCI (%)\": round(rep.total_connectedness_index, 2),\n",
                "        \"Norm TCI (%)\": round(rep.normalized_tci, 2),\n",
                "        \"Top 1 Phát tán\": f\"{t1} (+{rep.net_spillovers[t1]:.1f}%)\",\n",
                "        \"Top 2 Phát tán\": f\"{t2} (+{rep.net_spillovers[t2]:.1f}%)\",\n",
                "        \"Top 3 Phát tán\": f\"{t3} (+{rep.net_spillovers[t3]:.1f}%)\",\n",
                "        \"Tổn thương nhất\": f\"{r1} ({rep.net_spillovers[r1]:.1f}%)\"\n",
                "    })\n",
                "\n",
                "regime_df = pd.DataFrame(regime_summary)\n",
                "display(regime_df)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Cấu trúc Tô-pô Đồ thị Phức hợp (Hubs, Authorities, Bridges & Safe-Haven)\n",
                "Khắc phục giới hạn của Diebold-Yilmaz truyền thống bằng các chỉ số đồ thị mạng lưới:\n",
                "- **Hub Centrality (Tâm điểm Phát tán)**: Ngân hàng có sức ảnh hưởng lan truyền lớn nhất tới các nút trọng yếu.\n",
                "- **Authority Centrality (Tâm điểm Tổn thương)**: Ngân hàng nhạy cảm nhất, tiếp nhận rủi ro từ các nguồn phát lớn.\n",
                "- **Betweenness Centrality (Ngân hàng Cầu nối / Bridge Banks)**: Thắt nút cổ chai giữa Big 4 và nhóm UPCoM.\n",
                "- **Flight-to-Safety Index (Nơi trú ẩn An toàn)**: Ngân hàng hấp thụ dòng tiền phòng thủ khi hệ thống hoảng loạn."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"=== CÁC CHỈ SỐ TÔ-PÔ TRUNG TÂM MẠNG LƯỚI (COMPLEX GRAPH CENTRALITY) ===\")\n",
                "print(\"\\n1. Top 5 Ngân hàng Tâm điểm Phát tán (Hub Centrality):\")\n",
                "sorted_hub = sorted(rep_macro.eigenvector_hub.items(), key=lambda x: x[1], reverse=True)[:5]\n",
                "for b, val in sorted_hub:\n",
                "    print(f\"   - {b:5s}: {val:.2f}% (Tài sản: {ASSETS_TRILLION.get(b):,d} nghìn tỷ)\")\n",
                "\n",
                "print(\"\\n2. Top 5 Ngân hàng Tâm điểm Hấp thụ Tổn thương (Authority Centrality):\")\n",
                "sorted_auth = sorted(rep_macro.eigenvector_authority.items(), key=lambda x: x[1], reverse=True)[:5]\n",
                "for b, val in sorted_auth:\n",
                "    print(f\"   - {b:5s}: {val:.2f}% (Mức độ nhạy cảm trước sốc hệ thống)\")\n",
                "\n",
                "print(\"\\n3. Top 5 Ngân hàng Cầu nối / Thắt nút Cổ chai (Betweenness Centrality - Bridge Banks):\")\n",
                "sorted_bc = sorted(rep_macro.betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]\n",
                "for b, val in sorted_bc:\n",
                "    print(f\"   - {b:5s}: {val:.2f}% (Vai trò trung chuyển liên thị trường)\")\n",
                "\n",
                "print(\"\\n4. Top 5 Tài sản Trú ẩn Phòng thủ khi Khủng hoảng (Flight-to-Safety Absorbers):\")\n",
                "sorted_fts = sorted(rep_macro.flight_to_safety_index.items(), key=lambda x: x[1], reverse=True)[:5]\n",
                "for b, val in sorted_fts:\n",
                "    print(f\"   - {b:5s}: {val:.2f}% (Kháng chịu sốc lan truyền)\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 10. Trực quan hóa Hồ sơ Rủi ro Ròng & Quy mô Tài sản (Core vs Periphery)\n",
                "Biểu đồ bong bóng trực quan hóa vị thế của 27 ngân hàng:\n",
                "- Trục Y: NET Spillover (+: Nguồn phát tán rủi ro; -: Hấp thụ tổn thương)\n",
                "- Kích thước bong bóng: Tỷ lệ thuận với Căn bậc hai Tổng tài sản (Sub-linear Macro Footprint)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(14, 6), dpi=150)\n",
                "banks_list = list(syn_full.columns)\n",
                "net_vals = [rep_macro.net_spillovers[b] for b in banks_list]\n",
                "colors = [\"#dc2626\" if v > 0 else \"#2563eb\" for v in net_vals]\n",
                "sizes = [np.sqrt(ASSETS_TRILLION.get(b, 50)) * 18 for b in banks_list]\n",
                "\n",
                "plt.scatter(range(len(banks_list)), net_vals, s=sizes, c=colors, alpha=0.75, edgecolors=\"black\", linewidth=1.5)\n",
                "plt.axhline(0, color=\"gray\", linestyle=\"--\", alpha=0.7)\n",
                "\n",
                "for i, b in enumerate(banks_list):\n",
                "    plt.annotate(b, (i, net_vals[i]), textcoords=\"offset points\", xytext=(0, 9 if net_vals[i] >= 0 else -15),\n",
                "                 ha=\"center\", fontsize=8, fontweight=\"bold\")\n",
                "\n",
                "plt.title(\"Hồ Sơ Lan Truyền Rủi Ro Ròng (NET Spillover) & Quy Mô Tài Sản 27 Ngân Hàng (2017 - 2025)\", fontsize=13, fontweight=\"bold\")\n",
                "plt.xlabel(\"Ngân hàng Thương mại\", fontsize=11)\n",
                "plt.ylabel(\"NET Systemic Spillover (%)\", fontsize=11)\n",
                "plt.xticks(range(len(banks_list)), banks_list, rotation=90)\n",
                "plt.grid(True, linestyle=\":\", alpha=0.5)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        }
    ]

    nb_data = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open('nirvana_bank_colab_demo.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb_data, f, indent=1, ensure_ascii=False)

    print('Generated clean and complete nirvana_bank_colab_demo.ipynb (10 sections)!')

if __name__ == '__main__':
    generate_clean_colab_notebook()
