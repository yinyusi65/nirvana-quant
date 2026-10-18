import json

with open('nirvana_bank_colab_demo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Title markdown with Colab Badge
badge_md = [
    "# 🏦 NIRVANA QUANT: PHÂN TÍCH RỦI RO HỆ THỐNG MẠNG LƯỚI TOÀN BỘ 27 NGÂN HÀNG VIỆT NAM (2017 - 2025)\n",
    "### Tác giả: yinyusi65\n",
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yinyusi65/nirvana-algorithm/blob/main/nirvana_bank_colab_demo.ipynb)\n",
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
nb['cells'][0]['source'] = badge_md

# 2. Insert Step 0 Colab setup cell
step0_md = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## 0. Khởi tạo Môi trường & Tự động Cài đặt trên Google Colab\n']
}

step0_code = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        "# @title 🛠️ Bước 0: Tự động Clone và Cài đặt Nirvana Quant nếu chạy trên Google Colab\n",
        "import os, sys\n",
        "\n",
        "if 'google.colab' in sys.modules:\n",
        "    # Tự động clone mã nguồn nếu chưa có sẵn\n",
        "    if not os.path.exists('nirvana') and not os.path.exists('data'):\n",
        "        !git clone https://github.com/yinyusi65/nirvana-algorithm.git\n",
        "        if os.path.exists('nirvana-algorithm'):\n",
        "            %cd nirvana-algorithm\n",
        "    !pip install -q -e .\n",
        "    print('✓ Đã cài đặt Nirvana Quant thành công trên Google Colab!')\n",
        "else:\n",
        "    print('✓ Đang chạy trên môi trường cục bộ.')\n",
        "\n",
        "# Hàm tải dữ liệu thông minh (tự tìm trong data/ hoặc thư mục hiện tại)\n",
        "def load_dataset(filename):\n",
        "    import pandas as pd\n",
        "    for candidate in [filename, f'data/{filename}', f'../data/{filename}']:\n",
        "        if os.path.exists(candidate):\n",
        "            return pd.read_csv(candidate, index_col=0, parse_dates=True)\n",
        "    raise FileNotFoundError(f'Không tìm thấy {filename}. Vui lòng kiểm tra thư mục data/.')\n"
    ]
}

nb['cells'].insert(1, step0_md)
nb['cells'].insert(2, step0_code)

# 3. Update data reading in balanced 27 panel
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'all27_panel_data_floor_2021_2025.csv' in src:
            new_lines = []
            for line in cell['source']:
                if 'pd.read_csv' in line and 'all27_panel_data_floor_2021_2025.csv' in line:
                    new_lines.append("    df_27_balanced = load_dataset('all27_panel_data_floor_2021_2025.csv')\n")
                else:
                    new_lines.append(line)
            cell['source'] = new_lines
        elif 'all27_panel_full_2017_2025.csv' in src:
            new_lines = []
            for line in cell['source']:
                if 'pd.read_csv' in line and 'all27_panel_full_2017_2025.csv' in line:
                    new_lines.append("    raw_full = load_dataset('all27_panel_full_2017_2025.csv')\n")
                else:
                    new_lines.append(line)
            cell['source'] = new_lines

# 4. Append C++20 Colab cell at the end
cpp_md = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        "## 11. (Tuỳ chọn) Biên dịch & Thực thi Lõi C++20 Header-Only Siêu Tốc trên Colab\n",
        "Môi trường Google Colab (Ubuntu Linux) đã tích hợp sẵn trình biên dịch `g++`. Bạn có thể chạy demo C++ trực tiếp:\n"
    ]
}

cpp_code = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        "# @title ⚡ Biên dịch & Chạy Nirvana C++20 Engine\n",
        "!g++ -std=c++20 -O3 -I nirvana_cpp/include nirvana_cpp/examples/demo_banking.cpp -o demo_banking\n",
        "!./demo_banking\n"
    ]
}

nb['cells'].append(cpp_md)
nb['cells'].append(cpp_code)

with open('nirvana_bank_colab_demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Updated nirvana_bank_colab_demo.ipynb successfully!')
