import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🏦 NIRVANA: HIGH-DIMENSIONAL CYCLICAL TIME-SERIES & NETWORK ANALYSIS\n",
                "### Author: yinyusi65\n",
                "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yinyusi65/nirvana-algorithm/blob/main/nirvana_bank_colab_demo.ipynb)\n",
                "---\n",
                "Empirical case study on interconnected cyclical volatility dynamics across **all 27 commercial banks** (HOSE, HNX, UPCoM):\n",
                "`['ABB', 'ACB', 'BAB', 'BID', 'BVB', 'CTG', 'EIB', 'HDB', 'KLB', 'LPB', 'MBB', 'MSB', 'NAB', 'NVB', 'OCB', 'PGB', 'SGB', 'SHB', 'SSB', 'STB', 'TCB', 'TPB', 'VAB', 'VBB', 'VCB', 'VIB', 'VPB']`\n",
                "\n",
                "Applying the Chrono-Cancer Meta-Architecture:\n",
                "1. **Price-Band Normalization**: Neutralizes artificial 4.6x variance distortion across UPCoM (±15%), HNX (±10%), and HOSE (±7%).\n",
                "2. **Settlement Cycle T+2.5**: Multi-lag convolution kernels (Lags 1, 2, 3) with exponential decay weights `[0.5, 0.3, 0.2]`.\n",
                "3. **Sub-linear Scale Elasticity**: Square-root scale weighting to preserve empirical network topology.\n",
                "4. **Market Factor Orthogonalization**: Strips principal common market factor via SVD to expose idiosyncratic network topology.\n",
                "5. **Statistical Significance Bootstrap (p < 0.05)**: Automated apoptosis pruning of spurious noise edges.\n",
                "6. **Complex Graph Centrality**: Identifies Hub Centrality (shock transmitters), Authority Centrality (systemic absorbers), Betweenness Bridge Nodes, and Flight-to-Safety indices.\n",
                "7. **Multiverse Counterfactual Synthesis**: Reconstructs unobserved pre-listing periods with stochastic ensemble simulation and cubic Hermite boundary ramp."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 0. Environment Setup & Auto-Installation on Google Colab\n",
                "Automatically detects if the environment is Google Colab, clones the repository, and installs the `nirvana` package in editable mode."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# @title 🛠️ Step 0: Auto-clone & Setup on Google Colab (Private/Public Repo Support)\n",
                "import os, sys, getpass\n",
                "\n",
                "if 'google.colab' in sys.modules:\n",
                "    repo_name = 'nirvana-algorithm'\n",
                "    if not os.path.exists('nirvana') and not os.path.exists('data'):\n",
                "        token = None\n",
                "        try:\n",
                "            from google.colab import userdata\n",
                "            token = userdata.get('GH_TOKEN') or userdata.get('GITHUB_TOKEN')\n",
                "        except Exception:\n",
                "            token = None\n",
                "\n",
                "        if not token and not os.path.exists(repo_name):\n",
                "            print(\"🔑 Private Repository Notice: To auto-authenticate without typing every session,\")\n",
                "            print(\"   add secret name 'GH_TOKEN' in Colab Secrets (key icon on left sidebar).\")\n",
                "            token = getpass.getpass(\"Enter GitHub Personal Access Token (or press Enter if public): \").strip()\n",
                "\n",
                "        if not os.path.exists(repo_name):\n",
                "            if token:\n",
                "                !git clone https://{token}@github.com/yinyusi65/nirvana-algorithm.git\n",
                "            else:\n",
                "                !git clone https://github.com/yinyusi65/nirvana-algorithm.git\n",
                "\n",
                "        if os.path.exists(repo_name):\n",
                "            %cd {repo_name}\n",
                "\n",
                "    if os.path.exists('setup.py') or os.path.exists('pyproject.toml'):\n",
                "        !pip install -q -e .\n",
                "        print('✓ Nirvana Quant package installed successfully on Google Colab!')\n",
                "    else:\n",
                "        print('[-] Error: Repository could not be cloned. Please verify your GitHub PAT and permissions.')\n",
                "else:\n",
                "    print('✓ Running in local environment.')\n",
                "\n",
                "# Smart dataset loader (searches data/ and current directory)\n",
                "def load_dataset(filename):\n",
                "    import pandas as pd\n",
                "    for candidate in [filename, f'data/{filename}', f'../data/{filename}']:\n",
                "        if os.path.exists(candidate):\n",
                "            return pd.read_csv(candidate, index_col=0, parse_dates=True)\n",
                "    raise FileNotFoundError(f'Cannot locate {filename}. Please check the data/ directory.')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Import Dependencies & Load Nirvana Engine"
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
                "    NirvanaConnectedness,\n",
                "    MultiverseReconstructor,\n",
                "    MultiplexNetworkEngine,\n",
                "    compute_parkinson,\n",
                "    compute_yang_zhang\n",
                ")\n",
                "print('✓ Nirvana Quantitative Engine loaded successfully!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Balanced Panel Analysis: 27 Institutions (2021 - 2025)\n",
                "Evaluates the standard balanced sample across all 27 institutions over 1,000 trading sessions."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_27_balanced = load_dataset('all27_panel_data_floor_2021_2025.csv')\n",
                "print(f'[OK] Loaded balanced panel (2021-2025): {df_27_balanced.shape[0]} sessions x {df_27_balanced.shape[1]} institutions')\n",
                "\n",
                "engine = NirvanaConnectedness(kalman_q=1e-3, kalman_r=1e-2, apoptosis_threshold=0.015)\n",
                "report_27 = engine.fit(df_27_balanced)\n",
                "\n",
                "print(f'=== TOTAL CONNECTEDNESS INDEX (Raw TCI): {report_27.total_connectedness_index:.2f}% ===')\n",
                "print(f'=== DIMENSION-NORMALIZED TCI:            {report_27.normalized_tci:.2f}%\\n')\n",
                "\n",
                "print('Top 5 Systemic Shock Transmitters (NET > 0):')\n",
                "for b in report_27.top_transmitters[:5]:\n",
                "    print(f'  [+] {b}: +{report_27.net_spillovers[b]:.2f}%')\n",
                "\n",
                "print('\\nTop 5 Systemic Shock Receivers / Absorbers (NET < 0):')\n",
                "for b in report_27.top_receivers[:5]:\n",
                "    print(f'  [-] {b}: {report_27.net_spillovers[b]:.2f}%')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Directional Connectedness Heatmap (2021 - 2025)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "banks = list(df_27_balanced.columns)\n",
                "matrix_core = report_27.spillover_table.loc[banks, banks].values\n",
                "\n",
                "plt.figure(figsize=(14, 11), dpi=150)\n",
                "plt.imshow(matrix_core, cmap='viridis', interpolation='nearest')\n",
                "plt.title('Directional Connectedness Matrix: 27 Banking Institutions (2021 - 2025)', fontsize=14, fontweight='bold')\n",
                "plt.colorbar(label='Spillover Share (%)')\n",
                "plt.xticks(range(len(banks)), banks, rotation=90, fontsize=9)\n",
                "plt.yticks(range(len(banks)), banks, fontsize=9)\n",
                "plt.xlabel('Transmitter Node (Source / Column j)', fontsize=11)\n",
                "plt.ylabel('Receiver Node (Target / Row i)', fontsize=11)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Multiverse Counterfactual Synthesis: Full Historical Panel (2017 - 2025)\n",
                "Reconstructs unobserved pre-listing intervals across 3 stochastic loop universes with apoptosis outlier pruning and cubic Hermite boundary smoothing."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "raw_full = load_dataset('all27_panel_full_2017_2025.csv')\n",
                "reconstructor = MultiverseReconstructor(n_universes=3, boundary_window=30)\n",
                "syn_full = reconstructor.reconstruct(raw_full)\n",
                "print(f'[OK] Completed Multiverse Reconstruction (2017-2025): {syn_full.shape[0]} sessions, 0 missing values!')\n",
                "\n",
                "rep_full = engine.fit(syn_full)\n",
                "print(f'\\n=== FULL PERIOD CONNECTEDNESS (2017 - 2025 TCI): {rep_full.total_connectedness_index:.2f}% | Norm TCI: {rep_full.normalized_tci:.2f}% ===')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Dynamic Rolling Total Connectedness Index (2017 - 2025)\n",
                "Tracks the temporal evolution of systemic spillover intensity over 2,227 trading sessions."
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
                "plt.plot(pd.to_datetime(rolling_tci.index), rolling_tci.values, color='#0284c7', lw=2, label='Nirvana Dynamic Rolling TCI (%)')\n",
                "plt.title('Dynamic Systemic Connectedness Time Series (2017 - 2025)', fontsize=13, fontweight='bold')\n",
                "plt.xlabel('Date', fontsize=11)\n",
                "plt.ylabel('Total Connectedness Index (%)', fontsize=11)\n",
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
                "## 6. Sub-linear Asset Elasticity & Bootstrap Significance (p < 0.05)\n",
                "Incorporates institutional balance-sheet scale weighting with sub-linear elasticity $(\\gamma = 0.5)$ and circular-roll bootstrap edge verification."
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
                "print('=== ECONOMETRIC REPORT (ASSET-WEIGHTED & BOOTSTRAP p < 0.05) ===')\n",
                "print(f'Raw TCI: {rep_macro.total_connectedness_index:.2f}% | Dimension-Normalized TCI: {rep_macro.normalized_tci:.2f}%')\n",
                "\n",
                "print('\\nTop 10 Systemic Risk Transmitters (NET > 0):')\n",
                "for b in rep_macro.top_transmitters[:10]:\n",
                "    print(f'  [+] {b:5s}: NET = {rep_macro.net_spillovers[b]:+7.2f}% | Total Assets: {ASSETS_TRILLION.get(b):,d}T VND')\n",
                "\n",
                "print('\\nTop 5 Most Vulnerable Absorbers (NET < 0):')\n",
                "for b in rep_macro.top_receivers[:5]:\n",
                "    print(f'  [-] {b:5s}: NET = {rep_macro.net_spillovers[b]:+7.2f}% | Total Assets: {ASSETS_TRILLION.get(b):,d}T VND')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Market Factor Orthogonalization (SVD Beta Decomposition)\n",
                "Isolates pervasive market co-movements via deflated truncated SVD to extract the true idiosyncratic network topology."
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
                "print('=== IDIOSYNCRATIC CONNECTEDNESS NETWORK (MARKET ORTHOGONALIZED) ===')\n",
                "print(f'Common Market Factor Variance Share: {rep_ortho.market_variance_share:.2f}%')\n",
                "print(f'Idiosyncratic TCI: {rep_ortho.total_connectedness_index:.2f}% (Normalized: {rep_ortho.normalized_tci:.2f}%)\\n')\n",
                "\n",
                "print('Top 5 Idiosyncratic Spillover Transmitters:')\n",
                "for b in rep_ortho.top_transmitters[:5]:\n",
                "    print(f'  [+] {b:5s}: NET = {rep_ortho.net_spillovers[b]:+7.2f}% (Market Beta: {rep_ortho.market_beta.get(b, 0.0):.2f})')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Regime Analysis Across 4 Historical Eras (2017 - 2025)\n",
                "Evaluates structural shifts across four distinct economic and monetary regimes:\n",
                "1. **Pre-Covid (2017 - 2019)**: Normal credit expansion and monetary conditions.\n",
                "2. **Covid-19 Crisis (2020 - 2021)**: Global pandemic shock and record low interest rates.\n",
                "3. **Bond & Liquidity Shock (2022 - 2023)**: Corporate bond liquidity crunch and real estate debt restructuring.\n",
                "4. **Post-Crisis Recovery (2024 - 2025)**: Post-crisis recovery and balance-sheet divergence."
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
                "        'Regime': name,\n",
                "        'Raw TCI (%)': round(rep.total_connectedness_index, 2),\n",
                "        'Norm TCI (%)': round(rep.normalized_tci, 2),\n",
                "        'Top 1 Transmitter': f'{t1} (+{rep.net_spillovers[t1]:.1f}%)',\n",
                "        'Top 2 Transmitter': f'{t2} (+{rep.net_spillovers[t2]:.1f}%)',\n",
                "        'Top 3 Transmitter': f'{t3} (+{rep.net_spillovers[t3]:.1f}%)',\n",
                "        'Most Vulnerable': f'{r1} ({rep.net_spillovers[r1]:.1f}%)'\n",
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
                "## 9. Complex Network Centrality (Hubs, Authorities, Bridges & Safe-Havens)\n",
                "Decomposes node topology beyond linear aggregate metrics:\n",
                "- **Hub Centrality (Kleinberg HITS)**: Nodes exercising maximum spillover transmission into influential downstream nodes.\n",
                "- **Authority Centrality**: Nodes exhibiting maximum vulnerability and susceptibility to systemic shocks.\n",
                "- **Betweenness Centrality**: Structural bridge nodes connecting core and peripheral clusters.\n",
                "- **Flight-to-Safety Index**: Defensive nodes absorbing capital reallocation during high-stress episodes."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('=== COMPLEX NETWORK GRAPH CENTRALITY ===')\n",
                "print('\\n1. Top 5 Shock Transmitters (Hub Centrality):')\n",
                "sorted_hub = sorted(rep_macro.eigenvector_hub.items(), key=lambda x: x[1], reverse=True)[:5]\n",
                "for b, val in sorted_hub:\n",
                "    print(f'   - {b:5s}: {val:.2f}% (Total Assets: {ASSETS_TRILLION.get(b):,d}T VND)')\n",
                "\n",
                "print('\\n2. Top 5 Vulnerable Shock Absorbers (Authority Centrality):')\n",
                "sorted_auth = sorted(rep_macro.eigenvector_authority.items(), key=lambda x: x[1], reverse=True)[:5]\n",
                "for b, val in sorted_auth:\n",
                "    print(f'   - {b:5s}: {val:.2f}% (Systemic Vulnerability Metric)')\n",
                "\n",
                "print('\\n3. Top 5 Bridge Nodes (Betweenness Centrality):')\n",
                "sorted_bc = sorted(rep_macro.betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]\n",
                "for b, val in sorted_bc:\n",
                "    print(f'   - {b:5s}: {val:.2f}% (Cross-cluster Intermediation)')\n",
                "\n",
                "print('\\n4. Top 5 Defensive Assets (Flight-to-Safety Absorbers):')\n",
                "sorted_fts = sorted(rep_macro.flight_to_safety_index.items(), key=lambda x: x[1], reverse=True)[:5]\n",
                "for b, val in sorted_fts:\n",
                "    print(f'   - {b:5s}: {val:.2f}% (Flight-to-Quality Resilience)')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 10. Bubble Chart: Net Spillover vs. Structural Scale\n",
                "Visualizes institutional systemic positioning:\n",
                "- **Y-axis**: Net Spillover (+: Net Shock Transmitter; -: Net Shock Absorber)\n",
                "- **Bubble Size**: Proportional to Square Root of Total Assets $(\\sqrt{\\text{Assets}})$"
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
                "colors = ['#dc2626' if v > 0 else '#2563eb' for v in net_vals]\n",
                "sizes = [np.sqrt(ASSETS_TRILLION.get(b, 50)) * 18 for b in banks_list]\n",
                "\n",
                "plt.scatter(range(len(banks_list)), net_vals, s=sizes, c=colors, alpha=0.75, edgecolors='black', linewidth=1.5)\n",
                "plt.axhline(0, color='gray', linestyle='--', alpha=0.7)\n",
                "\n",
                "for i, b in enumerate(banks_list):\n",
                "    plt.annotate(b, (i, net_vals[i]), textcoords='offset points', xytext=(0, 9 if net_vals[i] >= 0 else -15),\n",
                "                 ha='center', fontsize=8, fontweight='bold')\n",
                "\n",
                "plt.title('Net Systemic Spillover vs. Balance-Sheet Scale across 27 Institutions (2017 - 2025)', fontsize=13, fontweight='bold')\n",
                "plt.xlabel('Commercial Banking Institution', fontsize=11)\n",
                "plt.ylabel('Net Systemic Spillover (%)', fontsize=11)\n",
                "plt.xticks(range(len(banks_list)), banks_list, rotation=90)\n",
                "plt.grid(True, linestyle=':', alpha=0.5)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 11. (Optional) High-Performance C++20 Header-Only Engine Execution\n",
                "Google Colab's Linux environment includes GCC (`g++`). You can compile and execute the native C++20 engine directly:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# @title ⚡ Compile and Execute Nirvana C++20 Engine\n",
                "!g++ -std=c++20 -O3 -I nirvana_cpp/include nirvana_cpp/examples/demo_banking.cpp -o demo_banking\n",
                "!./demo_banking\n"
            ]
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open('nirvana_bank_colab_demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print('Generated nirvana_bank_colab_demo.ipynb in fluent academic English successfully!')
