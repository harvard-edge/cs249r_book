#!/usr/bin/env python3
"""Data plots for Lecture 6b (Network Architectures, Part 2).

Every plotted value and every number in a label comes from calcs.py, which
reads MLSysIM: the model registry (GPT-2 XL, Llama 2 and 3.1, PaLM, OPT-13B),
the H100 and A100, the attention, KV-cache, decode, and recommender scenarios,
and the cited series in Literature (published context windows,
FlashAttention's Tables 20 and 21, Fiction.LiveBench). Positions, axis
limits, and label offsets are layout and stay here.

Run from this folder: python3 make_figs.py (values: calcs.py)
"""
from types import SimpleNamespace
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "mlsysim"))
from mlsysim.core.units import Q_
import mlsysim
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
from mlsysim.core.series_loader import load_series
out = Path(__file__).parent
C = sp.C
plt = sp.plt

def save(fig, path, margins):
    """Save with fixed margins so every step of a build keeps one canvas."""
    fig.subplots_adjust(**margins)
    fig.savefig(path, format='svg', facecolor='white')
    plt.close(fig)
KV_XL = 307200.0
KI = 1024

def k_label(tokens):
    """1,000,000 -> 1M, 128,000 -> 128K: the short form the plots print."""
    return f'{tokens / 1000000.0:.0f}M' if tokens >= 1000000.0 else f'{tokens / 1000.0:.0f}K'

def ki_label(tokens):
    """4,096 -> 4K: a power-of-two length in binary thousands."""
    return f'{tokens / KI:.0f}K'

def context_windows():
    fig, ax = plt.subplots(figsize=(5.8, 1.95))
    gpt4t, claude = (128000.0, 200000.0)
    gemini = 1000000.0
    pts = [(2018.8, 512.0, 'BERT 512', (0, -9), 'center', 'top'), (2019.15, 1024.0, 'GPT-2 1,024', (0, 9), 'center', 'bottom'), (2020.4, 2048.0, 'GPT-3 2,048', (9, 0), 'left', 'center'), (2023.85, gpt4t, f'GPT-4 Turbo {k_label(gpt4t)}', (-9, -5), 'right', 'center'), (2023.89, claude, f'Claude 2.1 {k_label(claude)}', (-9, 7), 'right', 'center'), (2024.12, gemini, f'Gemini 1.5 Pro {k_label(gemini)} (preview)', (-9, 2), 'right', 'center')]
    for x, y, label, off, ha, va in pts:
        col = C['danger'] if y > 2048.0 else C['memory']
        ax.plot([x], [y], 'o', ms=10, color=col, mec='white', mew=1.5, zorder=5)
        ax.annotate(label, (x, y), xytext=off, textcoords='offset points', ha=ha, va=va, fontsize=10.5, fontweight='bold', color=col)
    ax.text(2021.75, 8000.0, 'GPT-3 to Gemini 1.5:\nabout 500× in four years', color=C['text'], fontsize=11.5, fontweight='bold', ha='center')
    ax.set_yscale('log')
    ax.set_xlim(2018.3, 2024.5)
    ax.set_ylim(100, 4000000.0)
    ax.set_xticks([2018, 2019, 2020, 2021, 2022, 2023, 2024])
    ax.set_xticklabels(['2018', '2019', '2020', '2021', '2022', '2023', '2024'])
    ax.set_yticks([1000.0, 10000.0, 100000.0, 1000000.0])
    ax.set_yticklabels(['1K', '10K', '100K', '1M'])
    ax.minorticks_off()
    ax.set_ylabel('Context window\n(tokens, log scale)')
    ax.grid(color=C['grid'], lw=0.6)
    save(fig, out / 'context_windows.svg', dict(left=0.15, right=0.98, bottom=0.14, top=0.97))

def state_memory():
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    s = np.logspace(1, 4, 50)
    rnn = 2048
    ax.plot(s, np.full_like(s, rnn), color=C['memory'], lw=3.5)
    ax.text(12, rnn * 1.9, 'RNN state: 2 KB at any length', color=C['memory'], fontsize=12, fontweight='bold')
    ax.plot(s, KV_XL * s, color=C['danger'], lw=3.5)
    ax.text(10500, 3072000000.0, '3.1 GB', color=C['danger'], fontsize=12, fontweight='bold', va='center')
    ax.text(1500, 100000.0, 'Transformer: 307.2 KB\nkept per token (GPT-2 XL)', color=C['danger'], fontsize=12, fontweight='bold', ha='center', va='center')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(10, 10000.0)
    ax.set_ylim(500.0, 30000000000.0)
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(['10', '100', '1,000', '10,000'])
    ax.set_yticks([1000.0, 1000000.0, 1000000000.0])
    ax.set_yticklabels(['1 KB', '1 MB', '1 GB'])
    ax.minorticks_off()
    ax.set_xlabel('Sequence length (tokens, log scale)')
    ax.set_ylabel('State per sequence')
    ax.grid(color=C['grid'], lw=0.6)
    save(fig, out / 'state_memory.svg', dict(left=0.14, right=0.86, bottom=0.21, top=0.96))

def gpt2xl_params():
    attn = 491520000 / 1000000000.0
    mlp = 983040000 / 1000000000.0
    table = 80411200 / 1000000000.0
    published = 1.5
    fig, ax = plt.subplots(figsize=(5.8, 1.3))
    segs = [(attn, 'compute', 'Attention\nprojections 0.49B'), (mlp, 'memory', 'MLP 0.98B'), (table, 'network', '')]
    left = 0
    for w, ck, txt in segs:
        ax.barh(0, w, left=left, color=C[ck], height=0.55, edgecolor='white', lw=1.5)
        if txt:
            ax.text(left + w / 2, 0, txt, ha='center', va='center', color='white', fontsize=11.5, fontweight='bold', linespacing=1.0)
        left += w
    ax.annotate('Token table 0.08B', (left - table / 2, -0.28), xytext=(left - 0.02, -0.72), ha='right', va='center', fontsize=11, fontweight='bold', color=C['network'], arrowprops=dict(arrowstyle='-', color=C['network'], lw=1.2))
    ax.text(left + 0.03, 0, '= 1.55B', va='center', fontsize=13, fontweight='bold', color=C['text'])
    ax.plot([published, published], [-0.42, 0.42], color=C['danger'], lw=2.5)
    ax.text(published, 0.5, 'published 1.5B', ha='center', va='bottom', fontsize=11, fontweight='bold', color=C['danger'])
    ax.set_xlim(0, 1.9)
    ax.set_ylim(-0.95, 0.85)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([0, 0.5, 1.0, 1.5])
    ax.set_xticklabels(['0', '0.5B', '1.0B', '1.5B'])
    save(fig, out / 'gpt2xl_params.svg', dict(left=0.03, right=0.97, bottom=0.17, top=0.97))

def block_macs(step):
    fig, ax = plt.subplots(figsize=(5.8, 2.45))
    s = np.logspace(np.log10(128), np.log10(200000.0), 200)
    ax.plot(s, [3932160000.0, 4080191322.0, 4233795478.0, 4393182264.0, 4558569374.0, 4730182699.0, 4908256633.0, 5093034394.0, 5284768356.0, 5483720395.0, 5690162244.0, 5904375868.0, 6126653844.0, 6357299766.0, 6596628657.0, 6844967398.0, 7102655177.0, 7370043950.0, 7647498924.0, 7935399055.0, 8234137564.0, 8544122476.0, 8865777177.0, 9199540992.0, 9545869782.0, 9905236575.0, 10278132200.0, 10665065970.0, 11066566380.0, 11483181790.0, 11915481230.0, 12364055160.0, 12829516240.0, 13312500210.0, 13813666740.0, 14333700350.0, 14873311310.0, 15433236630.0, 16014241070.0, 16617118190.0, 17242691400.0, 17891815140.0, 18565376000.0, 19264293940.0, 19989523560.0, 20742055400.0, 21522917290.0, 22333175750.0, 23173937460.0, 24046350740.0, 24951607160.0, 25890943150.0, 26865641670.0, 27877034000.0, 28926501520.0, 30015477630.0, 31145449660.0, 32317960980.0, 33534613030.0, 34797067530.0, 36107048790.0, 37466346020.0, 38876815770.0, 40340384510.0, 41859051220.0, 43434890120.0, 45070053550.0, 46766774850.0, 48527371450.0, 50354248020.0, 52249899750.0, 54216915780.0, 56257982710.0, 58375888280.0, 60573525190.0, 62853895020.0, 65220112370.0, 67675409070.0, 70223138640.0, 72866780830.0, 75609946390.0, 78456382030.0, 81409975470.0, 84474760810.0, 87654924020.0, 90954808640.0, 94378921750.0, 97931940090.0, 101618716500.0, 105444286400.0, 109413874900.0, 113532903800.0, 117806998900.0, 122241997900.0, 126843958300.0, 131619165600.0, 136574141700.0, 141715654500.0, 147050726200.0, 152586643800.0, 158330968200.0, 164291545200.0, 170476516000.0, 176894328100.0, 183553747200.0, 190463868800.0, 197634131100.0, 205074327200.0, 212794619400.0, 220805552000.0, 229118066700.0, 237743517000.0, 246693683600.0, 255980791000.0, 265617523700.0, 275617043900.0, 285993009100.0, 296759591200.0, 307931495400.0, 319523980500.0, 331552880000.0, 344034623200.0, 356986258000.0, 370425474000.0, 384370627000.0, 398840763600.0, 413855647400.0, 429435786300.0, 445602459900.0, 462377749200.0, 479784566200.0, 497846685700.0, 516588777400.0, 536036439700.0, 556216234800.0, 577155724700.0, 598883509200.0, 621429264700.0, 644823784700.0, 669099022100.0, 694288132700.0, 720425520400.0, 747546884300.0, 775689267600.0, 804891107800.0, 835192289700.0, 866634199200.0, 899259780700.0, 933113595000.0, 968241880600.0, 1004692617000.0, 1042515588000.0, 1081762455000.0, 1122486822000.0, 1164744312000.0, 1208592639000.0, 1254091695000.0, 1301303622000.0, 1350292905000.0, 1401126453000.0, 1453873696000.0, 1508606679000.0, 1565400157000.0, 1624331700000.0, 1685481798000.0, 1748933972000.0, 1814774886000.0, 1883094467000.0, 1953986029000.0, 2027546397000.0, 2103876041000.0, 2183079215000.0, 2265264097000.0, 2350542936000.0, 2439032209000.0, 2530852776000.0, 2626130050000.0, 2724994161000.0, 2827580141000.0, 2934028106000.0, 3044483443000.0, 3159097017000.0, 3278025369000.0, 3401430935000.0, 3529482266000.0, 3662354256000.0, 3800228387000.0, 3943292971000.0, 4091743409000.0, 4245782458000.0, 4405620509000.0, 4571475874000.0, 4743575081000.0, 4922153189000.0, 5107454104000.0, 5299730915000.0, 5499246239000.0, 5706272579000.0, 5921092697000.0, 6144000000000.0], color=C['compute'], lw=3.2)
    ax.plot(s, [52428800.0, 56450606.38, 60780924.99, 65443421.78, 70463578.08, 75868829.92, 81688717.92, 87955048.76, 94702069.02, 101966652.3, 109788500.9, 118210362.4, 127278263.8, 137041762.7, 147554218.3, 158873083.1, 171060216.5, 184182223.3, 198310817.5, 213523214.3, 229902552.0, 247538346.7, 266526980.8, 286972230.5, 308985832.7, 332688095.5, 358208555.6, 385686687.0, 415272662.2, 447128173.7, 481427317.3, 518357543.7, 558120682.9, 600934047.3, 647031619.3, 696665330.0, 750106436.1, 807647002.5, 869601498.2, 936308515.1, 1008132619.0, 1085466340.0, 1168732320.0, 1258385623.0, 1354916220.0, 1458851665.0, 1570759985.0, 1691252778.0, 1820988556.0, 1960676350.0, 2111079576.0, 2273020213.0, 2447383296.0, 2635121748.0, 2837261591.0, 3054907556.0, 3289249114.0, 3541566983.0, 3813240122.0, 4105753272.0, 4420705067.0, 4759816773.0, 5124941694.0, 5518075299.0, 5941366132.0, 6397127549.0, 6887850365.0, 7416216463.0, 7985113456.0, 8597650463.0, 9257175103.0, 9967291791.0, 10731881440.0, 11555122660.0, 12441514600.0, 13395901550.0, 14423499410.0, 15529924160.0, 16721222600.0, 18003905400.0, 19384982620.0, 20872002090.0, 22473090640.0, 24196998470.0, 26053147060.0, 28051680550.0, 30203521290.0, 32520429450.0, 35015067330.0, 37701068550.0, 40593112580.0, 43707004920.0, 47059763540.0, 50669711830.0, 54556578790.0, 58741606790.0, 63247667750.0, 68099388050.0, 73323283180.0, 78947902620.0, 85003985880.0, 91524630500.0, 98545472900.0, 106104883200.0, 114244174900.0, 123007830600.0, 132443745100.0, 142603487500.0, 153542582300.0, 165320813600.0, 178002551600.0, 191657104000.0, 206359095300.0, 222188874500.0, 239232953900.0, 257584482400.0, 277343754300.0, 298618757300.0, 321525763100.0, 346189962200.0, 372746148800.0, 401339457000.0, 432126154200.0, 465274494800.0, 500965640300.0, 539394648900.0, 580771541700.0, 625322450600.0, 673290853800.0, 724938906900.0, 780548875400.0, 840424677300.0, 904893544200.0, 974307809300.0, 1049046834000.0, 1129519079000.0, 1216164339000.0, 1309456146000.0, 1409904356000.0, 1518057934000.0, 1634507959000.0, 1759890850000.0, 1894891847000.0, 2040248752000.0, 2196755966000.0, 2365268828000.0, 2546708290000.0, 2742065949000.0, 2952409469000.0, 3178888412000.0, 3422740525000.0, 3685298502000.0, 3967997267000.0, 4272381818000.0, 4600115667000.0, 4952989936000.0, 5332933145000.0, 5742021748000.0, 6182491483000.0, 6656749594000.0, 7167387983000.0, 7717197376000.0, 8309182575000.0, 8946578881000.0, 9632869774000.0, 10371805950000.0, 11167425830000.0, 12024077600000.0, 12946443010000.0, 13939562960000.0, 15008865010000.0, 16160193080000.0, 17399839370000.0, 18734578760000.0, 20171705850000.0, 21719074760000.0, 23385142130000.0, 25179013320000.0, 27110492130000.0, 29190134430000.0, 31429305820000.0, 33840243750000.0, 36436124410000.0, 39231134730000.0, 42240549940000.0, 45480816990000.0, 48969644490000.0, 52726099500000.0, 56770711680000.0, 61125585540000.0, 65814521190000.0, 70863144480000.0, 76299047010000.0, 82151936900000.0, 88453801210000.0, 95239080710000.0, 102544858100000.0, 110411060700000.0, 118880678700000.0, 128000000000000.0], color=C['memory'], lw=3.2)
    ax.text(140, 300000000000.0, 'Projections + MLP: 12 S d²', color=C['compute'], fontsize=11.5, fontweight='bold')
    ax.text(14000.0, 30000000000.0, 'Attention\nproducts: 2 S² d', color=C['memory'], fontsize=11.5, fontweight='bold', va='top', linespacing=1.05)
    for k, b, label, col, pos, ha, va in [(2, SimpleNamespace(attention_macs=3355443200, attention_projections=4, attention_share=0.09638554217, attention_step_flops=Q_(0.9663676416, 'TFLOP'), attention_to_dense=0.1066666667, attention_weights=491520000, attention_weights_per_block=10240000, batch_size=1, block_coefficient=12, block_macs=34812723200, block_rule_parameters=1554971200, block_weights=1474560000, context_tokens=1024, crossover=9600.0, crossover_multiple=6.0, dense_macs=31457280000, dense_to_attention=9.375, exact_parameters=1557611200, ffn_multiple=4, flops_per_token=Q_(2949120000, 'flop'), hardware=None, layer_forward_flops=Q_(69.6254464, 'GFLOP'), mlp_share=0.6666666667, mlp_weights=983040000, mlp_weights_per_block=20480000, name='GPT-2 XL decoder blocks at its 1,024-token window', output_layer_flops=Q_(160822400, 'flop'), prompt_attention_flops=Q_(322122547200, 'flop'), prompt_dense_flops=Q_(3019898880000, 'flop'), prompt_flops=Q_(3342021427200, 'flop'), six_p_step_flops=Q_(9.216, 'TFLOP'), token_table=80411200, training_step_flops=Q_(10.02606428, 'TFLOP'), training_to_forward=3.0, weights_per_block=30720000, width=1600, workload=SimpleNamespace(architecture='Transformer', context_length=1024, data_rate=None, embedding_entries=None, head_dim=None, head_width=64, heads=25, hidden_dim=1600, inference_energy=None, inference_flops=Q_(3000000000.0, 'flop'), kv_heads=None, kv_lora_rank=None, kv_width=1600, layers=48, metadata=SimpleNamespace(last_verified=None, source_label='Radford et al. (2019), Language Models are Unsupervised Multitask Learners', version=None), model_size=None, name='GPT-2 (1.5B)', parameter_range_max=None, parameter_range_min=None, parameters=Q_(1500000000.0, 'param'), qk_rope_head_dim=None, queries_per_kv_head=1, release_year=None, training_accelerators_ref=None, training_co2e=None, training_dataset_size=Q_(40, 'GB'), training_days_ref=None, training_energy_mwh=None, training_gpu_days=None, training_hardware_label=None, training_ops=Q_(1.5e+21, 'flop'), training_tokens=None, vocab_size=50257)), '1,024 tokens:\nattention is about 10%', '#1A1A2E', (1150, 1200000000.0), 'left', 'top'), (3, SimpleNamespace(attention_macs=294912000000, attention_projections=4, attention_share=0.5, attention_step_flops=Q_(84.934656, 'TFLOP'), attention_to_dense=1.0, attention_weights=491520000, attention_weights_per_block=10240000, batch_size=1, block_coefficient=12, block_macs=589824000000, block_rule_parameters=1554971200, block_weights=1474560000, context_tokens=9600, crossover=9600.0, crossover_multiple=6.0, dense_macs=294912000000, dense_to_attention=1.0, exact_parameters=1557611200, ffn_multiple=4, flops_per_token=Q_(2949120000, 'flop'), hardware=None, layer_forward_flops=Q_(1179.648, 'GFLOP'), mlp_share=0.6666666667, mlp_weights=983040000, mlp_weights_per_block=20480000, name='GPT-2 XL decoder blocks at its 1,024-token window', output_layer_flops=Q_(160822400, 'flop'), prompt_attention_flops=Q_(28311552000000, 'flop'), prompt_dense_flops=Q_(28311552000000, 'flop'), prompt_flops=Q_(56623104000000, 'flop'), six_p_step_flops=Q_(86.4, 'TFLOP'), token_table=80411200, training_step_flops=Q_(169.869312, 'TFLOP'), training_to_forward=3.0, weights_per_block=30720000, width=1600, workload=SimpleNamespace(architecture='Transformer', context_length=1024, data_rate=None, embedding_entries=None, head_dim=None, head_width=64, heads=25, hidden_dim=1600, inference_energy=None, inference_flops=Q_(3000000000.0, 'flop'), kv_heads=None, kv_lora_rank=None, kv_width=1600, layers=48, metadata=SimpleNamespace(last_verified=None, source_label='Radford et al. (2019), Language Models are Unsupervised Multitask Learners', version=None), model_size=None, name='GPT-2 (1.5B)', parameter_range_max=None, parameter_range_min=None, parameters=Q_(1500000000.0, 'param'), qk_rope_head_dim=None, queries_per_kv_head=1, release_year=None, training_accelerators_ref=None, training_co2e=None, training_dataset_size=Q_(40, 'GB'), training_days_ref=None, training_energy_mwh=None, training_gpu_days=None, training_hardware_label=None, training_ops=Q_(1.5e+21, 'flop'), training_tokens=None, vocab_size=50257)), '6d = 9,600:\nequal', '#CB202D', (7500, 1500000000000.0), 'right', 'bottom'), (4, SimpleNamespace(attention_macs=32000000000000, attention_projections=4, attention_share=0.9124087591, attention_step_flops=Q_(9216.0, 'TFLOP'), attention_to_dense=10.41666667, attention_weights=491520000, attention_weights_per_block=10240000, batch_size=1, block_coefficient=12, block_macs=35072000000000, block_rule_parameters=1554971200, block_weights=1474560000, context_tokens=100000, crossover=9600.0, crossover_multiple=6.0, dense_macs=3072000000000, dense_to_attention=0.096, exact_parameters=1557611200, ffn_multiple=4, flops_per_token=Q_(2949120000, 'flop'), hardware=None, layer_forward_flops=Q_(70144.0, 'GFLOP'), mlp_share=0.6666666667, mlp_weights=983040000, mlp_weights_per_block=20480000, name='GPT-2 XL decoder blocks at a 100,000-token context', output_layer_flops=Q_(160822400, 'flop'), prompt_attention_flops=Q_(3072000000000000, 'flop'), prompt_dense_flops=Q_(294912000000000, 'flop'), prompt_flops=Q_(3366912000000000, 'flop'), six_p_step_flops=Q_(900.0, 'TFLOP'), token_table=80411200, training_step_flops=Q_(10100.736, 'TFLOP'), training_to_forward=3.0, weights_per_block=30720000, width=1600, workload=SimpleNamespace(architecture='Transformer', context_length=1024, data_rate=None, embedding_entries=None, head_dim=None, head_width=64, heads=25, hidden_dim=1600, inference_energy=None, inference_flops=Q_(3000000000.0, 'flop'), kv_heads=None, kv_lora_rank=None, kv_width=1600, layers=48, metadata=SimpleNamespace(last_verified=None, source_label='Radford et al. (2019), Language Models are Unsupervised Multitask Learners', version=None), model_size=None, name='GPT-2 (1.5B)', parameter_range_max=None, parameter_range_min=None, parameters=Q_(1500000000.0, 'param'), qk_rope_head_dim=None, queries_per_kv_head=1, release_year=None, training_accelerators_ref=None, training_co2e=None, training_dataset_size=Q_(40, 'GB'), training_days_ref=None, training_energy_mwh=None, training_gpu_days=None, training_hardware_label=None, training_ops=Q_(1.5e+21, 'flop'), training_tokens=None, vocab_size=50257)), '100K tokens: attention\nis 10× the rest', '#006395', (55000.0, 180000000000000.0), 'center', 'bottom')]:
        alpha = 1.0 if step >= k else 0.0
        x = {2: 1024, 3: 9600, 4: 100000}[k]
        lo, hi = sorted(({2: 3355443200, 3: 294912000000, 4: 32000000000000}[k], {2: 31457280000, 3: 294912000000, 4: 3072000000000}[k]))
        if k == 3:
            ax.plot([x], [hi], 'o', ms=10, color=col, mec='white', zorder=5, alpha=alpha)
        else:
            ax.plot([x, x], [lo, hi], color=col, lw=1.4, ls='--', alpha=alpha)
            for y in (lo, hi):
                ax.plot([x], [y], 'o', ms=7, color=col, mec='white', zorder=5, alpha=alpha)
        ax.text(*pos, label, color=col, fontsize=11, fontweight='bold', ha=ha, va=va, alpha=alpha, linespacing=1.05)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(128, 200000.0)
    ax.set_ylim(10000000.0, 1e+17)
    ax.set_xticks([128, 1024.0, 9600.0, 100000.0])
    ax.set_xticklabels(['128', '1,024', '9,600', '100K'])
    ax.set_yticks([1000000000.0, 1000000000000.0, 1000000000000000.0])
    ax.set_yticklabels(['10⁹', '10¹²', '10¹⁵'])
    ax.minorticks_off()
    ax.set_xlabel('Context S (tokens, log scale)')
    ax.set_ylabel('MACs per block\n(log scale)')
    ax.grid(color=C['grid'], lw=0.6)
    save(fig, out / f'block_macs_step{step}.svg', dict(left=0.13, right=0.97, bottom=0.2, top=0.97))

def quadratic():
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    s = np.linspace(0, 8800, 200)
    per_pair = 2.4e-08
    ax.plot(s, s ** 2 * per_pair, color=C['compute'], lw=3.5)
    growth = 4.0
    for sc, label, col in [(SimpleNamespace(bytes_per_score=Q_(2, 'B'), hardware=SimpleNamespace(accelerator_count=None, battery_capacity=None, compute=SimpleNamespace(clock_rate=None, core_count=None, fp32_core_count=16896, fp32_cores_per_sm=128, max_registers_per_thread=255, max_threads_per_sm=2048, peak_flops=Q_(989.0, 'TFLOPs / s'), precision_flops={'tf32': Q_(494.5, 'TFLOPs / s'), 'fp16_dense': Q_(989.0, 'TFLOPs / s'), 'fp16_sparse': Q_(1979.0, 'TFLOPs / s'), 'fp8': Q_(1979.0, 'TFLOPs / s'), 'int8': Q_(1979, 'TOPS'), 'tf32_sparse': Q_(989.0, 'TFLOPs / s'), 'int8_sparse': Q_(3958, 'TOPS'), 'fp32': Q_(67.0, 'TFLOPs / s')}, sm_count=132, systolic_array_dim=None, systolic_macs_per_cycle=None, tensor_core_count=528, tensor_cores_per_sm=4, warp_size=32), die_area=Q_(814, 'mm ** 2'), die_to_die_bandwidth=None, dispatch_tax=Q_(0.01, 'ms'), embodied_carbon=Q_(164.0, 'kg'), embodied_carbon_kg=164.0, interconnect=SimpleNamespace(bandwidth=Q_(64.0, 'GB / s'), bandwidth_per_direction=Q_(64.0, 'GB / s'), direction='per_direction', latency=None, name='PCIe Gen5 x16'), memory=SimpleNamespace(bandwidth=Q_(3.35, 'TB / s'), bus_width=None, capacity=Q_(80, 'GiB'), flash_bandwidth=None, flash_capacity=None, hbm_stacks=5, l2_cache=Q_(50, 'MiB'), pin_data_rate=None, register_file_per_sm=Q_(256, 'KiB'), shared_memory_per_sm=Q_(228, 'KiB'), sram_bandwidth=None, sram_capacity=None), metadata=SimpleNamespace(last_verified=None, source_label='NVIDIA H100 datasheet; NVIDIA Hopper Architecture In-Depth; Choquette (2023), IEEE Micro; NVIDIA HGX H100 product carbon footprint', version=None), mig_profiles=None, name='NVIDIA H100', nvlink=SimpleNamespace(bandwidth=Q_(900.0, 'GB / s'), bandwidth_per_direction=Q_(450.0, 'GB / s'), direction='bidirectional_total', latency=Q_(500, 'ns'), name='NVLink 4.0'), release_year=2022, storage=SimpleNamespace(bandwidth=Q_(7.0, 'GB / s'), capacity=Q_(2, 'TiB'), latency=None), tdp=Q_(700, 'W'), tdp_max=None, tdp_min=None, transistor_count=80000000000.0, unit_cost=Q_(25000, 'dollar'), unit_cost_max=Q_(30000, 'dollar')), heads=12, layer_bytes=Q_(0.402653184, 'GB'), layer_device_multiple=0.0050331648, layers=32, name='The same scores at 4,096 tokens', scores_per_layer=201326592, seq_len=4096, total_bytes=Q_(12.88490189, 'GB')), f'{ki_label(4096)}: 0.40 GB', C['data']), (SimpleNamespace(bytes_per_score=Q_(2, 'B'), hardware=SimpleNamespace(accelerator_count=None, battery_capacity=None, compute=SimpleNamespace(clock_rate=None, core_count=None, fp32_core_count=16896, fp32_cores_per_sm=128, max_registers_per_thread=255, max_threads_per_sm=2048, peak_flops=Q_(989.0, 'TFLOPs / s'), precision_flops={'tf32': Q_(494.5, 'TFLOPs / s'), 'fp16_dense': Q_(989.0, 'TFLOPs / s'), 'fp16_sparse': Q_(1979.0, 'TFLOPs / s'), 'fp8': Q_(1979.0, 'TFLOPs / s'), 'int8': Q_(1979, 'TOPS'), 'tf32_sparse': Q_(989.0, 'TFLOPs / s'), 'int8_sparse': Q_(3958, 'TOPS'), 'fp32': Q_(67.0, 'TFLOPs / s')}, sm_count=132, systolic_array_dim=None, systolic_macs_per_cycle=None, tensor_core_count=528, tensor_cores_per_sm=4, warp_size=32), die_area=Q_(814, 'mm ** 2'), die_to_die_bandwidth=None, dispatch_tax=Q_(0.01, 'ms'), embodied_carbon=Q_(164.0, 'kg'), embodied_carbon_kg=164.0, interconnect=SimpleNamespace(bandwidth=Q_(64.0, 'GB / s'), bandwidth_per_direction=Q_(64.0, 'GB / s'), direction='per_direction', latency=None, name='PCIe Gen5 x16'), memory=SimpleNamespace(bandwidth=Q_(3.35, 'TB / s'), bus_width=None, capacity=Q_(80, 'GiB'), flash_bandwidth=None, flash_capacity=None, hbm_stacks=5, l2_cache=Q_(50, 'MiB'), pin_data_rate=None, register_file_per_sm=Q_(256, 'KiB'), shared_memory_per_sm=Q_(228, 'KiB'), sram_bandwidth=None, sram_capacity=None), metadata=SimpleNamespace(last_verified=None, source_label='NVIDIA H100 datasheet; NVIDIA Hopper Architecture In-Depth; Choquette (2023), IEEE Micro; NVIDIA HGX H100 product carbon footprint', version=None), mig_profiles=None, name='NVIDIA H100', nvlink=SimpleNamespace(bandwidth=Q_(900.0, 'GB / s'), bandwidth_per_direction=Q_(450.0, 'GB / s'), direction='bidirectional_total', latency=Q_(500, 'ns'), name='NVLink 4.0'), release_year=2022, storage=SimpleNamespace(bandwidth=Q_(7.0, 'GB / s'), capacity=Q_(2, 'TiB'), latency=None), tdp=Q_(700, 'W'), tdp_max=None, tdp_min=None, transistor_count=80000000000.0, unit_cost=Q_(25000, 'dollar'), unit_cost_max=Q_(30000, 'dollar')), heads=12, layer_bytes=Q_(1.610612736, 'GB'), layer_device_multiple=0.0201326592, layers=32, name='The same scores at 8,192 tokens', scores_per_layer=805306368, seq_len=8192, total_bytes=Q_(51.53960755, 'GB')), f'{ki_label(8192)}: 1.61 GB\n= {growth:.0f}×', C['danger'])]:
        x, y = ({'#008F45': 4096, '#CB202D': 8192}[col], {'#008F45': 0.402653184, '#CB202D': 1.610612736}[col])
        ax.plot([x, x], [0, y], ls='--', color=col, lw=1.2)
        ax.plot([x], [y], 'o', ms=10, color=col, mec='white', mew=1.5, zorder=5)
        ax.text(x - 400, y + 0.05, label, ha='right', va='bottom', color=col, fontsize=12, fontweight='bold')
    ax.set_xlim(0, 8800)
    ax.set_ylim(0, 2.1)
    ax.set_xticks([0, 4096, 8192])
    ax.set_xticklabels(['0', ki_label(4096), ki_label(8192)])
    ax.set_xlabel('Context (tokens)')
    ax.set_ylabel('Scores, one layer (GB)')
    ax.text(0.03, 0.97, '12 heads, FP16', transform=ax.transAxes, fontsize=10, color=C['neutral'], va='top')
    ax.grid(color=C['grid'], lw=0.6)
    save(fig, out / 'quadratic.svg', dict(left=0.22, right=0.96, bottom=0.19, top=0.97))

def decode_ridge():
    ridge = 295.2238806
    fig, ax = plt.subplots(figsize=(3.4, 1.1))
    rows = [('Decode,\nbatch 1', 1.0, 'memory', '1 FLOP/byte'), ('H100\nridge', ridge, 'compute', '295 FLOP/byte')]
    for i, (label, v, ck, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v - 0.3, left=0.3, color=C[ck], height=0.6)
        ax.text(0.26, y, label, ha='right', va='center', fontsize=11.5, fontweight='bold', color=C['text'], linespacing=1.0)
        inside = v > 50
        ax.text(v / 1.15 if inside else v * 1.2, y, txt, va='center', ha='right' if inside else 'left', fontsize=11.5, fontweight='bold', color='white' if inside else C[ck])
    ax.set_xscale('log')
    ax.set_xlim(0.3, 1000)
    ax.set_ylim(-0.55, 1.55)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(['1', '10', '100', '1,000'])
    ax.minorticks_off()
    ax.set_xlabel('FLOP per byte (log scale)')
    ax.grid(axis='x', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    save(fig, out / 'decode_ridge.svg', dict(left=0.26, right=0.92, bottom=0.4, top=0.98))

def context_memory(step):
    fig, ax = plt.subplots(figsize=(5.8, 2.35))
    s = np.logspace(np.log10(512), np.log10(131072), 200)
    dev = 85.89934592
    weights = 3.0
    ax.axhline(dev, color=C['danger'], lw=1.3, ls=':')
    ax.text(600, dev / 1.5, 'One 80 GB accelerator', color=C['danger'], fontsize=11, va='top')
    ax.plot(s, np.full_like(s, weights), color=C['memory'], lw=3.2)
    ax.text(600, weights * 1.4, 'Weights, 3 GB (fixed)', color=C['memory'], fontsize=12, fontweight='bold')
    if step >= 2:
        ax.plot(s, 0.0003072 * s, color=C['compute'], lw=3.2)
        x = 9765.625
        ax.plot([x], [weights], 'o', ms=9, color=C['compute'], mec='white', zorder=5)
        ax.text(x * 1.25, 0.06, 'KV cache passes\nthe weights near 10K', color=C['compute'], fontsize=11, fontweight='bold')
    if step >= 3:
        ax.plot(s, 5e-08 * s ** 2, color=C['danger'], lw=3.2, ls='--')
        x = 41448.60575
        ax.plot([x], [dev], 'o', ms=9, color=C['danger'], mec='white', zorder=5)
        ax.text(x / 1.9, dev * 2.2, "One layer's scores\nfill it near 40K", color=C['danger'], fontsize=11, fontweight='bold', ha='right')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(512, 131072)
    ax.set_ylim(0.01, 2000)
    ax.set_xticks([512, 2048, 8192, 32768, 131072])
    ax.set_xticklabels(['512', '2K', '8K', '32K', '128K'])
    ax.set_yticks([0.01, 1, 100])
    ax.set_yticklabels(['10 MB', '1 GB', '100 GB'])
    ax.minorticks_off()
    ax.set_xlabel('Context length (tokens, log scale)')
    ax.set_ylabel('Memory (log scale)')
    ax.grid(color=C['grid'], lw=0.6)
    save(fig, out / f'context_memory_step{step}.svg', dict(left=0.14, right=0.86, bottom=0.21, top=0.96))

def kv_per_token():
    rows = [('GPT-2 XL (1.5B)', 307.2, 'neutral', '307 KB: 25 own heads', None), ('Llama 2 7B', 524.288, 'danger', '524 KB: 32 own heads', None), ('Llama 3.1 8B', 131.072, 'memory', '131 KB: 8 shared', None), ('Llama 2 70B', 327.68, 'memory', '328 KB: 8 shared', 2621.44), ('PaLM 540B', 120.832, 'memory', f'121 KB: {1} shared', None)]
    fig, ax = plt.subplots(figsize=(5.8, 2.45))
    n = len(rows)
    for i, (label, v, ck, txt, ghost) in enumerate(rows):
        y = n - 1 - i
        if ghost:
            ax.barh(y, ghost - 30, left=30, color='white', edgecolor=C['danger'], ls='--', lw=1.4, height=0.62)
            ax.text(ghost * 1.1, y, '2.6 MB\nunshared', va='center', fontsize=10.5, color=C['danger'], fontweight='bold', linespacing=1.0)
        ax.barh(y, v - 30, left=30, color=C[ck], height=0.62)
        ax.text(27, y, label, ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'])
        if ghost:
            ax.text(36, y, txt, va='center', fontsize=11, fontweight='bold', color='white')
        else:
            ax.text(v * 1.1, y, txt, va='center', fontsize=11, fontweight='bold', color=C[ck])
    ax.set_xscale('log')
    ax.set_xlim(30, 9000)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([100, 1000])
    ax.set_xticklabels(['100 KB', '1 MB'])
    ax.minorticks_off()
    ax.set_xlabel('KV cache per token, FP16 (log scale)')
    ax.grid(axis='x', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    save(fig, out / 'kv_per_token.svg', dict(left=0.27, right=0.98, bottom=0.2, top=0.98))

def recsys_rate():
    hbm, pcie = (251652644.2, 4807692.308)
    fig, ax = plt.subplots(figsize=(2.6, 2.55))
    for x, v, ck, txt in [(0, hbm, 'memory', '252M'), (1, pcie, 'network', '4.8M')]:
        ax.bar(x, v, color=C[ck], width=0.62)
        ax.text(x, v * 1.4, txt, ha='center', va='bottom', fontsize=12.5, fontweight='bold', color=C[ck])
    ax.text(1, 32000000.0, '52× fewer', ha='center', fontsize=12.5, fontweight='bold', color=C['danger'])
    ax.set_yscale('log')
    ax.set_ylim(1000000.0, 3000000000.0)
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['HBM\n3.35 TB/s', 'PCIe Gen5\n64 GB/s'])
    ax.set_yticks([1000000.0, 10000000.0, 100000000.0, 1000000000.0])
    ax.set_yticklabels(['1M', '10M', '100M', '1B'])
    ax.minorticks_off()
    ax.set_ylabel('Requests/s, upper bound')
    ax.grid(axis='y', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    save(fig, out / 'recsys_rate.svg', dict(left=0.3, right=0.97, bottom=0.22, top=0.97))

def capacity():
    fig, ax = plt.subplots(figsize=(3.6, 1.85))
    dev = 80.0
    rows = [('One table', 51.2, 'memory', '51.2 GB'), ('Two tables', 102.4, 'danger', '102.4 GB')]
    for i, (label, v, ck, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=C[ck], height=0.6)
        ax.text(-2, y, label, ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'])
        if v < dev:
            ax.text(v - 2, y, txt, va='center', ha='right', fontsize=12, fontweight='bold', color='white')
        else:
            ax.text(v + 3, y, txt, va='center', fontsize=12, fontweight='bold', color=C[ck])
    ax.axvline(dev, color=C['danger'], lw=2, ls='--')
    ax.text(dev - 2, 1.62, '80 GB A100', ha='right', fontsize=11, color=C['danger'], fontweight='bold')
    ax.set_xlim(0, 150)
    ax.set_ylim(-0.5, 1.9)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_xlabel('GB (100M rows, 128-wide, FP32)')
    ax.grid(axis='x', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    save(fig, out / 'capacity.svg', dict(left=0.27, right=0.97, bottom=0.27, top=0.97))

def million_request(step):
    weights = 16.06
    kv = 131.072
    scores = 2000.0
    rows = [('Weights, BF16', weights, 'memory', '16.1 GB'), (f'KV cache, {k_label(1000000)} tokens', kv, 'compute', '131 GB'), ('Scores if stored', scores, 'danger', '2 TB per head, per layer')]
    fig, ax = plt.subplots(figsize=(5.8, 1.95))
    n = len(rows)
    for i, (label, v, ck, txt) in enumerate(rows):
        y = n - 1 - i
        shown = i < step
        ghost = i == 2
        if shown:
            if ghost:
                ax.barh(y, v - 1, left=1, color=C['danger_l'], edgecolor=C[ck], ls='--', lw=1.4, height=0.6)
            else:
                ax.barh(y, v - 1, left=1, color=C[ck], height=0.6)
            if ghost:
                ax.text(1.12, y, txt, va='center', ha='left', fontsize=11, fontweight='bold', color=C['danger'], zorder=6, bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
            elif v < 80.0:
                ax.text(v / 1.12, y, txt, va='center', ha='right', fontsize=12, fontweight='bold', color='white')
            else:
                ax.text(v * 1.15, y, txt, va='center', fontsize=12, fontweight='bold', color=C[ck])
        ax.text(0.92, y, label, ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'] if shown else 'white')
    ax.axvline(80.0, color=C['danger'], lw=2, ls=':')
    ax.text(80.0 * 1.1, n - 0.42, '80 GB device', ha='left', va='bottom', fontsize=10.5, color=C['danger'], fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlim(1, 8000)
    ax.set_ylim(-0.55, n - 0.05)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(['1 GB', '10 GB', '100 GB', '1 TB'])
    ax.minorticks_off()
    ax.set_xlabel('Memory for one request (log scale)')
    ax.grid(axis='x', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    save(fig, out / f'million_request_step{step}.svg', dict(left=0.3, right=0.97, bottom=0.25, top=0.88))

def fiction_livebench():
    row = {'model': 'chutes/DeepSeek-R1-0528', 'release_date': '2025-05-28', 'score_0': 1.0, 'score_400': 0.917, 'score_1k': 0.833, 'score_2k': 0.829, 'score_4k': 0.889, 'score_8k': 0.861, 'score_16k': 0.75, 'score_32k': 0.694, 'score_60k': 0.583, 'score_120k': None, 'score_192k': None}
    cols = ['score_400', 'score_1k', 'score_2k', 'score_4k', 'score_8k', 'score_16k', 'score_32k', 'score_60k']
    xs = [400.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 32000.0, 60000.0]
    ys = [row[c] for c in cols]
    fig, ax = plt.subplots(figsize=(2.9, 2.2))
    ax.plot(xs, ys, color=C['memory'], lw=3, marker='o', ms=6)
    ax.text(xs[0], ys[0] + 0.04, '0.917', ha='left', va='bottom', fontsize=11, fontweight='bold', color=C['memory'])
    ax.text(xs[-1], ys[-1] - 0.05, '0.583', ha='right', va='top', fontsize=11, fontweight='bold', color=C['danger'])
    ax.set_xscale('log')
    ax.set_xlim(300, 80000.0)
    ax.set_ylim(0.4, 1.05)
    ax.set_xticks([1000.0, 10000.0])
    ax.set_xticklabels(['1K', '10K'])
    ax.minorticks_off()
    ax.set_xlabel('Tokens of story in context')
    ax.set_ylabel('Score')
    ax.grid(color=C['grid'], lw=0.6)
    save(fig, out / 'fiction_livebench.svg', dict(left=0.22, right=0.95, bottom=0.22, top=0.96))

def flash_measured():
    rows = ({'seq_len': 128.0, 'pytorch_runtime_ms': 0.67, 'flashattention_runtime_ms': 0.31, 'pytorch_memory_mb': 36.0, 'flashattention_memory_mb': 22.0}, {'seq_len': 256.0, 'pytorch_runtime_ms': 0.7, 'flashattention_runtime_ms': 0.31, 'pytorch_memory_mb': 104.0, 'flashattention_memory_mb': 44.0}, {'seq_len': 512.0, 'pytorch_runtime_ms': 1.18, 'flashattention_runtime_ms': 0.73, 'pytorch_memory_mb': 336.0, 'flashattention_memory_mb': 104.0}, {'seq_len': 1024.0, 'pytorch_runtime_ms': 3.67, 'flashattention_runtime_ms': 2.29, 'pytorch_memory_mb': 1184.0, 'flashattention_memory_mb': 209.0}, {'seq_len': 2048.0, 'pytorch_runtime_ms': 13.22, 'flashattention_runtime_ms': 7.64, 'pytorch_memory_mb': 4416.0, 'flashattention_memory_mb': 418.0}, {'seq_len': 4096.0, 'pytorch_runtime_ms': 50.44, 'flashattention_runtime_ms': 30.09, 'pytorch_memory_mb': 17024.0, 'flashattention_memory_mb': 836.0}, {'seq_len': 8192.0, 'pytorch_runtime_ms': None, 'flashattention_runtime_ms': 118.5, 'pytorch_memory_mb': None, 'flashattention_memory_mb': 1672.0}, {'seq_len': 16384.0, 'pytorch_runtime_ms': None, 'flashattention_runtime_ms': 470.51, 'pytorch_memory_mb': None, 'flashattention_memory_mb': 3344.0}, {'seq_len': 32768.0, 'pytorch_runtime_ms': None, 'flashattention_runtime_ms': 1876.08, 'pytorch_memory_mb': None, 'flashattention_memory_mb': 6688.0}, {'seq_len': 65536.0, 'pytorch_runtime_ms': None, 'flashattention_runtime_ms': 7492.85, 'pytorch_memory_mb': None, 'flashattention_memory_mb': 13376.0})
    seq = [r['seq_len'] for r in rows]

    def col(name, scale=1):
        return [r[name] / scale for r in rows if r[name] is not None]

    def per_doubling(name):
        last, prev = (rows[-1][name], rows[-2][name])
        return f'×{last / prev:.0f}'
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 1.4))
    panels = [(axs[0], col('pytorch_runtime_ms'), col('flashattention_runtime_ms'), 'Runtime (ms)', f"FlashAttention:\n{per_doubling('flashattention_runtime_ms')} per doubling", (0.1, 30000.0), 0.45), (axs[1], col('pytorch_memory_mb', 1000.0), col('flashattention_memory_mb', 1000.0), 'Memory (GB)', f"FlashAttention:\n{per_doubling('flashattention_memory_mb')} per doubling", (0.01, 3000), 0.045)]
    for ax, pt, fa, ylab, lab, ylim, ylab_y in panels:
        ax.plot(seq[:len(pt)], pt, color=C['danger'], lw=2.8, marker='o', ms=4.5)
        ax.plot(seq, fa, color=C['memory'], lw=2.8, marker='o', ms=4.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(100, 90000.0)
        ax.set_ylim(*ylim)
        ax.set_xticks([128, 1024, 8192, 65536])
        ax.set_xticklabels(['128 tokens', '1K', '8K', '64K'])
        ax.minorticks_off()
        ax.set_ylabel(ylab)
        ax.grid(color=C['grid'], lw=0.6)
        ax.text(85000.0, ylab_y, lab, ha='right', va='center', fontsize=10.5, fontweight='bold', color=C['memory'], linespacing=1.0)
    axs[0].set_yticks([1, 100, 10000.0])
    axs[0].set_yticklabels(['1', '100', '10⁴'])
    axs[1].set_yticks([0.01, 1, 100])
    axs[1].set_yticklabels(['0.01', '1', '100'])
    axs[0].text(140, 2500.0, 'PyTorch:\nstores scores', color=C['danger'], fontsize=10.5, fontweight='bold', va='center', linespacing=1.0)
    axs[1].text(140, 300, f"PyTorch: out of\nmemory after {ki_label(seq[len(col('pytorch_memory_mb')) - 1])}", color=C['danger'], fontsize=10.5, fontweight='bold', va='center', linespacing=1.0)
    save(fig, out / 'flash_measured.svg', dict(left=0.12, right=0.985, bottom=0.17, top=0.97, wspace=0.42))
context_windows()
state_memory()
gpt2xl_params()
for k in (1, 2, 3, 4):
    block_macs(k)
quadratic()
decode_ridge()
for k in (1, 2, 3):
    context_memory(k)
kv_per_token()
recsys_rate()
capacity()
for k in (1, 2, 3):
    million_request(k)
fiction_livebench()
flash_measured()
print('figures written')
