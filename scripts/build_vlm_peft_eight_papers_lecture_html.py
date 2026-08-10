#!/usr/bin/env python3
"""Build a self-contained Chinese lecture for eight PEFT/VLM papers.

The active repository intentionally keeps generated artifacts out of source control.
The canonical PDFs, rendered pages, and five previously verified single-paper HTML
summaries live in the adjacent ``vlmintune copy`` material tree.  This builder reads
those sources, copies every referenced image into one namespaced asset directory,
adds new LoRA/QLoRA/L2T chapters, and emits one portable HTML file.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT.parent / "vlmintune copy"
DEFAULT_OUT = ROOT / "output/pdf/vlm_peft_eight_papers_lecture_html"


# Coordinates refer to the existing 850 x 1100 rendered pages.  Crops include the
# original caption whenever practical so a reader can audit our Chinese explanation
# against the paper itself.
MANUAL_CROPS: dict[str, list[tuple[str, int, tuple[int, int, int, int]]]] = {
    "lora": [
        ("lora__fig1_reparameterization.png", 1, (530, 705, 740, 935)),
        ("lora__table1_latency.png", 4, (55, 45, 810, 270)),
        ("lora__table2_glue.png", 6, (55, 45, 815, 405)),
        ("lora__table3_e2e.png", 7, (55, 40, 815, 315)),
        ("lora__table4_gpt3.png", 8, (55, 40, 815, 320)),
        ("lora__fig2_scaling.png", 8, (50, 545, 815, 1035)),
        ("lora__table5_target_modules.png", 10, (55, 185, 815, 475)),
        ("lora__table6_rank.png", 10, (55, 585, 815, 930)),
        ("lora__fig3_subspace.png", 11, (50, 430, 815, 840)),
        ("lora__fig4_table7_directions.png", 12, (45, 40, 815, 850)),
    ],
    "qlora": [
        ("qlora__fig1_memory_paths.png", 3, (45, 45, 815, 330)),
        ("qlora__fig2_3_ablations.png", 6, (420, 170, 815, 950)),
        ("qlora__table3_precision.png", 7, (45, 45, 815, 340)),
        ("qlora__table4_mmlu.png", 8, (45, 35, 815, 280)),
        ("qlora__table6_vicuna.png", 10, (45, 35, 815, 605)),
        ("qlora__table7_elo.png", 11, (45, 35, 815, 360)),
        ("qlora__fig4_rank.png", 22, (45, 360, 815, 815)),
        ("qlora__table10_loss_scope.png", 24, (125, 75, 725, 240)),
        ("qlora__table11_data_quality.png", 24, (115, 835, 740, 1035)),
        ("qlora__fig6_memory_breakdown.png", 26, (125, 75, 725, 495)),
    ],
    "l2t": [
        ("l2t__fig1_benchmarks.png", 1, (435, 560, 815, 1035)),
        ("l2t__fig2_shortcut.png", 2, (510, 180, 805, 620)),
        ("l2t__fig3_4_objective.png", 3, (50, 45, 815, 590)),
        ("l2t__fig5_7_visual_evidence.png", 4, (45, 35, 815, 550)),
        ("l2t__table1_main.png", 6, (45, 35, 815, 690)),
        ("l2t__table2_hallucination.png", 7, (45, 35, 815, 315)),
        ("l2t__fig8_9_scaling.png", 8, (45, 35, 815, 430)),
        ("l2t__table4_template_removal.png", 8, (45, 410, 815, 805)),
        ("l2t__table6_fig10_generalization_cost.png", 9, (45, 35, 815, 620)),
        ("l2t__table7_self_improvement.png", 10, (45, 35, 815, 225)),
        ("l2t__fig11_cases.png", 19, (45, 35, 815, 700)),
        ("l2t__table11_compute.png", 19, (45, 690, 815, 1010)),
    ],
}


IMPORTED_PAPERS = {
    "dora": "output/pdf/paper_2402_09353_html/paper_2402_09353_summary.html",
    "mole": "output/pdf/llava_mole_html/llava_mole_summary.html",
    "reft": "output/pdf/reft_representation_finetuning_html/reft_representation_finetuning_summary.html",
    "mores": "output/pdf/mores_llava_steering_html/mores_llava_steering_summary.html",
    "vl_adapter": "output/pdf/paper_2112_06825_html/paper_2112_06825_summary.html",
}


OFFICIAL_LINKS = {
    "lora": "https://openreview.net/forum?id=nZeVKeeFYf9",
    "qlora": "https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html",
    "dora": "https://proceedings.mlr.press/v235/liu24bn.html",
    "mole": "https://arxiv.org/abs/2401.16160",
    "reft": "https://proceedings.neurips.cc/paper_files/paper/2024/hash/75008a0fba53bf13b0bb3b7bff986e0e-Abstract-Conference.html",
    "mores": "https://arxiv.org/abs/2412.12359",
    "vl_adapter": "https://openaccess.thecvf.com/content/CVPR2022/html/Sung_VL-Adapter_Parameter-Efficient_Transfer_Learning_for_Vision-and-Language_Tasks_CVPR_2022_paper.html",
    "l2t": "https://arxiv.org/abs/2503.22215",
}


PAPER_ORDER = [
    ("权重空间与系统配方", "lora", "LoRA"),
    ("权重空间与系统配方", "qlora", "QLoRA"),
    ("权重空间与系统配方", "dora", "DoRA"),
    ("权重空间与系统配方", "mole", "LLaVA-MoLE"),
    ("表示空间", "reft", "ReFT / LoReFT"),
    ("表示空间", "mores", "MoReS / LLaVA Steering"),
    ("插入模块与监督目标", "vl_adapter", "VL-Adapter"),
    ("插入模块与监督目标", "l2t", "L2T / Learning to Instruct"),
]


# These are rendered to static SVG with Graphviz during the build.  Static vector
# diagrams keep the lecture fully readable offline and, unlike client-side Mermaid,
# are already present when Chromium lays out the A4 print document.
ARCHITECTURE_DOT_BODIES = {
    "lora": r'''
    tokens [label="输入 token / hidden x", fillcolor="#e6f0f7"];
    rest [label="其余 Transformer 计算\n冻结", fillcolor="#e8edf3"];
    x [label="目标 Linear 的输入 x", fillcolor="#e6f0f7"];
    base [label="基座分支 W₀x\nW₀ 冻结", fillcolor="#e8edf3"];
    down [label="降维 u=Ax\nA: r×k 可训练", fillcolor="#e4f3e8"];
    up [label="升维 v=Bu\nB: d×r 可训练", fillcolor="#e4f3e8"];
    scale [label="δ=(α/r)v", fillcolor="#e4f3e8"];
    add [label="相加 y=W₀x+δ", fillcolor="#fff0c9"];
    next [label="下一子层 / task loss", fillcolor="#e8edf3"];
    merge [label="MERGE\nW*=W₀+(α/r)BA", fillcolor="#eee7f7"];
    infer [label="推理只算 y=W*x", fillcolor="#e8edf3"];

    tokens -> rest -> x;
    x -> base -> add;
    x -> down -> up -> scale -> add;
    add -> next;
    base -> merge [style=dashed, label="训练后"];
    down -> merge [style=dashed];
    up -> merge [style=dashed];
    merge -> infer;
    ''',
    "qlora": r'''
    w0 [label="高精度预训练 W₀", fillcolor="#e8edf3"];
    quant [label="按块 NF4 量化\n得到 4-bit codes", fillcolor="#fff0c9"];
    dq [label="Double Quantization\n再量化第一层 scales", fillcolor="#fff0c9"];
    store [label="常驻：codes + scale 元数据\n全部冻结", fillcolor="#e8edf3"];
    x [label="BF16 hidden x", fillcolor="#e6f0f7"];
    deq [label="按需反量化到 BF16\n不长期复制整份权重", fillcolor="#fff0c9"];
    base [label="基座分支 Ŵx\n4-bit 存储 / BF16 计算", fillcolor="#e8edf3"];
    a [label="BF16 LoRA A\n可训练", fillcolor="#e4f3e8"];
    b [label="BF16 LoRA B\n可训练", fillcolor="#e4f3e8"];
    low [label="δ=(α/r)B(Ax)", fillcolor="#e4f3e8"];
    add [label="y=Ŵx+δ", fillcolor="#fff0c9"];
    loss [label="任务 CE", fillcolor="#eee7f7"];
    paged [label="Paged optimizer\n峰值时 page LoRA 状态", fillcolor="#eee7f7"];
    keep [label="常见推理：保留\nNF4 基座 + LoRA 支路", fillcolor="#e8edf3", peripheries=2];

    w0 -> quant -> dq -> store;
    store -> deq -> base;
    x -> base;
    x -> a -> b -> low;
    base -> add;
    low -> add;
    add -> loss;
    loss -> paged [style=dashed];
    paged -> a [style=dashed];
    paged -> b [style=dashed];
    store -> keep [style=dashed, label="训练后"];
    a -> keep [style=dashed];
    b -> keep [style=dashed];
    ''',
    "dora": r'''
    w0 [label="预训练 W₀\n冻结", fillcolor="#e8edf3"];
    init [label="初始化 m₀=逐列 ||W₀||₂\nB=0", fillcolor="#fff0c9"];
    x [label="输入 x", fillcolor="#e6f0f7"];
    a [label="A 可训练", fillcolor="#e4f3e8"];
    b [label="B 可训练", fillcolor="#e4f3e8"];
    cand [label="方向候选 V~=W₀+(α/r)BA", fillcolor="#e4f3e8"];
    norm [label="逐列归一化\nU=V~/||V~||c", fillcolor="#fff0c9"];
    m [label="逐列 magnitude m\n可训练", fillcolor="#e4f3e8"];
    weff [label="W_eff=U·diag(m)", fillcolor="#fff0c9"];
    y [label="输出 y=W_eff x", fillcolor="#e6f0f7"];
    detach [label="主实验：范数分母\n仅 backward detach", fillcolor="#eee7f7"];
    merge [label="MERGE 为静态 W*\n推理无额外分支", fillcolor="#e8edf3"];

    w0 -> init -> m;
    w0 -> cand;
    a -> cand;
    b -> cand;
    cand -> norm -> weff;
    m -> weff;
    x -> y;
    weff -> y;
    norm -> detach [style=dashed];
    weff -> merge [style=dashed, label="训练后"];
    ''',
    "mole": r'''
    img [label="图像", fillcolor="#e6f0f7"];
    ve [label="CLIP ViT-L\n冻结", fillcolor="#e8edf3"];
    proj [label="2-layer MLP projector\nSFT 状态原文未单列", fillcolor="#fff0c9"];
    text [label="问题 / token embedding", fillcolor="#e6f0f7"];
    mix [label="视觉 + 文本 token 序列", fillcolor="#e6f0f7"];
    sa [label="Self-Attention\n基座冻结 + 普通 LoRA 可训练", fillcolor="#e4f3e8"];
    x [label="某个 token 的 hidden x", fillcolor="#e6f0f7"];
    ffn [label="原 FFN 输出 f(x)\n冻结", fillcolor="#e8edf3"];
    router [label="同一 FFN 共享 Router G(x)\n可训练", fillcolor="#e4f3e8"];
    top1 [label="每 token 选 top-1", shape=diamond, fillcolor="#fff0c9"];
    expert [label="仅执行命中的 LoRA expert\nA_k,B_k 可训练", fillcolor="#e4f3e8", peripheries=2];
    add [label="h=f(x)+E_k(x)", fillcolor="#fff0c9"];
    next [label="下一 Transformer 层 / LM head", fillcolor="#e8edf3"];
    lm [label="L_LM", fillcolor="#eee7f7"];
    lb [label="负载均衡 L_lb\n仅训练", fillcolor="#eee7f7"];
    total [label="L=L_LM+10⁻² mean(L_lb)", fillcolor="#eee7f7"];

    img -> ve -> proj -> mix;
    text -> mix;
    mix -> sa -> x;
    x -> ffn -> add;
    x -> router -> top1 -> expert -> add;
    add -> next -> lm -> total;
    router -> lb [style=dashed];
    lb -> total [style=dashed];
    ''',
    "reft": r'''
    prompt [label="Prompt：n 个 token", fillcolor="#e6f0f7"];
    emb [label="Embedding + Blocks 0…l\n冻结", fillcolor="#e8edf3"];
    h [label="第 l 层 hidden h_p", fillcolor="#e6f0f7"];
    mask [label="p 是否属于\nReFT position mask P？", shape=diamond, fillcolor="#fff0c9"];
    same [label="否：h'=h", fillcolor="#e8edf3"];
    rh [label="Rh\n读取当前低维坐标", fillcolor="#e4f3e8"];
    target [label="Wh+b\n产生目标低维坐标", fillcolor="#e4f3e8"];
    delta [label="低维差 (Wh+b-Rh)", fillcolor="#fff0c9"];
    lift [label="Rᵀ(Wh+b-Rh)\nR,W,b 可训练", fillcolor="#e4f3e8", peripheries=2];
    update [label="是：h'=h+Rᵀ(Wh+b-Rh)", fillcolor="#fff0c9"];
    scatter [label="写回该层序列", fillcolor="#fff0c9"];
    post [label="Blocks l+1…L + LM head\n冻结", fillcolor="#e8edf3"];
    out [label="Answer logits / generation", fillcolor="#e6f0f7"];
    loss [label="answer / output CE", fillcolor="#eee7f7"];

    prompt -> emb -> h -> mask;
    mask -> same [label="否"];
    mask -> rh [label="是"];
    mask -> target [label="是"];
    rh -> delta;
    target -> delta;
    delta -> lift -> update;
    same -> scatter;
    update -> scatter;
    scatter -> post -> out -> loss;
    ''',
    "mores": r'''
    img [label="图像", fillcolor="#e6f0f7"];
    ve [label="Vision encoder\nFigure 3 中冻结", fillcolor="#e8edf3"];
    conn [label="Connector\nrecipe-dependent / 图中可训练", fillcolor="#e4f3e8"];
    vt [label="视觉 tokens", fillcolor="#e6f0f7"];
    text [label="问题 / 文本 tokens", fillcolor="#e6f0f7"];
    mix [label="合并多模态序列", fillcolor="#e6f0f7"];
    block [label="冻结 LLM Block l", fillcolor="#e8edf3"];
    choose [label="视觉位置 ∩ 选中 ρ%？", shape=diamond, fillcolor="#fff0c9"];
    same [label="文本或未选视觉 token\nh'=h", fillcolor="#e8edf3"];
    down [label="W_down h\n当前低维坐标", fillcolor="#e4f3e8"];
    linear [label="Linear(h)\n目标低维坐标", fillcolor="#e4f3e8"];
    phi [label="φ(h)=Linear(h)-W_down h", fillcolor="#fff0c9"];
    up [label="Δh=W_up φ(h)\nMoReS 参数可训练", fillcolor="#e4f3e8", peripheries=2];
    residual [label="h'=h+Δh", fillcolor="#fff0c9"];
    next [label="下一冻结 LLM Block", fillcolor="#e8edf3"];
    head [label="冻结 LM head → answer", fillcolor="#e8edf3"];

    img -> ve -> conn -> vt -> mix;
    text -> mix;
    mix -> block -> choose;
    choose -> same [label="否"];
    choose -> down [label="是"];
    choose -> linear [label="是"];
    down -> phi;
    linear -> phi;
    phi -> up -> residual;
    same -> next;
    residual -> next -> head;
    ''',
    "vl_adapter": r'''
    img [label="图像 / 视频帧", fillcolor="#e6f0f7"];
    clip [label="CLIP visual encoder\n冻结", fillcolor="#e8edf3"];
    proj [label="Visual projection\n可训练", fillcolor="#e4f3e8"];
    text [label="Task prompt + text", fillcolor="#e6f0f7"];
    emb [label="BART/T5 embedding\n冻结", fillcolor="#e8edf3"];
    mix [label="视觉与文本表示", fillcolor="#e6f0f7"];
    enc [label="Encoder 主权重冻结\nSelf-Attn / FFN 后插 Adapter", fillcolor="#e8edf3"];
    dec [label="Decoder 主权重冻结\nSelf/Cross-Attn / FFN 后插 Adapter", fillcolor="#e8edf3"];
    x [label="子层输出 x", fillcolor="#e6f0f7"];
    down [label="Down projection", fillcolor="#e4f3e8"];
    gelu [label="GELU", fillcolor="#e4f3e8"];
    up [label="Up projection", fillcolor="#e4f3e8"];
    res [label="h=x+Up(GELU(Down(x)))", fillcolor="#fff0c9", peripheries=2];
    ln [label="所有 LayerNorm\n可训练", fillcolor="#e4f3e8"];
    head [label="Tied output head\n冻结", fillcolor="#e8edf3"];
    ce [label="token-level CE", fillcolor="#eee7f7"];

    img -> clip -> proj -> mix;
    text -> emb -> mix;
    mix -> enc -> dec -> head -> ce;
    enc -> x [style=dashed, label="任一插入点"];
    x -> down -> gelu -> up -> res;
    res -> dec [style=dashed];
    ln -> enc [style=dashed];
    ln -> dec [style=dashed];
    ''',
    "l2t": r'''
    img [label="图像 → visual tokens", fillcolor="#e6f0f7"];
    inst [label="完整 user instruction", fillcolor="#e6f0f7"];
    ans [label="ground-truth response", fillcolor="#e6f0f7"];
    seq [label="[visual | instruction | response]", fillcolor="#e6f0f7"];
    llm [label="原 VIT 的 causal MLLM\nforward 结构不变", fillcolor="#e8edf3"];
    logits [label="每个位置的 next-token logits", fillcolor="#e6f0f7"];
    filter [label="无参数 supervision filter", fillcolor="#fff0c9"];
    zero [label="system / role / 高频模板\nmask=0", fillcolor="#fff0c9"];
    onei [label="有视觉语义 instruction\nmask=1", fillcolor="#fff0c9"];
    onea [label="response\nmask=1", fillcolor="#fff0c9"];
    mask [label="L2T loss mask", fillcolor="#fff0c9"];
    ce [label="只在 mask=1 计算 CE\nL=L_instruction+L_response", fillcolor="#eee7f7"];
    train [label="按原 recipe 更新 VE / connector / LLM\nL2T 自身零新增参数", fillcolor="#e4f3e8"];
    infer [label="推理仍是：给 instruction → 生成 response\n无 filter、无额外模块", fillcolor="#e8edf3"];

    img -> seq;
    inst -> seq;
    ans -> seq;
    seq -> llm -> logits -> ce;
    inst -> filter;
    filter -> zero -> mask;
    filter -> onei -> mask;
    ans -> onea -> mask;
    mask -> ce;
    ce -> train [style=dashed];
    llm -> infer [style=dashed, label="部署"];
    ''',
}


ARCHITECTURE_GUIDE = {
    "lora": {
        "caption": "LoRA：转换发生在选定 Linear 的并行低秩增量支路；绿色 A/B 是唯一默认可训练权重。",
        "look": "从 <code>x</code> 同时沿灰色 <code>W₀x</code> 和绿色 <code>A→B</code> 两条支路走，二者在黄色加法节点汇合。原论文多数实验只选 self-attention 的 q/v。",
        "meaning": "forward 改的是线性映射的任务增量 <code>ΔW=(α/r)BA</code>，不是把 <code>W₀</code> 压成低秩。训练前 <code>B=0</code>，故初始函数不变。",
        "runtime": "训练只更新 A/B；静态部署可把增量并入 <code>W*</code>。只有 merge 后才是一次普通矩阵乘和零额外分支。",
    },
    "qlora": {
        "caption": "QLoRA：LoRA 转换位置不变，新增的是冻结底座的 NF4 存储、反量化计算和分页优化器路径。",
        "look": "左上是一次性 NF4 与 Double Quantization；中间是每次前向按需反量化到 BF16；右侧绿色仍是普通 LoRA。",
        "meaning": "4-bit 描述的是冻结权重的存储，不是让梯度、LoRA 参数和矩阵乘都变成 4-bit；Double Quantization 量化的是第一层 scale。",
        "runtime": "优化器只维护 LoRA 状态。常见推理保留 4-bit 基座和 adapter；反量化、合并、再量化并不是普通 LoRA 的无损 merge。",
    },
    "dora": {
        "caption": "DoRA：在目标 Linear 内把逐列 magnitude 与归一化 direction 分开学习，再形成可合并的有效权重。",
        "look": "先看 <code>W₀+(α/r)BA</code> 的方向候选，再逐列归一化，最后乘上独立 magnitude <code>m</code>。",
        "meaning": "A/B 只直接限制方向候选的低秩变化；归一化和独立 m 使最终 <code>W_eff−W₀</code> 不必仍是低秩。",
        "runtime": "训练更新 A/B/m，并在最终论文实验中对范数分母做 backward detach；部署可一次性计算 W*，移除归一化分支。",
    },
    "mole": {
        "caption": "LLaVA-MoLE：Self-Attention 使用普通 LoRA，只有 FFN 的 LoRA 分支按 token 由共享 router 选择 top-1 专家。",
        "look": "同一 token 一边进入冻结 FFN，一边进入 router；黄色菱形只激活一个 LoRA expert，结果再加回原 FFN 输出。",
        "meaning": "专家不是完整 FFN 副本，也不是按数据集 ID 固定选择。同一 FFN 内的线性层共享 router，而不同 token 可去不同专家。",
        "runtime": "训练有语言建模损失与负载均衡损失；推理仍需 router，但若实现真正跳过未命中专家，每 token 只计算一组 LoRA 增量。",
    },
    "reft": {
        "caption": "ReFT / LoReFT：转换发生在指定 Transformer 层、指定 prompt 位置的 residual representation，而不是权重矩阵。",
        "look": "先用 position mask 决定 token 是否进入干预；命中后读取 <code>Rh</code>、生成 <code>Wh+b</code>，把低维差经 <code>Rᵀ</code> 残差写回。",
        "meaning": "intervention mask 选择 forward 中被改的 prompt token；answer loss mask 选择产生 CE 的输出 token，两者不是同一种 mask。",
        "runtime": "LM 权重冻结，只更新 R/W/b。推理必须保留 intervention，生成任务通常在 prompt prefill 的 prefix/suffix 位置执行。",
    },
    "mores": {
        "caption": "MoReS：在每个冻结 LLM 层后，仅对选中的视觉 token 做 down–difference–up 表示转向。",
        "look": "黄色 selector 先取视觉位置与 ρ 子集的交集；命中的 token 计算低维目标与当前坐标之差，升维成 <code>Δh</code> 后残差相加。",
        "meaning": "文本 token 与未选视觉 token 完全走 identity。论文主实验约选择 1% 视觉 token；标题参数口径只统计 LLM 内 TP*。",
        "runtime": "整个 LLM 冻结，MoReS 参数可训练且推理时保留；视觉编码器/connector 的冻结状态依具体 recipe，不能由 TP* 反推。",
    },
    "vl_adapter": {
        "caption": "VL-Adapter：在 encoder/decoder 的 Attention 与 FFN 后插入非线性瓶颈残差模块。",
        "look": "主干 BART/T5 与 CLIP 是灰色；每个插入点把子层输出 x 经 Down→GELU→Up 后残差加回。LayerNorm 与视觉投影也训练。",
        "meaning": "Adapter 是显式新增模块，不是 LoRA 权重增量；Single Adapter 表示跨任务共享，不表示全网只有一个实例。",
        "runtime": "训练更新 adapters、LayerNorm 和 visual projection；推理必须执行 Down/GELU/Up，不能静态 merge 回原主干。",
    },
    "l2t": {
        "caption": "L2T：模型 forward 完全不变，转换发生在监督标签构造——哪些 token 的 next-token logits 进入损失。",
        "look": "visual/instruction/response 仍进入同一 causal MLLM；无参数 filter 把 system、role、高频模板标 0，把有意义 instruction 与 answer 标 1。",
        "meaning": "mask=0 的 token 仍留在上下文，只是不贡献 CE。L2T 扩展监督 token，不是在推理时先生成问题。",
        "runtime": "训练时 loss 变为 instruction NLL + response NLL；推理没有 filter 或新模块，仍由用户给 instruction、模型生成 response。",
    },
}


WORKED_EXAMPLES = {
    "lora": r'''
<h3>完整算例：3→2 的 Linear，rank 1</h3>
<p class="example-note">统一用列向量。设 <code>x∈R³</code>、<code>W₀∈R²ˣ³</code>、<code>A∈R¹ˣ³</code>、<code>B∈R²ˣ¹</code>，并取 <code>r=1, α=2</code>，所以缩放 <code>s=α/r=2</code>。</p>
<div class="calc-eq"><b>给定</b><br>x=[1, 2, −1]ᵀ<br>W₀=[[1,0,2], [−1,1,0]]<br>A=[1,−1,2]，B=[1,−0.5]ᵀ</div>
<ol class="calc-steps">
  <li><b>冻结支路：</b><code>W₀x=[1−2, −1+2]ᵀ=[−1,1]ᵀ</code>。</li>
  <li><b>降维：</b><code>Ax=1−2−2=−3</code>，结果是一个标量。</li>
  <li><b>升维并缩放：</b><code>B(Ax)=[−3,1.5]ᵀ</code>，所以 <code>δ=2B(Ax)=[−6,3]ᵀ</code>。</li>
  <li><b>双支路输出：</b><code>y=W₀x+δ=[−7,4]ᵀ</code>。</li>
  <li><b>merge 校验：</b><code>ΔW=2BA=[[2,−2,4], [−1,1,−2]]</code>；<code>W*=W₀+ΔW=[[3,−2,6], [−2,2,−2]]</code>；直接算 <code>W*x=[−7,4]ᵀ</code>，与双支路严格相同。</li>
</ol>
<p class="example-check"><strong>完整性检查：</strong>输入维度 3，经 A 变成 1，经 B 回到 2，再与 W₀ 的二维输出相加；没有把 <code>AB</code> 与 <code>BA</code> 写反。</p>
''',
    "qlora": r'''
<h3>完整算例：两块 NF4 + Double Quantization + LoRA</h3>
<p class="example-note">为可手算，每两个权重为一块；论文真实权重 block size 为 64，scale 的第二层 block size 为 256。</p>
<div class="calc-eq"><b>高精度冻结权重</b><br>W₀=[[2.0,0.9], [1.0,−1.1]]；按 row-major 得 b₁=(2.0,0.9)、b₂=(1.0,−1.1)。</div>
<ol class="calc-steps">
  <li><b>块 1：</b><code>c₁=absmax=2</code>，归一化为 <code>(1,0.45)</code>；NF4 最近值为 <code>(1,0.440709829)</code>，故反量化 <code>b̂₁=(2,0.881419658)</code>。</li>
  <li><b>块 2：</b><code>c₂=1.1</code>，归一化 <code>(0.90909,−1)</code>，最近 NF4 值为 <code>(1,−1)</code>，故 <code>b̂₂=(1.1,−1.1)</code>。于是 <code>Ŵ=[[2,0.881419658],[1.1,−1.1]]</code>。</li>
  <li><b>Double Quantization：</b>scales <code>c=(2,1.1)</code>，均值 <code>μ=1.55</code>；中心化为 <code>(0.45,−0.45)</code>。取二级 scale <code>0.45</code> 与 codes <code>(1,−1)</code>，恢复 <code>ĉ=1.55+0.45(1,−1)=(2,1.1)</code>。</li>
  <li><b>BF16 基座前向：</b>取 <code>x=[2,1]ᵀ</code>，忽略手算舍入，<code>Ŵx=[4.881419658,1.1]ᵀ</code>；原高精度 <code>W₀x=[4.9,0.9]ᵀ</code>。</li>
  <li><b>LoRA：</b>取 <code>r=α=1</code>、<code>A=[1,−1]</code>、<code>B=[0.018580342,−0.2]ᵀ</code>。<code>Ax=1</code>，所以 <code>B(Ax)=B</code>。</li>
  <li><b>最终输出：</b><code>y=Ŵx+B(Ax)=[4.9,0.9]ᵀ</code>。本例特意让 rank-1 分支补偿该输入上的量化误差，不代表它能对所有输入无损恢复 W₀。</li>
</ol>
<p class="example-check"><strong>关键口径：</strong>NF4 codes、两层 scale 元数据均冻结；4-bit 是存储，前向计算与 LoRA 参数仍使用较高精度。</p>
''',
    "dora": r'''
<h3>完整算例：2×2 权重的逐列 magnitude / direction</h3>
<div class="calc-eq"><b>给定</b><br>W₀=[[3,0],[4,2]]；两列范数 m₀=[5,2]ᵀ。<br>r=α=1，训练后 A=[1,0]，B=[1,0]ᵀ，m=[6,1]ᵀ。</div>
<ol class="calc-steps">
  <li><b>rank-1 方向增量：</b><code>BA=[[1,0],[0,0]]</code>。</li>
  <li><b>方向候选：</b><code>V~=W₀+BA=[[4,0],[4,2]]</code>。</li>
  <li><b>逐列归一化：</b>列范数 <code>c₁=4√2</code>、<code>c₂=2</code>，所以 <code>U=[[1/√2,0],[1/√2,1]]</code>。</li>
  <li><b>应用新 magnitude：</b><code>W_eff=U·diag(6,1)=[[3√2,0],[3√2,1]]</code>。</li>
  <li><b>完整前向：</b>取 <code>x=[1,2]ᵀ</code>，<code>y=W_eff x=[3√2,3√2+2]ᵀ≈[4.242641,6.242641]ᵀ</code>。</li>
  <li><b>初始化校验：</b>若 <code>B=0,m=m₀</code>，则 <code>normalize(W₀)·diag(5,2)=W₀</code>，初始模型不变。</li>
</ol>
<p class="example-check"><strong>一个不直观的结论：</strong><code>W_eff−W₀=[[3√2−3,0],[3√2−4,−1]]</code> 的行列式为 <code>−(3√2−3)≠0</code>，所以最终更新 rank=2；低秩约束只直接施加在方向候选 BA。</p>
''',
    "mole": r'''
<h3>完整算例：一个 token、3 个专家、top-1</h3>
<p class="example-note">真实 FFN 是多层 SwiGLU；这里压缩为二维冻结线性层，以便完整展示 router、专家增量与 next-token 输出。</p>
<div class="calc-eq"><b>输入与 router</b><br>x=[2,−1]ᵀ；Wᵍ=[[1,0],[0,1],[0.5,−0.5]]。<br>G(x)=Wᵍx=[2,−1,1.5]ᵀ，softmax≈[0.6038,0.0301,0.3661]。</div>
<ol class="calc-steps">
  <li><b>路由：</b><code>argmax G(x)=expert 1</code>；expert 2/3 对这个 token 完全跳过。</li>
  <li><b>冻结 FFN：</b>设 <code>W=I₂</code>，则 <code>f(x)=Wx=[2,−1]ᵀ</code>。</li>
  <li><b>expert 1：</b>取 <code>r=1, α/r=1</code>，<code>A₁=[1,1]</code>、<code>B₁=[0.2,−0.4]ᵀ</code>。<code>A₁x=1</code>，故 <code>E₁(x)=B₁A₁x=[0.2,−0.4]ᵀ</code>。</li>
  <li><b>FFN 最终输出：</b><code>h=f(x)+E₁(x)=[2.2,−1.4]ᵀ</code>。</li>
  <li><b>冻结 LM head：</b>设 <code>W_vocab=[[1,0],[0,1],[−1,0]]</code>，则 logits <code>z=[2.2,−1.4,−2.2]</code>，softmax≈<code>[0.9619,0.0263,0.0118]</code>，因此选择第一个词元。</li>
</ol>
<p class="example-check"><strong>另一个 token 会独立路由：</strong>top-1 是 token 粒度；同一 batch、同一样本中的不同位置不必选择同一专家。</p>
''',
    "reft": r'''
<h3>完整算例：D=3、低维 r=2 的 LoReFT</h3>
<p class="example-note">先区分两个 mask：prompt 为 <code>[x₁ x₂ x₃ x₄]</code>，若 prefix=1、suffix=1，则 intervention mask 是 <code>[1,0,0,1 | 0(answer)]</code>；answer loss mask 则是 <code>[0,0,0,0 | 1(answer)]</code>。</p>
<div class="calc-eq"><b>给定被选中的 token</b><br>h=[2,3,4]ᵀ；R=[[1,0,0],[0,1,0]]；W=[[0,1,0],[0,0,1]]；b=[1,−2]ᵀ。<br>R,W∈R²ˣ³，且 RRᵀ=I₂。</div>
<ol class="calc-steps">
  <li><b>读取当前低维坐标：</b><code>Rh=[2,3]ᵀ</code>。</li>
  <li><b>产生目标低维坐标：</b><code>Wh+b=[3,4]ᵀ+[1,−2]ᵀ=[4,2]ᵀ</code>。</li>
  <li><b>低维差：</b><code>Wh+b−Rh=[2,−1]ᵀ</code>。</li>
  <li><b>升回三维：</b><code>Rᵀ[2,−1]ᵀ=[2,−1,0]ᵀ</code>。</li>
  <li><b>残差写回：</b><code>h'=h+[2,−1,0]ᵀ=[4,2,4]ᵀ</code>。</li>
  <li><b>验证：</b><code>Rh'=[4,2]ᵀ=Wh+b</code>；R 行空间外的第三维仍为 4。未被 position mask 选中的 token 直接保持 <code>h'=h</code>。</li>
</ol>
<p class="example-check"><strong>实现口径：</strong>论文公式用列向量。若代码批量 hidden 写成行向量 H，则是 <code>H'=H+(HWᵀ+1bᵀ−HRᵀ)R</code>。</p>
''',
    "mores": r'''
<h3>完整算例：只转向一个被选视觉 token</h3>
<p class="example-note">取高维 <code>D=3</code>、低维 <code>d=2</code>。为避开原论文约束式的维度歧义，本例采用可检验的 tied 口径 <code>W_up=W_downᵀ</code>、<code>W_down W_up=I₂</code>。</p>
<div class="calc-eq"><b>给定</b><br>h=[2,3,4]ᵀ<br>W_down=[[1,0,0],[0,1,0]]，W_up=[[1,0],[0,1],[0,0]]<br>Linear 矩阵 L=[[0,1,0],[0,0,1]]</div>
<ol class="calc-steps">
  <li><b>当前低维坐标：</b><code>W_down h=[2,3]ᵀ</code>。</li>
  <li><b>目标低维坐标：</b><code>Lh=[3,4]ᵀ</code>。</li>
  <li><b>低维 steering 差：</b><code>φ(h)=Lh−W_down h=[1,1]ᵀ</code>。</li>
  <li><b>升维更新量：</b><code>Δh=W_up φ(h)=[1,1,0]ᵀ</code>。</li>
  <li><b>残差结果：</b><code>h'=h+Δh=[3,4,4]ᵀ</code>。若它不是视觉 token 或未被 ρ-subset 选中，则仍为 <code>[2,3,4]ᵀ</code>。</li>
  <li><b>不能全局双射：</b><code>W_up W_down=diag(1,1,0)≠I₃</code>；当 d&lt;D 时只能重建选定子空间。</li>
</ol>
<p class="example-check"><strong>原文校勘：</strong>PDF 写 <code>W_down W_upᵀ=I_D</code>，按其给出的形状无法相乘。这里不偷偷把错误公式当真，而明确采用维度成立的教学口径。</p>
''',
    "vl_adapter": r'''
<h3>完整算例：3→2→3 的 GELU Adapter</h3>
<p class="example-note">采用列向量，并把该步 LayerNorm 简化为恒等映射。真实论文在 encoder/decoder 多个子层后重复同样模块。</p>
<div class="calc-eq"><b>给定</b><br>x=[1,2,−1]ᵀ<br>W_D=[[1,0,1],[0,1,1]]<br>W_U=[[1,0],[0,2],[1,−1]]</div>
<ol class="calc-steps">
  <li><b>Down：</b><code>u=W_Dx=[1−1,2−1]ᵀ=[0,1]ᵀ</code>。</li>
  <li><b>GELU：</b><code>GELU(0)=0</code>、<code>GELU(1)≈0.8413</code>，故 <code>v=[0,0.8413]ᵀ</code>。</li>
  <li><b>Up：</b><code>Δx=W_Uv=[0,1.6826,−0.8413]ᵀ</code>。</li>
  <li><b>Residual：</b><code>h=x+Δx=[1,3.6826,−1.8413]ᵀ</code>。</li>
  <li><b>冻结 head：</b>设 <code>W_head=[[1,0,0],[0,0.5,−0.5]]</code>，则 logits <code>z=[1,2.7620]</code>，softmax≈<code>[0.1465,0.8535]</code>。</li>
  <li><b>单 token CE：</b>若第二个词元为目标，<code>L=−ln(0.8535)≈0.1584</code>。</li>
</ol>
<p class="example-check"><strong>维度链：</strong>3 维 x 经 Down 到 2 维，经 GELU 后由 Up 回 3 维，才能和 residual x 相加；这个模块在推理时仍需执行。</p>
''',
    "l2t": r'''
<h3>完整算例：同一序列的两种 loss mask</h3>
<p class="example-note">每个概率都是 causal probability；mask=0 的 token 仍是上下文条件，只是不贡献损失。</p>
<div class="table-wrap"><table class="calc-table">
<thead><tr><th>位置</th><th>token</th><th>含义</th><th>VIT</th><th>L2T</th><th>正确词概率</th></tr></thead>
<tbody>
<tr><td>0</td><td><code>&lt;image&gt;</code></td><td>视觉前缀</td><td>0</td><td>0</td><td>—</td></tr>
<tr><td>1</td><td><code>USER:</code></td><td>role 模板</td><td>0</td><td>0</td><td>—</td></tr>
<tr><td>2</td><td>What</td><td>有效 instruction</td><td>0</td><td>1</td><td>0.50</td></tr>
<tr><td>3</td><td>color</td><td>有效 instruction</td><td>0</td><td>1</td><td>0.25</td></tr>
<tr><td>4</td><td>?</td><td>有效 instruction</td><td>0</td><td>1</td><td>0.80</td></tr>
<tr><td>5</td><td>Answer</td><td>高频模板</td><td>0</td><td>0</td><td>—</td></tr>
<tr><td>6</td><td>briefly</td><td>高频模板</td><td>0</td><td>0</td><td>—</td></tr>
<tr><td>7</td><td><code>ASSISTANT:</code></td><td>role 模板</td><td>0</td><td>0</td><td>—</td></tr>
<tr><td>8</td><td>red</td><td>response</td><td>1</td><td>1</td><td>0.60</td></tr>
<tr><td>9</td><td><code>&lt;eos&gt;</code></td><td>response 结束</td><td>1</td><td>1</td><td>0.90</td></tr>
</tbody></table></div>
<ol class="calc-steps">
  <li><b>标准 VIT：</b><code>L_response=−ln0.60−ln0.90=0.5108+0.1053=0.6161</code>。</li>
  <li><b>L2T instruction：</b><code>L_instruction=−ln0.50−ln0.25−ln0.80=0.6931+1.3863+0.2231=2.3025</code>。</li>
  <li><b>L2T 总和：</b><code>L=2.3025+0.6161=2.9186</code>；若对 5 个有效 token 取均值，则 <code>0.5837</code>。标准 VIT 对 2 个 token 的均值是 <code>0.3081</code>，因监督任务与分母不同，二者不能用绝对 loss 大小判断优劣。</li>
</ol>
<p class="example-check"><strong>推理：</strong>用户照常提供 “What color?”，模型只生成 “red”；不会先运行 filter，也不会自己先生成 instruction。</p>
''',
}


CORRECTION_NOTES = {
    "dora": """
<div class="audit-note"><strong>精读校勘：</strong>DoRA 的理论梯度分析使用 Eq. 6 的切空间投影，但主实验实际采用 denominator detach 的 Eq. 11 省内存近似；Table 7 支持“几乎不掉点且更省显存”，却不能证明理论梯度与实际优化动力学完全相同。最终 v6/PMLR 的 LLaMA-7B 主结果为 78.4；ReFT 引用的早期版本为 78.1。Eq. 5 省略了缩放，这份讲义在统一公式里补写 <code>α/r</code>。</div>
""",
    "mole": """
<div class="audit-note"><strong>结构校勘：</strong>MoLE 不是把整个 Transformer 变成专家。Self-Attention 仍用普通 LoRA；只有 FFN/MLP 的线性层拥有多组 LoRA experts，同一 FFN 内共享 router，且路由粒度是每个 token 的 top-1。只有在未命中的专家确实不执行时，top-1 才带来稀疏计算收益。</div>
""",
    "reft": """
<div class="audit-note"><strong>评测校勘：</strong>Commonsense 主表的 LoRA/DoRA 数字取自其它论文，而 ReFT 自跑、用了额外输出规范化且训练 6 epochs；应同时阅读附录 Table 14 的 3-epoch 公平版本。ReFT 在长 chain-of-thought 算术任务上落后 LoRA，是已被实验支持的边界。Table 17 的 7B DiReFT AQuA “221.3”是排版错误，应为 21.3。</div>
""",
    "mores": """
<div class="audit-note"><strong>公式与口径校勘：</strong>论文给出 <code>W_down∈R^(d×D)</code>、<code>W_up∈R^(D×d)</code>，随后写 <code>W_down W_up^T = I_D</code>；按这些形状左式无法相乘，且 <code>d&lt;D</code> 的降维不能称为两个空间间的双射。标题的“500×”只统计 LLM 内 TP*，不含视觉编码器与 connector；主表平均分也未完全追平 LoRA。</div>
""",
    "vl_adapter": """
<div class="audit-note"><strong>口径校勘：</strong>主文 Table 1、摘要和 Figure 1 报 Single Adapter 为 4.18%，附录 Table 11 对看似同一 <code>d=96</code> 配置报 4.36%。本讲义沿用主文 4.18%，并保留该内部差异。另请注意：Single 指“跨任务共享同一套逐层 adapters”，不是整网只有一个 adapter。</div>
""",
}


def extract_main(text: str) -> str:
    match = re.search(r"<main[^>]*>(.*?)</main>", text, re.S | re.I)
    if match:
        return match.group(1).strip()
    body = re.search(r"<body[^>]*>(.*?)</body>", text, re.S | re.I)
    return (body.group(1) if body else text).strip()


def crop_manual_assets(source_root: Path, asset_dir: Path) -> None:
    rendered_root = source_root / "tmp/pdfs/vlmintune_method_taxonomy"
    asset_dir.mkdir(parents=True, exist_ok=True)
    for slug, specs in MANUAL_CROPS.items():
        for name, page, box in specs:
            source = rendered_root / slug / f"page-{page:02d}.png"
            if not source.exists():
                raise FileNotFoundError(f"Missing rendered source page: {source}")
            with Image.open(source) as page_image:
                crop = page_image.convert("RGB").crop(box)
                # Small diagrams otherwise render at their literal pixel width.  Upsampling
                # is only for display geometry; no claim of added visual information.
                if crop.width < 900:
                    factor = min(4.0, 900 / crop.width)
                    crop = crop.resize(
                        (round(crop.width * factor), round(crop.height * factor)),
                        Image.Resampling.LANCZOS,
                    )
                crop.save(asset_dir / name, optimize=True)


def copy_imported_assets(fragment: str, source_html: Path, slug: str, asset_dir: Path) -> str:
    """Copy referenced raster assets and rewrite both src and image href targets."""
    refs = set(re.findall(r"(?:src|href)=([\"'])([^\"']+)\1", fragment, re.I))
    for _quote, ref in refs:
        if re.match(r"^(?:https?:|data:|file:|#|/)", ref):
            continue
        source = (source_html.parent / ref).resolve()
        if source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            continue
        if not source.exists():
            raise FileNotFoundError(f"Imported HTML refers to missing image: {source}")
        destination_name = f"{slug}__{source.name}"
        shutil.copy2(source, asset_dir / destination_name)
        fragment = fragment.replace(ref, f"assets/{destination_name}")
    return fragment


def figure(asset: str, caption: str, explanations: list[tuple[str, str]], alt: str | None = None) -> str:
    blocks = "".join(
        f"<p><strong>{html.escape(label)}：</strong>{text}</p>" for label, text in explanations
    )
    return f"""
<figure>
  <img src="assets/{html.escape(asset)}" alt="{html.escape(alt or caption)}">
  <figcaption>{caption}</figcaption>
  <div class="fig-explain">{blocks}</div>
</figure>
"""


def render_architecture_diagrams(asset_dir: Path) -> None:
    dot = shutil.which("dot")
    if not dot:
        raise RuntimeError("Graphviz 'dot' is required to render the eight architecture diagrams")

    header = r'''digraph G {
    graph [rankdir=TB, bgcolor="transparent", pad="0.08", nodesep="0.22",
           ranksep="0.34", splines=polyline, fontname="PingFang SC", fontsize=10];
    node [shape=box, style="rounded,filled", color="#96a3ad", fillcolor="#f6f7f9",
          fontname="PingFang SC", fontsize=10, margin="0.09,0.06"];
    edge [color="#71808d", fontcolor="#4f5f6c", fontname="PingFang SC",
          fontsize=8.5, arrowsize=0.65, penwidth=1.0];
'''
    for slug, body in ARCHITECTURE_DOT_BODIES.items():
        source = f"{header}\n{body}\n}}\n"
        result = subprocess.run(
            [dot, "-Tsvg"],
            input=source,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Graphviz failed for {slug}: {result.stderr.strip()}")
        (asset_dir / f"{slug}__architecture.svg").write_text(result.stdout, encoding="utf-8")


def architecture_lab(slug: str) -> str:
    guide = ARCHITECTURE_GUIDE[slug]
    return f'''
<section id="mechanism-{html.escape(slug)}" class="mechanism-lab">
  <div class="section-kicker">新增讲解 · 结构定位 + 可复算例子</div>
  <h2>模型结构、转换发生位置与完整小维度算例</h2>
  <div class="mechanism-legend" aria-label="结构图图例">
    <span class="legend-chip frozen">冻结</span>
    <span class="legend-chip trainable">可训练</span>
    <span class="legend-chip selector">selector / router / mask</span>
    <span class="legend-chip objective">loss / optimizer</span>
    <span class="legend-note">双边框＝推理仍需保留；MERGE＝可并入静态权重</span>
  </div>
  <div class="mechanism-grid">
    <div class="architecture-figure">
      <figure>
        <img src="assets/{html.escape(slug)}__architecture.svg" alt="{html.escape(guide['caption'])}">
        <figcaption>{guide['caption']}</figcaption>
        <div class="fig-explain">
          <p><strong>怎么看：</strong>{guide['look']}</p>
          <p><strong>转换在哪里：</strong>{guide['meaning']}</p>
          <p><strong>训练 / 推理：</strong>{guide['runtime']}</p>
        </div>
      </figure>
    </div>
    <aside class="worked-example" aria-label="{html.escape(slug)} 的完整小维度算例">
      {WORKED_EXAMPLES[slug]}
    </aside>
  </div>
</section>
'''


def insert_architecture_lab(fragment: str, slug: str) -> str:
    """Place the new lab after the one-line summary and before motivation."""
    heading = re.search(r"<h2>\s*研究动机\s*</h2>", fragment, re.I)
    if not heading:
        raise RuntimeError(f"Could not locate the motivation section for {slug}")
    insert_at = heading.start()
    # Some source summaries wrap each heading in <section>; others use a bare h2.
    # If the heading is the first child of a section, insert before that section so
    # we never create invalid nested sections.
    prior = fragment[: heading.start()]
    section_start = prior.rfind("<section")
    if section_start >= 0:
        section_open_end = prior.find(">", section_start)
        if section_open_end >= 0 and not prior[section_open_end + 1 :].strip():
            insert_at = section_start
    return fragment[:insert_at] + architecture_lab(slug) + fragment[insert_at:]


def lora_fragment() -> str:
    fig1 = figure(
        "lora__fig1_reparameterization.png",
        "LoRA Figure 1：冻结预训练矩阵 W，只训练低秩 A、B；两条支路的输出相加。",
        [
            ("怎么看", "蓝色支路是冻结的 <code>W₀x</code>；橙色支路先经 <code>A</code> 降到 rank <code>r</code>，再经 <code>B</code> 升回输出维度，得到 <code>BAx</code>。"),
            ("说明什么", "LoRA 约束的是任务增量 <code>ΔW=BA</code>，不是把预训练权重 <code>W₀</code> 本身做低秩压缩；最终 <code>W₀+ΔW</code> 通常仍为满秩。"),
            ("为什么重要", "这张图同时解释参数节省、零初始化和部署 merge：训练只存 A/B，部署可把 BA 加回 W₀。"),
        ],
    )
    table1 = figure(
        "lora__table1_latency.png",
        "LoRA Table 1：GPT-2 Medium 推理延迟；串联 Adapter 在小 batch、短序列时增加约 20.7%–30.3%。",
        [
            ("怎么看", "同一列先看 Fine-Tune/LoRA，再看 AdapterL 与 AdapterH 的括号增幅。batch=1、seq=128 时，19.8ms 对 23.9/25.8ms；长序列大 batch 时相对差距变小。"),
            ("说明什么", "LoRA 的低秩支路可以预先并入权重，不增加网络深度；串行 Adapter 必须在每次前向中执行额外层。"),
            ("证据边界", "“零额外时延”只在静态 merge 后成立。若运行时保留 A/B、动态切换 adapter，或同一 batch 混用多个任务，就仍有低秩矩阵乘或调度成本。"),
        ],
    )
    table2 = figure(
        "lora__table2_glue.png",
        "LoRA Table 2：RoBERTa / DeBERTa 在 GLUE 上的参数—性能比较。",
        [
            ("怎么看", "比较每个模型块中的 Full FT 与 LoRA：RoBERTa-base 125M 参数全调平均 86.4，LoRA 只训 0.3M 得 87.2；DeBERTa-XXL 1.5B 全调 91.1，LoRA 4.7M 得 91.3。"),
            ("说明什么", "在这些 NLU 设置中，限制权重增量为低秩没有明显伤害，还可充当一种正则化。"),
            ("证据边界", "星号行来自既有工作，只有作者自跑行有相应置信区间；不同训练 recipe 与模型不能据此推出“LoRA 普遍显著优于 FT”。"),
        ],
    )
    table3 = figure(
        "lora__table3_e2e.png",
        "LoRA Table 3：GPT-2 Medium/Large 在 E2E NLG 上的生成结果。",
        [
            ("怎么看", "GPT-2 Medium LoRA 只训 0.35M，BLEU 70.4，高于 354.92M 参数 Full FT 的 68.2；Large 只训 0.77M，同样得到 BLEU 70.4。"),
            ("说明什么", "论文不仅在分类任务验证低秩增量，也在自回归生成任务对比了 Prefix 与 Adapter。"),
            ("实际含义", "LoRA 的价值不是单一 benchmark 的百分点，而是让一个共享底座配多份 MB 级任务权重；不过最优 rank 在 E2E 并非恒为 1。"),
        ],
    )
    table4 = figure(
        "lora__table4_gpt3.png",
        "LoRA Table 4：GPT-3 175B 上 WikiSQL、MNLI 与 SAMSum 主结果。",
        [
            ("怎么看", "LoRA 4.7M 对 Full FT 175,255.8M：WikiSQL 73.4 vs 73.8；MNLI 91.7 vs 89.5；SAMSum 53.8/29.8/45.9 vs 52.0/28.0/44.5。"),
            ("说明什么", "在巨大底座上，任务特定增量可以比全量参数小四个数量级，却维持同一量级的任务效果。"),
            ("不要误读", "37.7M LoRA 并非在每项都比 4.7M 好，说明“可训练参数更多”不是单调收益；GPT-3 成本高，论文只给每任务典型波动而非完整重复实验。"),
        ],
    )
    fig2 = figure(
        "lora__fig2_scaling.png",
        "LoRA Figure 2：GPT-3 验证分数随可训练参数量变化。",
        [
            ("怎么看", "横轴是可训练参数的对数，纵轴是 WikiSQL/MNLI 验证准确率；LoRA 点在较少参数区间已达到高分，Prefix 方法增加过多 special tokens 后反而下降。"),
            ("说明什么", "不同 PEFT 的容量不是单一“参数数目”可以解释；参数放在哪里、是否改变输入分布同样关键。"),
            ("为什么重要", "这为后续 QLoRA 的 all-linear 发现、DoRA 的参数化改造和 ReFT 的位置选择埋下共同问题：预算相同，结构分配决定有效容量。"),
        ],
    )
    table5 = figure(
        "lora__table5_target_modules.png",
        "LoRA Table 5：固定约 18M 参数时，适配哪些 attention 权重更有效。",
        [
            ("怎么看", "只调一个矩阵时 rank=8；调 q+v 时 rank=4；调 q/k/v/o 时 rank=2。WikiSQL 上 q+v 与四矩阵都是 73.7，明显好于只调 q 的 70.4。"),
            ("说明什么", "在固定预算下，把低 rank 分散到更多相关矩阵，常比把大 rank 集中到一个矩阵更好；原论文多数主实验选择 q/v。"),
            ("实际含义", "今天配置 LoRA 时，<code>target_modules</code> 往往比盲目增加 rank 更值得先搜索。QLoRA 后来在 all-linear 覆盖下得到类似但更强的系统结论。"),
        ],
    )
    table6 = figure(
        "lora__table6_rank.png",
        "LoRA Table 6：GPT-3 的 rank 消融；q+v 上 rank 1 已很强。",
        [
            ("怎么看", "q+v 的 WikiSQL 在 r=1/2/4/8/64 为 73.4/73.3/73.7/73.8/73.5，MNLI 为 91.3/91.4/91.3/91.6/91.4。"),
            ("说明什么", "这些任务的有效更新方向非常少，支持“适配增量具有低 intrinsic rank”的工作假设。"),
            ("证据边界", "论文脚注已提醒跨语言等任务未必如此；GPT-2 E2E 的验证损失约在 r=16 才饱和。结论应写成“某些任务极低 rank 足够”，不是普适 rank=1 定律。"),
        ],
    )
    fig3 = figure(
        "lora__fig3_subspace.png",
        "LoRA Figure 3：不同 rank 训练得到的 A 因子右奇异子空间相似度。",
        [
            ("怎么看", "热图比较 r=8 与 r=64 的首若干方向；最前方向重合明显，后续方向快速变弱。"),
            ("说明什么", "作者据此认为更大的 rank 主要重复了少量稳定方向，新增方向贡献有限，间接支持低 intrinsic rank。"),
            ("审读提醒", "图比较的是低秩因子 A 的子空间，不是直接对最终 <code>ΔW=BA</code> 做 SVD；BA 分解具有基变换非唯一性，因此它是经验线索，不是“证明 ΔW 只有 rank 1”。"),
        ],
    )
    fig4 = figure(
        "lora__fig4_table7_directions.png",
        "LoRA Figure 4 / Table 7：跨随机种子的方向稳定性，以及 W 与 ΔW 的关系。",
        [
            ("怎么看", "热图比较不同 seed 的 A 子空间；Table 7 把 W 投影到 ΔW 的奇异方向。第 48 层 Wq 中，<code>||W||F=61.95</code>，r=4 的 <code>||ΔW||F=6.91</code>，而 W 在该子空间投影仅 0.32。"),
            ("说明什么", "任务更新没有简单复用 W 的主奇异方向，而更像放大预训练权重中已存在但未被强调的方向；作者给出的放大比约 6.91/0.32≈21.5。"),
            ("为什么重要", "这把“低秩有效”从工程观察提升为表征假设：下游任务可能主要重标少数方向，而不是重写整个模型。"),
        ],
    )
    return f"""
<h1>LoRA: Low-Rank Adaptation of Large Language Models</h1>
<div class="meta">Edward J. Hu 等 · ICLR 2022 · arXiv:2106.09685v2 · 本地 PDF 26 页</div>

<section><h2>一句话总结</h2>
<p class="summary">LoRA 冻结预训练权重，把每个任务需要的更新限制为低秩矩阵 <code>ΔW=(α/r)BA</code>；训练时只更新 A/B，部署时可把增量合并回 W₀。论文用 RoBERTa、DeBERTa、GPT-2、GPT-3 说明，在若干 NLU/NLG 任务上，极少的任务参数即可达到全量微调同量级性能。</p></section>

<section><h2>研究动机</h2>
<p>全量微调为每个任务复制一整套模型，并为全部参数维护梯度与 Adam 状态。GPT-3 175B 的多任务部署和训练显存都难以承受。串联 Adapter 会增加在线推理深度，Prefix/Prompt 又占上下文并可能改变输入分布。</p>
<p>LoRA 的关键假设不是“预训练权重低秩”，而是<strong>任务适配所需的权重增量具有低 intrinsic rank</strong>。如果成立，训练自由度就能从矩阵面积缩到两个细长矩阵的周长量级。</p>{fig1}</section>

<section><h2>方法设计</h2>
<div class="formula"><strong>权重与前向：</strong> W′ = W₀ + ΔW，ΔW = (α/r)BA；h = W₀x + (α/r)BAx。<br>A∈R<sup>r×k</sup>，B∈R<sup>d×r</sup>，r≪min(d,k)。单矩阵参数为 r(d+k)。</div>
<div class="grid"><div class="box"><h3>初始化</h3><p>A 用随机高斯，B=0，因此开始时 BA=0，模型函数严格等于底座。α/r 让换 rank 时更新尺度更稳定。</p></div><div class="box"><h3>部署</h3><p>静态任务可预计算 W=W₀+(α/r)BA；切换任务时减去旧增量、加上新增量。任务 checkpoint 只存 A/B。</p></div></div>
<p>原论文多数实验只在每层 self-attention 的 q/v 矩阵上加 LoRA，MLP、LayerNorm 与大多数 bias 冻结；这与后来 QLoRA 常用 all-linear 的 recipe 不同。</p>{table1}</section>

<section><h2>训练与实现要点</h2>
<ul>
  <li>模型/任务：RoBERTa-base/large、DeBERTa-XXL 做 GLUE；GPT-2 M/L 做 E2E、DART、WebNLG；GPT-3 175B 做 WikiSQL、MNLI、SAMSum。</li>
  <li>RoBERTa/DeBERTa 常用 q/v rank 8；GPT-2 q/v rank 4、α=32；GPT-3 LoRA 学习率 2e-4、2 epochs、batch 128。</li>
  <li>GPT-3 示例：训练显存约从 1.2TB 降到 350GB；任务 checkpoint 约从 350GB 降到 35MB，论文称约 10,000×；吞吐约从 32.5 到 43.1 token/s/V100。</li>
  <li>显存仍必须加载底座权重；LoRA 不解决底座精度，4-bit 底座是 QLoRA 的问题。</li>
</ul></section>

<section><h2>实验结果</h2>{table2}{table3}{table4}{fig2}</section>

<section><h2>Ablation 与机理分析</h2>{table5}{table6}{fig3}{fig4}
<div class="grid"><div class="box"><h3>组合性</h3><p>附录 LoRA+Prefix 在 WikiSQL 可高于单独 LoRA，MNLI 则无收益，说明“机制正交”不保证每个任务都叠加增益。</p></div><div class="box"><h3>低数据</h3><p>MNLI 100 样本 LoRA 63.8 vs FT 60.2；1K 时 85.6 vs 85.8。低秩限制有时像正则化，但优势并非所有数据规模都存在。</p></div></div></section>

<section><h2>局限与容易误解的点</h2>
<ul>
  <li>低秩的是 ΔW；最终 W₀+ΔW 通常满秩。</li>
  <li>“零推理时延”要求静态 merge；运行时多 adapter 或同 batch 多任务会改变该结论。</li>
  <li>很小 rank 的证据来自特定文本任务；原论文未系统研究 MLP、LayerNorm、视觉模型或跨语言。</li>
  <li>多张主表混合作者复现与前人已报数字，不能把少数设置中的领先泛化为统计显著的普遍优势。</li>
  <li>Fig. 3–4 的 A 因子子空间分析受 BA 分解非唯一性影响，应视为间接证据。</li>
</ul></section>

<section><h2>整篇文章的逻辑</h2>
<ol><li>全量任务副本与串行 Adapter 的成本不可持续。</li><li>假设任务权重增量低秩，用 BA 重参数化并冻结 W₀。</li><li>在理解与生成、从 125M 到 175B 的模型上验证参数—性能前沿。</li><li>用 target-module、rank 和子空间分析解释为何小 rank 可行。</li><li>留下两个开放问题：不同任务怎样选 rank/层，以及低秩规律能否跨模态、跨架构成立。</li></ol></section>
"""


def qlora_fragment() -> str:
    fig1 = figure(
        "qlora__fig1_memory_paths.png",
        "QLoRA Figure 1：Full FT、LoRA 与 QLoRA 的底座/优化器/适配器内存路径。",
        [
            ("怎么看", "Full FT 为整个 16-bit 模型维护 optimizer state；LoRA 冻结 16-bit 底座，只更新 adapter；QLoRA 再把冻结底座以 4-bit 保存，并在计算时反量化。"),
            ("说明什么", "普通 LoRA 已省掉基座梯度与 Adam 状态，但仍要常驻一份高精度大模型；QLoRA 真正新增的是数值存储与峰值内存系统。"),
            ("关键区别", "“4-bit training”不表示梯度和矩阵乘都是 4-bit。论文路径是 4-bit 存储、BF16 计算、高精度 LoRA 更新。"),
        ],
    )
    fig23 = figure(
        "qlora__fig2_3_ablations.png",
        "QLoRA Figures 2–3：LoRA 覆盖位置比 rank 更关键；NF4 的 bit-for-bit 精度优于 FP4。",
        [
            ("怎么看", "Figure 2 的不同配置显示只调 q/v 无法稳定复制 16-bit 表现，而覆盖全部线性层后结果更接近；Figure 3 在相同 bit 数下比较 Int/FP4/NF4。"),
            ("说明什么", "QLoRA 的性能不是“量化一次就自然恢复”。足够广的 LoRA 目标层负责吸收量化误差；NF4 又比普通 4-bit 格式减少权重表示误差。"),
            ("实际含义", "实践中应先核对 <code>target_modules=all-linear</code>、compute dtype 与 quantization recipe，再讨论 rank。"),
        ],
    )
    table3 = figure(
        "qlora__table3_precision.png",
        "QLoRA Table 3：GLUE 与 Super-NaturalInstructions 上的精度复现。",
        [
            ("怎么看", "RoBERTa-large GLUE：full BF16 88.6，BF16 LoRA 88.8，QLoRA Int8 88.8、FP4 88.6；T5-3B full replication 54.9，NF4+DQ 55.3。"),
            ("说明什么", "在作者真正能运行 full-FT 复现的较小模型上，4-bit 底座 + LoRA 可以复制同量级结果。"),
            ("证据边界", "作者只亲自复现 full FT 到 3B。T5-11B 虽引用 full-FT 62.0，但 QLoRA 60.9 未完全追平，不能把结论外推成所有尺度无损。"),
        ],
    )
    table4 = figure(
        "qlora__table4_mmlu.png",
        "QLoRA Table 4：LLaMA 7B–65B、不同量化格式的 5-shot MMLU。",
        [
            ("怎么看", "BF16 LoRA 总体均值 53.0，FP4 52.2，NF4+DQ 53.1。均值接近，但逐模型/数据并不完全相等，例如 33B FLAN 60.5 vs 59.2。"),
            ("说明什么", "论文的“fully recovers”主要是总体与统计口径；NF4+DQ 比 FP4 稳健，不代表每一格都等于或超过 BF16。"),
            ("不要误读", "7B–65B 这里对照的是 16-bit LoRA，不是 16-bit 全参数微调。"),
        ],
    )
    table6 = figure(
        "qlora__table6_vicuna.png",
        "QLoRA Table 6：Guanaco 在 80 条 Vicuna prompts 上相对 ChatGPT 的 GPT-4 评分。",
        [
            ("怎么看", "Guanaco-65B 4-bit、约 41GB，被报为 ChatGPT 分数的 99.3%±4.4%；33B 约 21GB、97.8%±4.4%。"),
            ("说明什么", "小而高质量的 OASST1 子集配合 QLoRA 能把 33B/65B 模型适配为强聊天模型，展示了单 GPU 大模型微调的可行性。"),
            ("证据边界", "这不是“能力等同 ChatGPT”。只有 80 prompts，置信区间宽，judge 有顺序与自偏好；论文自己的结论是现有 chatbot benchmark 不足以可靠排序。"),
        ],
    )
    table7 = figure(
        "qlora__table7_elo.png",
        "QLoRA Table 7：人类与 GPT-4 judge 在 Vicuna / OpenAssistant benchmark 上给出不同排序。",
        [
            ("怎么看", "同一模型在 Human Vicuna、GPT-4 Vicuna、GPT-4 OA 三组 Elo 与 rank 并不一致；例如 benchmark 改变后 ChatGPT 与 Guanaco 的相对次序变化。"),
            ("说明什么", "聊天能力的结论对 prompt 集与 judge 敏感，单一自动评测不能等同广义对话质量。"),
            ("论文自检", "系统级 GPT-4 与人类相关性仅中等，样本级一致性更低；这是 QLoRA 论文很值得保留的反身性结果。"),
        ],
    )
    fig4 = figure(
        "qlora__fig4_rank.png",
        "QLoRA Appendix Figure 4：all-linear 条件下，LoRA rank 8–64 的 Rouge-L 差异很小。",
        [
            ("怎么看", "每个 rank 叠加多组学习率、dropout 与随机种子；当所有线性层都覆盖后，各 rank 的点云高度重叠。"),
            ("说明什么", "在这套 LLaMA-7B/Alpaca recipe 里，容量瓶颈主要不是 rank，而是目标层是否完整。"),
            ("证据边界", "这是特定模型、数据和指标的消融；不能推出所有任务都对 rank 不敏感。"),
        ],
    )
    table10 = figure(
        "qlora__table10_loss_scope.png",
        "QLoRA Table 10：纯文本指令数据上，source+target loss 的 MMLU 均值低于 target-only。",
        [
            ("怎么看", "四个数据集均值：训练 source+target 为 37.5，只训练 target 为 38.6。"),
            ("说明什么", "把所有输入 token 都纳入 loss 不一定有益；输入中的模板、角色或低信息文本可能成为噪声。"),
            ("和 L2T 的关系", "这不与 L2T 矛盾。L2T 在视觉条件下只监督过滤后的 informative instruction；它反而说明 QLoRA+L2T 必须做联合实验证明，不能仅凭“正交”预设增益。"),
        ],
    )
    table11 = figure(
        "qlora__table11_data_quality.png",
        "QLoRA Table 11：数据集选择的影响远大于 50K–150K 样本数或 1–3 epochs。",
        [
            ("怎么看", "同一数据集内部扩大样本/epoch 通常只变 0–0.5 MMLU，而数据集之间可差 1.5–8.0。"),
            ("说明什么", "论文的核心经验是“dataset suitability/quality 比单纯规模更重要”；OASST1 实际筛后只有约 9,209 样本。"),
            ("实际含义", "PEFT 省显存并不修复数据问题。若任务分布、模板或质量错位，再大的 rank 与更多 epoch 也可能只放大偏差。"),
        ],
    )
    fig6 = figure(
        "qlora__fig6_memory_breakdown.png",
        "QLoRA Figure 6：7B–65B 的训练内存组成估算。",
        [
            ("怎么看", "总量约为 7B 6.9GB、13B 11.3GB、33B 24.7GB、65B 45.0GB；蓝色底座仍占大头，输入梯度和 optimizer 随模型增长。"),
            ("说明什么", "33B 略超 24GB，Paged Optimizer 通过统一内存处理峰值；65B 接近 48GB 单卡上限。"),
            ("证据边界", "图按 batch1、seq512、gradient checkpointing 估计，只计部分输入梯度且不含 attention activation；不能当作所有训练配置的峰值保证。"),
        ],
    )
    return f"""
<h1>QLoRA: Efficient Finetuning of Quantized LLMs</h1>
<div class="meta">Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer · NeurIPS 2023 · arXiv:2305.14314v1 · 本地 PDF 26 页</div>

<section><h2>一句话总结</h2>
<p class="summary">QLoRA 不是新的低秩表达式，而是“4-bit 冻结底座 + BF16 反量化计算 + LoRA 更新”的训练系统。NF4 降低权重量化误差，Double Quantization 再压缩 scale 常数，Paged Optimizer 处理显存尖峰；论文据此在单张 48GB GPU 上微调 65B 模型。</p></section>

<section><h2>研究动机</h2>
<p>普通 LoRA 只让少量参数获得梯度，却仍须加载 16-bit 底座。以 7B 为例，LoRA 参数可能只有约 26MB，但基座仍约 5GB（4-bit 时）或更高（16-bit），输入梯度与 activation 也可超过 adapter 本身。</p>
<p>因此 QLoRA 的问题不是“怎样再缩小 A/B”，而是<strong>怎样让冻结权重以更低精度常驻，同时让误差不破坏适配质量，并在长序列峰值时避免 OOM</strong>。</p>{fig1}</section>

<section><h2>方法设计</h2>
<div class="grid-3"><div class="box"><h3>NF4</h3><p>对近似零均值正态权重取等概率 quantiles，构造含精确 0 的 16 值 codebook。其“信息论最优”有分布前提。</p></div><div class="box"><h3>Double Quantization</h3><p>再量化第一层量化的 FP32 scale。开销由 0.5 降到约 0.127 bit/param，节省约 0.373 bit/param；65B 约 3GB。</p></div><div class="box"><h3>Paged Optimizer</h3><p>用 CUDA unified memory 在峰值时把 optimizer state page 到 CPU；解决瞬时 OOM，不是永久消除 activation。</p></div></div>
<div class="formula"><strong>单层计算：</strong>Y<sub>BF16</sub> = X<sub>BF16</sub>·doubleDequant(W<sub>NF4</sub>) + X<sub>BF16</sub>L₁L₂。<br>4-bit 只保存冻结 W；前/反向乘法与 LoRA 参数保持较高精度。</div>
<p>聊天模型实验固定 rank 64、α=16，并在 Transformer block 的所有线性层插 LoRA；这与 LoRA 原论文主要调 q/v 的配置不同。</p>{fig23}</section>

<section><h2>训练与实现要点</h2>
<ul>
  <li>精度复现：RoBERTa-large/GLUE，T5 80M–11B/Super-NaturalInstructions，LLaMA 7/13/33/65B + Alpaca/FLAN v2 + MMLU。</li>
  <li>聊天：LLaMA 7–65B、8 个 instruction datasets，仅做 cross-entropy SFT，无 RLHF；最佳 Guanaco 数据来自清洗后的 OASST1。</li>
  <li>NF4 + Double Quantization + BF16 compute + Paged Optimizer；Adam、constant LR、group-by-length。</li>
  <li>论文称 65B 单 48GB GPU 约 24 小时；33B 单 24GB GPU 少于 12 小时。</li>
</ul></section>

<section><h2>量化精度与主结果</h2>{table3}{table4}</section>

<section><h2>Guanaco 与评测可信度</h2>{table6}{table7}
<div class="note"><strong>定性反例也必须读：</strong>论文逐页展示事实幻觉、随机拒答、prompt injection 泄密、算术与心智理论错误。它们提醒我们，“高自动评分”与“可靠助手”不是同一个命题。</div></section>

<section><h2>Ablation 与系统边界</h2>{fig4}{table10}{table11}{fig6}
<ul>
  <li>NF4 的正态前提只近似成立；附录 Shapiro–Wilk 检查有约 7.5% neurons 判非正态。</li>
  <li>Paged Optimizer 没有系统 wall-clock benchmark；作者只报告代表设置未见明显减速。</li>
  <li>大尺度主要比较 QLoRA 与 16-bit LoRA；不能写成已证明 33B/65B 匹配 16-bit full FT。</li>
  <li>保持量化部署时通常保留底座与 adapter；若先解量化 merge 再量化，会引入另一轮误差，原论文未把它等同 LoRA 的零时延 merge。</li>
</ul></section>

<section><h2>整篇文章的逻辑</h2>
<ol><li>LoRA 省 optimizer state，却没有解决冻结底座本身的精度成本。</li><li>用 NF4、DQ、Paged Optimizer 分别处理精度、scale 开销与峰值。</li><li>在可运行 full FT 的小模型上复现精度，在 7–65B 上对照 16-bit LoRA。</li><li>用 1,000+ 次微调研究数据、rank、target modules 与聊天评测。</li><li>结论不是“4-bit 永远无损”，而是合适量化与足够覆盖的 LoRA 能把单卡大模型 SFT 变成现实。</li></ol></section>
"""


def l2t_fragment() -> str:
    fig1 = figure(
        "l2t__fig1_benchmarks.png",
        "L2T Figure 1：五种模型、16 个 benchmark 上 L2T 与标准 VIT 的雷达图。",
        [
            ("怎么看", "同一种颜色的实线/虚线比较 L2T 与 VIT；提升主要出现在 OCR、caption、grounding 等基础视觉能力，并非所有 VQA 点都同幅上升。"),
            ("说明什么", "只改变监督 token 就能跨 TinyLLaVA、LLaVA-1.5、LLaVA-NeXT 带来整体收益，支持目标设计而非特定结构技巧。"),
            ("口径校勘", "摘要称“最高 9%”，Introduction 写 overall“最高 6%”，Table 1 精确 overall 最大为 8.5%；三种表述不是同一个精确指标，本讲义以表中数值为准。"),
        ],
    )
    fig2 = figure(
        "l2t__fig2_shortcut.png",
        "L2T Figure 2：模型不看图也可凭语言先验猜中 “white / water” 的 shortcut。",
        [
            ("怎么看", "训练样例包含图像、问题与回答；下方只给 instruction，语言模型仍能生成常见答案。"),
            ("说明什么", "answer-only loss 只要求预测结果，若数据偏差足够强，模型可用语言共现完成任务而忽略视觉输入。"),
            ("为什么重要", "L2T 的目标不是一般语言 instruction tuning，而是让 instruction 自身成为需要图像才能预测的额外监督，从而提高视觉依赖。"),
        ],
    )
    objective = figure(
        "l2t__fig3_4_objective.png",
        "L2T Figures 3–4 / Equations 1–2：从 answer-only 改为 filtered instruction + answer。",
        [
            ("怎么看", "标准 VIT 只让粉色 answer tokens 产生 loss；L2T 让绿色、含视觉语义的 instruction tokens 也产生 loss，但排除 system、角色和任务模板。"),
            ("说明什么", "方法本质是 label mask / target serialization 的变化，没有新增网络参数，也没有独立 loss 权重。"),
            ("实际含义", "L2T 可以包在全量训练、LoRA、QLoRA 或 Adapter 外；因此“零新增参数”不能被写成“本身是 PEFT”。"),
        ],
    )
    visual = figure(
        "l2t__fig5_7_visual_evidence.png",
        "L2T Figures 5–7：视觉贡献 VC、attention 与定性案例。",
        [
            ("怎么看", "左侧 VC 分布比较有图与随机噪声图下回答对数概率差；中间 attention 颜色越深代表更关注视觉 token；下方案例比较 VIT/L2T 的 OCR、caption 与幻觉。"),
            ("说明什么", "VQAv2 train 的 VC 均值约 0.534→0.584，DocVQA test 1.274→1.379，约 9% 相对提高；attention 与案例提供相互呼应的机制证据。"),
            ("证据边界", "“无图”实现是随机噪声替换，可能引入分布外效应；更高 attention 也不自动等于因果上的视觉理解提升。"),
        ],
    )
    table1 = figure(
        "l2t__table1_main.png",
        "L2T Table 1：五个模型、四类共 16 个多模态 benchmark 的主结果。",
        [
            ("怎么看", "每个模型块最后一列是相对改善；overall 分别 +6.1%、+6.2%、+5.6%、+6.9%、+8.5%。分类均值显示 General VQA 约 +0.2%–1.5%，OCR +3.9%–8.8%，Caption +11.5%–17.6%。"),
            ("说明什么", "收益明显集中在 OCR/caption/grounding 等要求提取视觉内容的任务，符合“instruction loss 增加视觉依赖”的动机。"),
            ("不要误读", "表中存在退步项，例如 LLaVA-1.5-13B 的 COCO CIDEr 115.20→114.46；“across all benchmarks consistently”是过强概括。"),
        ],
    )
    table2 = figure(
        "l2t__table2_hallucination.png",
        "L2T Table 2：POPE、CHAIR、MMHAL 与 HallusionBench 的幻觉评估。",
        [
            ("怎么看", "MMHAL GPT score 1.73→2.36，hallucination rate 0.68→0.53；CHAIR greedy 的 sentence/instance 48.6/13.4→46.2/11.8。"),
            ("说明什么", "让模型学习图像相关 instruction，能降低一部分仅靠语言先验生成物体的倾向；多个指标方向一致。"),
            ("证据边界", "POPE random 88.47 基本不变，不是每个 hallucination slice 都有大幅收益；自动 judge 与数据构造仍会影响结论。"),
        ],
    )
    fig89 = figure(
        "l2t__fig8_9_scaling.png",
        "L2T Figures 8–9：训练/测试 loss 分布与数据规模消融。",
        [
            ("怎么看", "Figure 8 中 L2T 的 response-only 训练 loss 略高、DocVQA 测试 loss 更低；Figure 9 比较只用 40/60/80% SFT 或 pretraining data 的雷达图。"),
            ("说明什么", "L2T 更像正则化而非更强记忆答案；在部分小数据设置下相对 VIT 的收益反而更大。"),
            ("证据边界", "这是相对改善，不代表少数据 L2T 的绝对性能一定高于全数据 VIT；数据比例与 instruction 类型共同变化时要谨慎归因。"),
        ],
    )
    table4 = figure(
        "l2t__table4_template_removal.png",
        "L2T Table 4：逐步移除 system 与 task templates 的消融。",
        [
            ("怎么看", "只学 answer 是基线；直接学完整 instruction+answer 约 +6%；去掉 system 后约 +9%；再去高频 task templates 后约 +11%。"),
            ("说明什么", "收益不是来自盲目监督更多 token，而来自筛选真正携带图像语义的 token。格式与固定模板会成为捷径或噪声。"),
            ("实际含义", "复现 L2T 的关键不是把所有 prompt label 从 -100 改回来，而是可靠定位 instruction 内容、角色边界与高频模板。"),
        ],
    )
    table6cost = figure(
        "l2t__table6_fig10_generalization_cost.png",
        "L2T Table 6 / Figure 10：在 Prism-7B 上的迁移，以及低于 1% 的训练吞吐差异。",
        [
            ("怎么看", "Prism-7B 上 TextVQA 52.8→55.6、RefCOCO 56.7→66.0、VSR 53.2→61.7；LLaVA-1.5-7B 上 VIT 约 0.334±0.005 step/s，L2T 约 0.331±0.005。"),
            ("说明什么", "效果不只绑定 LLaVA 的单一视觉塔；多预测一段已有 token 不改变前向长度，因此额外训练吞吐损失很小。"),
            ("不要误读", "L2T 本身的增量开销低，不等于整体训练轻量。主实验仍训练 connector+LLM，LLaVA-NeXT 还训练 vision encoder。"),
        ],
    )
    table7 = figure(
        "l2t__table7_self_improvement.png",
        "L2T Table 7：加入 100K 自生成 instruction–response 后的 pilot 结果。",
        [
            ("怎么看", "VQAv2-L 56.68→60.80、RefCOCO-L 17.42→24.53；Flickr30k-L 68.55→68.32，仍有退步项。"),
            ("说明什么", "能预测 instruction 让模型有可能从 image-only 输入自举生成训练对，这是 L2T 超出 label mask 的潜在扩展。"),
            ("证据边界", "这里只是单轮 pilot，没有质量过滤、长期迭代稳定性或与外部生成器的公平对照，不能当作已解决自训练。"),
        ],
    )
    fig11 = figure(
        "l2t__fig11_cases.png",
        "L2T Figure 11：OCR、caption 与 hallucination mitigation 的更多案例。",
        [
            ("怎么看", "每组给相同图片与 instruction，依次列 VIT 与 L2T 输出；重点看文本读取、物体属性和是否捏造未出现内容。"),
            ("说明什么", "案例把 Table 1/2 的平均数落到具体行为：L2T 输出通常包含更准确的视觉细节，较少依赖常见语言模式。"),
            ("证据边界", "论文选择的定性样例不能估计失败率；应与完整 benchmark 和退步项一起读。"),
        ],
    )
    table11 = figure(
        "l2t__table11_compute.png",
        "L2T Table 11：不同 instruction/answer 长度比下的 samples/s 与 steps/s。",
        [
            ("怎么看", "Q/A ratio 从 0.05 到 20，VIT 与 L2T 的 step/s 都约 0.327–0.339，差异处于报告波动范围。"),
            ("说明什么", "两种方法处理同一完整序列，改变的主要是哪些 labels 参与交叉熵，因此没有新增推理计算，也几乎没有额外训练前向。"),
            ("实际含义", "部署端完全不需要保存 L2T 模块；模型大小与时延由底层全参/LoRA/QLoRA 等训练方案决定。"),
        ],
    )
    return f"""
<h1>Learning to Instruct for Visual Instruction Tuning</h1>
<div class="meta">Zhihan Zhou, Feng Hong, Jiaan Luo, Yushi Ye, Jiangchao Yao, Dongsheng Li, Bo Han, Ya Zhang, Yanfeng Wang · NeurIPS 2025 · arXiv:2503.22215v2 · 本地 PDF 22 页</div>
<div class="audit-note"><strong>版本校勘：</strong>arXiv 元数据列 8 位作者，v2 PDF 首页另列 Yushi Ye；本讲义按 PDF 列 9 人。摘要“up to 9%”、正文 Introduction“最高 6%”与 Table 1 精确 overall 最大 8.5%是不同口径，以下以表格为主。</div>

<section><h2>一句话总结</h2>
<p class="summary">传统视觉指令微调只让 answer tokens 产生 loss；L2T 还让经过模板过滤、真正携带视觉语义的 instruction tokens 产生 loss。它不增加模型参数或推理结构，却把监督从“学会回答”扩展为“看图后也能生成问题/指令”，以此减少语言捷径、过拟合与幻觉。</p>{fig1}</section>

<section><h2>研究动机</h2>
<p>如果训练集中“盘子通常是白色”“杯子里通常是水”，模型只需语言先验就能降低 answer loss。answer-only VIT 并未直接奖励“先理解图像再回答”，还可能牺牲 caption、OCR 等预训练能力。</p>{fig2}</section>

<section><h2>方法设计</h2>
<div class="formula"><strong>标准 VIT：</strong>L = −Σ<sub>i∈A</sub> log p(Aᵢ | V,I,A&lt;i)。<br><strong>L2T：</strong>L = −Σ<sub>i∈I*</sub> log p(Iᵢ | V,I&lt;i) − Σ<sub>j∈A</sub> log p(Aⱼ | V,I,A&lt;j)。</div>
<p><code>I*</code> 排除 system prompt、USER/ASSISTANT 等格式 token，以及通过全语料句频识别出的高频任务模板。L2T 只用于 end-to-end SFT；预训练的 image-caption 固定模板与图像无关，不应用。</p>{objective}</section>

<section><h2>训练与实现要点</h2>
<ul>
  <li>TinyLLaVA Qwen2-0.5B/Phi-2-3B、LLaVA-1.5 7B/13B、LLaVA-NeXT 7B，另测 Prism-7B。</li>
  <li>预训练用 LLaVA-pretrain-558k；Tiny/LLaVA-1.5 SFT 用 LLaVA-mix-665k；NeXT 用 LLaVA-NeXT-Data。</li>
  <li>Tiny/LLaVA-1.5 SFT：LR 2e-5、batch128、1 epoch，训练 MLP connector+LLM；NeXT SFT 还训练 vision encoder，LR1e-5、batch32。</li>
  <li>全部 AdamW、cosine decay、warmup ratio 0.03。论文主文没有把 L2T 与 LoRA/QLoRA 做 2×2 受控组合。</li>
</ul>{visual}</section>

<section><h2>实验结果</h2>{table1}{table2}</section>

<section><h2>Ablation、泛化与定性案例</h2>{fig89}{table4}{table6cost}{table7}{fig11}{table11}</section>

<section><h2>局限与容易误解的点</h2>
<ul>
  <li>“扩展训练数据”准确说是扩展监督 token，不是增加独立图文样本。</li>
  <li>同一图像有许多合理 instruction；预测现有 instruction 也可能学习数据集/任务分布，而非唯一视觉语义。</li>
  <li>Eq. 2 没有 instruction/answer loss 权重；较长 instruction 会隐式改变样本与任务权重。</li>
  <li>收益依赖 instruction 包含视觉信息；固定模板型 grounding 收益弱，偏置或有害 instruction 也可能被额外强化。</li>
  <li>只验证 supervised fine-tuning；RL/RLHF 的潜在收益属于作者讨论，没有实验。</li>
  <li>L2T 是目标函数策略，不是参数高效方法；“零新增参数”与“训练多少原参数”必须分开报告。</li>
</ul></section>

<section><h2>整篇文章的逻辑</h2>
<ol><li>answer-only VIT 允许模型用语言先验走捷径。</li><li>把有视觉信息的 instruction 也变成 target，但过滤固定模板。</li><li>用 VC、attention 与案例证明视觉依赖确实提高。</li><li>在五个模型、16 个任务与多类 hallucination 指标上验证，并做数据量/模板消融。</li><li>结论是“监督集合本身是一条独立设计轴”，不是“所有 prompt token 都应参与 loss”。</li></ol></section>
"""


def manual_fragment(slug: str) -> str:
    return {"lora": lora_fragment, "qlora": qlora_fragment, "l2t": l2t_fragment}[slug]()


def load_imported_fragment(source_root: Path, slug: str, asset_dir: Path) -> tuple[str, Path]:
    source_html = source_root / IMPORTED_PAPERS[slug]
    if not source_html.exists():
        raise FileNotFoundError(f"Missing verified single-paper summary: {source_html}")
    fragment = extract_main(source_html.read_text(encoding="utf-8", errors="ignore"))
    fragment = copy_imported_assets(fragment, source_html, slug, asset_dir)
    fragment = CORRECTION_NOTES.get(slug, "") + fragment
    return fragment, source_html


def front_matter() -> str:
    return r"""
<section id="coordinate-system" class="synthesis">
  <div class="section-kicker">导读 · 先建立统一坐标系</div>
  <h1>三类适配载体 + 两条正交覆盖轴</h1>
  <div class="summary"><strong>一句话总览：</strong>这八篇不是八个同层竞争者。LoRA、DoRA、MoLE 改权重映射；ReFT、MoReS 改隐藏表示；VL-Adapter 插入模块。QLoRA 回答“冻结底座怎样以 4-bit 存储和训练”，L2T 回答“哪些 token 产生损失”。</div>

  <div class="axis-map">
    <div class="axis-label">适配载体</div>
    <div class="axis-card weight"><b>权重空间</b><span>LoRA → DoRA / MoLE</span><small>任务增量 ΔW、幅值/方向、条件专家</small></div>
    <div class="axis-card representation"><b>表示空间</b><span>ReFT · MoReS</span><small>层 × token × 干预函数；视觉模态选择</small></div>
    <div class="axis-card module"><b>插入模块</b><span>VL-Adapter</span><small>Attention / FFN 后的残差瓶颈</small></div>
    <div class="axis-label">覆盖轴</div>
    <div class="axis-card system"><b>数值 / 内存轴</b><span>QLoRA</span><small>NF4、Double Quantization、Paged Optimizer</small></div>
    <div class="axis-card objective"><b>监督目标轴</b><span>L2T</span><small>informative instruction + answer；过滤模板</small></div>
  </div>

  <h2>从全量更新推到八篇论文</h2>
  <div class="derivation">
    <div><b>0 · Full FT</b><span>更新声明范围内的全部 θ</span></div><i>→</i>
    <div><b>1 · LoRA</b><span>把 ΔW 限制为低秩 BA</span></div><i>→</i>
    <div><b>2 · QLoRA</b><span>底座仍贵：改为 4-bit 存储</span></div><i>→</i>
    <div><b>3 · DoRA</b><span>低秩更新太耦合：拆幅值/方向</span></div><i>→</i>
    <div><b>4 · MoLE</b><span>单一 ΔW 冲突：按 token 选专家</span></div>
  </div>
  <div class="derivation second">
    <div><b>5 · ReFT</b><span>不改权重，直接编辑 h</span></div><i>→</i>
    <div><b>6 · MoReS</b><span>只编辑视觉 token 的 h</span></div>
    <div class="branch-note">另一条容量路线</div>
    <div><b>7 · VL-Adapter</b><span>插入显式残差瓶颈</span></div>
    <div class="branch-note">正交的监督路线</div>
    <div><b>8 · L2T</b><span>重画 loss mask，不改推理结构</span></div>
  </div>
  <p class="small">上图是教学推导，不表示严格的论文历史继承。尤其 MoReS 不是 ReFT 的软件子类，VL-Adapter 也早于其中多篇工作。</p>

  <h2>先算一次参数量：LoRA 为什么省</h2>
  <div class="grid-3">
    <div class="box"><h3>全量矩阵</h3><p><code>W ∈ R^(d_out×d_in)</code></p><p>可训练量：<code>d_out·d_in</code></p></div>
    <div class="box"><h3>LoRA 增量</h3><p><code>ΔW=(α/r)BA</code></p><p>可训练量：<code>r(d_in+d_out)</code></p></div>
    <div class="box"><h3>4096×4096, r=8</h3><p>全量 16,777,216；LoRA 65,536。</p><p><strong>只看该矩阵，少 256×。</strong></p></div>
  </div>
  <div class="note"><strong>三个不同概念：</strong>可训练参数少，不等于基座占用小；基座占用小，不等于 activation 小；没有新增参数，也不等于训练的是少量参数。LoRA、QLoRA、L2T 分别对应这三种常被混淆的说法。</div>

  <h2>八篇总表：改哪里、训练什么、怎样部署</h2>
  <div class="table-wrap"><table class="master-table">
    <thead><tr><th>方法</th><th>核心问题</th><th>干预对象</th><th>最小核心式/规则</th><th>运行时形态</th><th>证据边界</th></tr></thead>
    <tbody>
      <tr><th>LoRA</th><td>每任务全量权重太贵</td><td>静态权重增量</td><td><code>W′=W₀+(α/r)BA</code></td><td>可合并</td><td>原论文主要是文本 LM，常用 q/v</td></tr>
      <tr><th>QLoRA</th><td>冻结底座仍占大量显存</td><td>LoRA + 量化存储系统</td><td><code>dequant(NF4(W₀))x + BAx</code></td><td>通常量化底座 + adapter</td><td>大模型主要对照 16-bit LoRA，不是 full FT</td></tr>
      <tr><th>DoRA</th><td>LoRA 幅值/方向更新耦合</td><td>逐列 magnitude + 低秩 direction</td><td><code>m⊙(W₀+BA)/||W₀+BA||c</code></td><td>可合并</td><td>理论梯度与主实验 detach 近似有差别</td></tr>
      <tr><th>MoLE</th><td>多域混合数据负迁移</td><td>FFN 的 token 条件 LoRA experts</td><td><code>e(x)=argmax G(x)</code></td><td>保留 router / 专家</td><td>只验证 LLaVA-1.5 7B、三类域</td></tr>
      <tr><th>ReFT</th><td>适配未必必须写进权重</td><td>选定层与位置的 hidden state</td><td><code>h′=h+Rᵀ(Wh+b−Rh)</code></td><td>保留 intervention</td><td>文本 LM；长 CoT 算术较弱</td></tr>
      <tr><th>MoReS</th><td>文本模态压过视觉模态</td><td>稀疏视觉 token 表示</td><td><code>h′v=hv+Wup φ(hv)</code></td><td>保留 steering</td><td>TP* 只算 LLM 内参数，公式有维度疑点</td></tr>
      <tr><th>VL-Adapter</th><td>多任务 V&amp;L 全量迁移太贵</td><td>逐层 bottleneck modules</td><td><code>h′=h+Wup GELU(Wdown h)</code></td><td>保留 adapter</td><td>CLIP-BART/T5 时代，不是 decoder-only MLLM</td></tr>
      <tr><th>L2T</th><td>answer-only 监督鼓励语言捷径</td><td>loss mask / target tokens</td><td><code>I* + A</code> 产生 NLL</td><td>推理无新增结构</td><td>本身不是 PEFT；收益依赖指令含视觉信息</td></tr>
    </tbody>
  </table></div>

  <h2>阅读顺序</h2>
  <ol class="reading-order">
    <li><strong>先读 LoRA：</strong>建立低秩权重更新、参数计数、初始化与 merge 语义。</li>
    <li><strong>再把 QLoRA 与 DoRA 分开：</strong>一个改数值系统，一个改参数化几何。</li>
    <li><strong>用 MoLE 理解条件计算：</strong>“多个 LoRA”不是重点，router 与稀疏执行才是。</li>
    <li><strong>转到 ReFT / MoReS：</strong>从权重空间切换到 residual stream，并理解层、位置、模态三种选择。</li>
    <li><strong>最后读 VL-Adapter 与 L2T：</strong>前者增加模块容量，后者改变监督集合；它们回答完全不同的问题。</li>
  </ol>
</section>
"""


def closing_matter() -> str:
    return r"""
<section id="synthesis" class="synthesis ending">
  <div class="section-kicker">综合 · 从论文结论回到方法选择</div>
  <h1>组合、选型、边界与复习</h1>

  <h2>哪些组合有论文证据？</h2>
  <p class="legend"><span>● 原论文实证</span><span>△ 论文声称兼容但缺少二者受控联合实验</span><span>○ 机制可行、八篇未验证</span><span>⚠ 作用位置重叠，必须重新定义</span></p>
  <div class="table-wrap"><table class="matrix">
    <thead><tr><th>组合</th><th>证据级别</th><th>准确解释</th></tr></thead>
    <tbody>
      <tr><th>LoRA + QLoRA</th><td>● / 包含关系</td><td>QLoRA 本身就是 4-bit 冻结底座上的 LoRA。</td></tr>
      <tr><th>LoRA + DoRA</th><td>● / 包含关系</td><td>DoRA 用 LoRA 更新 direction；不是再并挂一份普通 LoRA。</td></tr>
      <tr><th>LoRA + MoLE</th><td>● / 结构组成</td><td>MoLE 在 attention 用普通 LoRA，在 FFN 用 token-level LoRA experts。</td></tr>
      <tr><th>QLoRA + DoRA = QDoRA</th><td>● 初步结果</td><td>DoRA 论文报告 Orca-Math 初步实验；证据规模小于主实验。</td></tr>
      <tr><th>LoRA + L2T</th><td>△</td><td>L2T 可包在 LoRA 外，但 L2T 主表没有 LoRA×L2T 受控消融。</td></tr>
      <tr><th>QLoRA + L2T</th><td>○</td><td>系统轴与目标轴可组合；QLoRA Table 10 的文本结果与 L2T 的视觉结果提醒我们必须重测，不能预设增益。</td></tr>
      <tr><th>ReFT + MoReS</th><td>⚠</td><td>都可能改同一 residual stream；必须规定先后顺序，最好用不重叠层/token。</td></tr>
      <tr><th>VL-Adapter + L2T</th><td>⚠</td><td>原 VL-Adapter 是 encoder–decoder 多任务生成；需先重新定义 instruction target serialization。</td></tr>
      <tr><th>DoRA + MoLE</th><td>⚠</td><td>需定义每个专家是否有独立 magnitude，以及归一化在路由前还是后。</td></tr>
    </tbody>
  </table></div>

  <h2>首要约束 → 方法选择</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>首要约束</th><th>优先考虑</th><th>原因</th><th>先检查什么</th></tr></thead>
    <tbody>
      <tr><td>默认通用 PEFT，部署要静态合并</td><th>LoRA</th><td>成熟、简单、任务权重小</td><td>target modules 往往比 rank 更关键</td></tr>
      <tr><td>单卡显存卡在大底座</td><th>QLoRA</th><td>NF4 把冻结权重常驻显存显著压低</td><td>activation、视觉塔、connector 仍可能是瓶颈</td></tr>
      <tr><td>同 rank LoRA 容量不够，仍需 merge</td><th>DoRA</th><td>幅值/方向解耦扩大有效更新族</td><td>训练归一化、显存与实现版本</td></tr>
      <tr><td>通用/文档/医学等混合后负迁移</td><th>MoLE</th><td>token 条件专家减少单一更新妥协</td><td>是否真稀疏、router 是否塌缩、域是否足够</td></tr>
      <tr><td>文本 LM、极端可训练参数预算</td><th>LoReFT</th><td>只在少数层/位置编辑表示</td><td>位置/层/rank 搜索成本和长生成退化</td></tr>
      <tr><td>MLLM 不看图、参数预算极低</td><th>MoReS</th><td>只 steering 视觉 token</td><td>image-token mask、层位、token 比例和 TP* 口径</td></tr>
      <tr><td>多个相近 V&amp;L 任务需共享容量</td><th>VL-Adapter</th><td>显式瓶颈和跨任务共享</td><td>现代 decoder-only 架构是否仍符合原结论</td></tr>
      <tr><td>instruction 本身含视觉语义，answer-only 走捷径</td><th>L2T</th><td>不改推理结构，直接改变监督</td><td>必须过滤 system/角色/高频任务模板</td></tr>
    </tbody>
  </table></div>

  <h2>十条最后检查</h2>
  <ol class="caveats">
    <li><strong>不要横向排名绝对分数：</strong>八篇的 backbone、数据、训练范围和参数分母不同。</li>
    <li><strong>LoRA 的低秩是 ΔW，不是 W₀：</strong>最终权重通常仍满秩。</li>
    <li><strong>“零时延”要写 merge 前提：</strong>动态 adapter 或同 batch 多任务不自动零开销。</li>
    <li><strong>QLoRA 的 4-bit 是存储：</strong>计算常在 BF16；33B/65B 未证明普遍追平 full FT。</li>
    <li><strong>DoRA 的零额外成本只指合并后推理：</strong>训练期归一化和 graph 有成本。</li>
    <li><strong>MoLE 的 top-1 不自动稀疏：</strong>不能先算所有专家再乘 one-hot。</li>
    <li><strong>ReFT 少参数不等于少搜索：</strong>层、位置、tie、rank 都会改变结果。</li>
    <li><strong>MoReS 的更高视觉 attention 是代理指标：</strong>不是因果证明，且论文公式有校勘问题。</li>
    <li><strong>VL-Adapter 不只训瓶颈：</strong>还训练 LayerNorm 与视觉投影。</li>
    <li><strong>L2T 零新增参数不等于 PEFT：</strong>实际训练量完全取决于底层方法。</li>
  </ol>

  <h2>复习题</h2>
  <div class="quiz">
    <details><summary>1. 为什么 QLoRA 不是一种新的低秩几何？</summary><p>它仍训练 LoRA 的 A、B；新增的是 NF4、Double Quantization、Paged Optimizer 与反量化计算路径。</p></details>
    <details><summary>2. 4096×4096 的线性层，rank 8 LoRA 有多少参数？</summary><p><code>8×(4096+4096)=65,536</code>，而全矩阵有 16,777,216。</p></details>
    <details><summary>3. DoRA 相比 LoRA 新增的核心自由度是什么？</summary><p>显式学习逐列 magnitude，低秩分支主要负责归一化 direction。</p></details>
    <details><summary>4. MoLE 怎样才有真正稀疏收益？</summary><p>top-1 后只执行命中的专家，并按专家对 token 分组；同时用负载均衡防止 router 塌缩。</p></details>
    <details><summary>5. 定义一个 ReFT intervention 至少要说明什么？</summary><p>干预函数 Φ、token 位置集合 P、Transformer 层 l。</p></details>
    <details><summary>6. ReFT 与 MoReS 的关键差别？</summary><p>ReFT 是通用层×位置表示干预框架；MoReS 只选择 MLLM 的视觉 token，目标是模态重平衡。</p></details>
    <details><summary>7. 哪些方法可静态合并？</summary><p>普通 LoRA 与 DoRA；MoLE、ReFT、MoReS、VL-Adapter 保留运行时结构。L2T 本身不新增推理结构。</p></details>
    <details><summary>8. 为什么 L2T 没有新增参数却不等于 PEFT？</summary><p>它只规定监督哪些 token；底层可以是全量训练，也可以是任意 PEFT。</p></details>
    <details><summary>9. QLoRA Table 10 与 L2T 是否矛盾？</summary><p>不矛盾。前者是纯文本四数据集的 source+target 消融，后者依赖视觉条件、过滤后的 informative instruction；它们说明联合方案必须验证。</p></details>
    <details><summary>10. 如何验证 QLoRA+L2T？</summary><p>做 16-bit/4-bit × answer-only/L2T 的 2×2，固定 backbone、数据、target modules 和步数，并同时报告精度、显存、吞吐与视觉贡献。</p></details>
  </div>

  <h2>原始论文</h2>
  <ol class="sources">
    <li><a href="https://openreview.net/forum?id=nZeVKeeFYf9">LoRA · ICLR 2022</a></li>
    <li><a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html">QLoRA · NeurIPS 2023</a></li>
    <li><a href="https://proceedings.mlr.press/v235/liu24bn.html">DoRA · ICML 2024</a></li>
    <li><a href="https://arxiv.org/abs/2401.16160">LLaVA-MoLE · arXiv:2401.16160</a></li>
    <li><a href="https://proceedings.neurips.cc/paper_files/paper/2024/hash/75008a0fba53bf13b0bb3b7bff986e0e-Abstract-Conference.html">ReFT · NeurIPS 2024</a></li>
    <li><a href="https://arxiv.org/abs/2412.12359">MoReS / LLaVA Steering · arXiv:2412.12359</a></li>
    <li><a href="https://openaccess.thecvf.com/content/CVPR2022/html/Sung_VL-Adapter_Parameter-Efficient_Transfer_Learning_for_Vision-and-Language_Tasks_CVPR_2022_paper.html">VL-Adapter · CVPR 2022</a></li>
    <li><a href="https://arxiv.org/abs/2503.22215">L2T · NeurIPS 2025 / arXiv:2503.22215</a></li>
  </ol>
</section>
"""


CSS = r"""
@page { size: A4; margin: 9mm 10mm; }
:root {
  --text:#202124; --muted:#60656f; --line:#d8dde5; --soft:#f6f7f9;
  --paper:#fff; --accent:#214f78; --blue:#eaf2f8; --green:#eaf5ef;
  --amber:#fbf3e9; --violet:#f0ecf8; --red:#fff0ef;
}
* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
body { margin:0; color:var(--text); background:#edf0f3; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",Arial,sans-serif; font-size:14px; line-height:1.52; }
main { max-width:1120px; margin:0 auto; padding:18px 22px 36px; background:var(--paper); }
.cover { border-bottom:2px solid var(--line); padding-bottom:12px; margin-bottom:14px; }
.cover h1 { margin:0 0 6px; font-size:28px; line-height:1.18; }
.cover p { margin:5px 0; color:var(--muted); }
.cover .lead { max-width:940px; font-size:15px; color:#38414b; }
.toc { background:var(--soft); border:1px solid var(--line); padding:10px 14px; margin:12px 0 18px; }
.toc h1,.toc h2 { margin-top:0; }
.toc h2 { font-size:17px; margin:12px 0 5px; border-bottom:1px solid var(--line); padding-bottom:3px; }
.toc ol { list-style:none; margin:4px 0 8px; padding:0; columns:2; column-gap:28px; }
.toc li { margin:2px 0; break-inside:avoid; }
a { color:#1a5d96; text-decoration:none; }
a:hover { text-decoration:underline; }
.paper { padding-top:18px; margin-top:20px; border-top:3px solid var(--line); break-before:page; }
.paper-kicker,.section-kicker { color:var(--accent); font-size:13px; font-weight:700; margin-bottom:3px; }
.paper-source { color:var(--muted); font-size:12px; margin-bottom:8px; }
h1 { font-size:24px; line-height:1.24; margin:0 0 6px; }
h2 { font-size:18px; margin:19px 0 8px; padding-bottom:4px; border-bottom:1px solid var(--line); }
h3 { font-size:15px; margin:12px 0 5px; }
p { margin:6px 0; }
ul,ol { margin:6px 0 8px 21px; padding:0; }
li { margin:3px 0; }
code { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.92em; overflow-wrap:anywhere; }
.meta { color:var(--muted); font-size:12px; margin-bottom:10px; }
.summary,.box,.note,.audit-note { border:1px solid var(--line); background:var(--soft); padding:8px 10px; margin:8px 0 12px; break-inside:avoid; border-radius:0; }
.summary { border-left:4px solid var(--accent); }
.note { background:#fbfcfe; }
.audit-note { border-left:4px solid #a75d00; background:var(--amber); }
.grid,.grid-2,.lead { display:grid; grid-template-columns:1fr 1fr; gap:10px 16px; }
.grid-3 { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }
.tag { display:inline-block; padding:1px 6px; border:1px solid var(--line); background:#eef4fb; font-size:12px; margin:2px 4px 2px 0; }
.small,.caption { color:var(--muted); font-size:12px; }
figure { margin:13px 0; padding:8px 0 10px; border-top:1px solid var(--line); break-inside:avoid; }
figure img { display:block; max-width:100%; max-height:90mm; width:auto; height:auto; margin:0 auto 5px; object-fit:contain; }
figcaption { text-align:center; color:var(--muted); font-size:12px; margin:4px 0 6px; }
.fig-explain { background:var(--soft); border:1px solid var(--line); padding:7px 9px; font-size:13px; }
.fig-explain p { margin:4px 0; }
.formula,.equation { margin:8px 0; padding:8px 10px; border-left:3px solid var(--accent); background:#f8fafc; font-family:"STIX Two Text","Times New Roman",serif; overflow-x:auto; }
table { border-collapse:collapse; width:100%; margin:8px 0; }
th,td { border:1px solid var(--line); padding:4px 6px; vertical-align:top; text-align:left; }
thead th { background:#eef3f7; color:#23445f; }
.table-wrap { overflow-x:auto; margin:8px 0 14px; }
.master-table { font-size:12px; }
.synthesis { scroll-margin-top:10px; padding:16px 0 8px; }
.synthesis.ending { break-before:page; border-top:4px double var(--line); margin-top:28px; }
.axis-map { display:grid; grid-template-columns:110px repeat(3,1fr); gap:8px; margin:12px 0; }
.axis-label { display:flex; align-items:center; justify-content:center; background:#293849; color:#fff; font-weight:700; padding:10px; }
.axis-card { min-height:92px; border:1px solid var(--line); padding:9px; display:flex; flex-direction:column; gap:3px; }
.axis-card b { font-size:15px; }.axis-card span { font-weight:700; color:var(--accent); }.axis-card small { color:var(--muted); }
.axis-card.weight { background:var(--blue); }.axis-card.representation { background:var(--violet); }.axis-card.module { background:var(--green); }
.axis-card.system { grid-column:2/3; background:var(--amber); }.axis-card.objective { grid-column:3/5; background:var(--red); }
.derivation { display:flex; gap:6px; align-items:stretch; overflow-x:auto; margin:9px 0; }
.derivation>div:not(.branch-note) { min-width:160px; flex:1; border:1px solid var(--line); background:#fff; padding:8px; }
.derivation b,.derivation span { display:block; }.derivation span { color:var(--muted); font-size:12px; }
.derivation i { align-self:center; color:#8290a0; font-style:normal; }
.derivation.second { align-items:center; }.branch-note { min-width:95px; color:var(--muted); font-size:11px; text-align:center; }
.legend { display:flex; flex-wrap:wrap; gap:5px 14px; font-size:12px; color:var(--muted); }
.mechanism-lab { margin:14px 0 20px; scroll-margin-top:10px; }
.mechanism-grid { display:grid; grid-template-columns:minmax(0,1.05fr) minmax(0,.95fr); gap:12px; align-items:start; }
.mechanism-legend { display:flex; flex-wrap:wrap; align-items:center; gap:5px 8px; margin:6px 0 9px; font-size:11px; color:var(--muted); }
.legend-chip { border:1px solid var(--line); padding:1px 6px; font-weight:650; color:#34404a; }
.legend-chip.frozen { background:#e8edf3; }.legend-chip.trainable { background:#e4f3e8; }
.legend-chip.selector { background:#fff0c9; }.legend-chip.objective { background:#eee7f7; }
.legend-note { margin-left:auto; }
.architecture-figure { margin:0; border:1px solid var(--line); padding:7px; background:#fff; }
.architecture-figure figure { margin:0; padding:0; border:0; }
.architecture-figure img { width:100%; max-height:165mm; }
.architecture-figure figcaption { text-align:left; font-weight:650; color:#394b5b; }
.worked-example { border:1px solid var(--line); border-left:4px solid #287249; background:#f8fbf9; padding:8px 10px; min-width:0; }
.worked-example h3 { margin-top:0; color:#214f3d; }
.example-note { color:var(--muted); font-size:12px; }
.calc-eq { margin:6px 0; padding:6px 8px; background:#fff; border:1px solid #d6e5dc; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:11.5px; line-height:1.48; overflow-wrap:anywhere; }
.calc-steps { margin:6px 0 8px 20px; }
.calc-steps li { margin:5px 0; }
.example-check { border-top:1px solid #cfe0d6; padding-top:6px; color:#355246; }
.calc-table { font-size:10.5px; background:#fff; }
.calc-table th,.calc-table td { padding:2px 4px; }
.matrix td:nth-child(2) { white-space:nowrap; font-weight:700; }
.quiz { display:grid; grid-template-columns:1fr 1fr; gap:6px 10px; }
.quiz details { border:1px solid var(--line); padding:6px 8px; background:#fbfcfd; break-inside:avoid; }
.quiz summary { cursor:pointer; font-weight:650; }
.sources { columns:2; column-gap:28px; }
@media (max-width:760px) {
  main { padding:14px; }.grid,.grid-2,.lead,.grid-3,.quiz { grid-template-columns:1fr; }
  .mechanism-grid { grid-template-columns:1fr; }.legend-note { margin-left:0; width:100%; }
  .axis-map { grid-template-columns:1fr; }.axis-card.system,.axis-card.objective { grid-column:auto; }
  .toc ol,.sources { columns:1; }.derivation { flex-direction:column; }.derivation i { transform:rotate(90deg); }
}
@media print {
  body { background:#fff; font-size:10px; line-height:1.34; }
  main { max-width:none; padding:0; }
  .cover h1 { font-size:18px; }.toc { padding:5px 7px; }.toc h2 { font-size:12px; margin:6px 0 2px; }
  .paper { padding-top:7px; margin-top:8px; }.paper-kicker,.paper-source,.section-kicker { font-size:8.8px; }
  h1 { font-size:16px; margin-bottom:3px; }h2 { font-size:12.5px; margin:9px 0 4px; padding-bottom:2px; }h3 { font-size:10.5px; margin:5px 0 2px; }
  p,li { margin-top:2px; margin-bottom:2px; }ul,ol { margin-top:2px; margin-bottom:4px; }
  .summary,.box,.note,.audit-note,.fig-explain { padding:4px 6px; }
  .grid,.grid-2,.lead,.grid-3 { gap:5px 8px; }
  figure { margin:6px 0; padding:3px 0 4px; }figure img { max-height:70mm; }figcaption { font-size:8.8px; margin:2px 0; }.fig-explain { font-size:9px; }
  .small,.caption,.meta { font-size:8.8px; }.master-table { font-size:8.5px; }.table-wrap { overflow:visible; }
  .axis-map { gap:3px; }.axis-card { min-height:0; padding:4px; }.axis-card b { font-size:10px; }.axis-card small,.derivation span { font-size:8px; }
  .mechanism-lab { margin:6px 0 9px; }.mechanism-grid { grid-template-columns:1.04fr .96fr; gap:5px; break-inside:avoid-page; }
  .mechanism-legend { font-size:7.8px; gap:2px 4px; margin:2px 0 4px; }.legend-chip { padding:0 3px; }.legend-note { margin-left:auto; width:auto; }
  .architecture-figure { padding:3px; }.architecture-figure img { max-height:145mm; }.architecture-figure figcaption { font-size:8px; }
  .worked-example { padding:4px 5px; border-left-width:2px; font-size:8.5px; line-height:1.34; }.worked-example h3 { font-size:9.5px; }
  .example-note { font-size:7.8px; }.calc-eq { padding:3px 4px; margin:3px 0; font-size:7.8px; line-height:1.32; }
  .calc-steps { margin:3px 0 4px 13px; }.calc-steps li { margin:2px 0; }.example-check { padding-top:3px; }
  .calc-table { font-size:6.8px; }.calc-table th,.calc-table td { padding:1px 2px; }
  .derivation>div:not(.branch-note) { min-width:0; padding:4px; }.branch-note { min-width:45px; font-size:7px; }
  .synthesis.ending { margin-top:0; }.quiz { gap:3px 5px; }.quiz details { padding:3px 4px; }
  thead { display:table-header-group; }tr,figure,.box,.note,.audit-note,details { break-inside:avoid; }
}
"""


def build(source_root: Path, out_dir: Path) -> Path:
    asset_dir = out_dir / "assets"
    if out_dir.exists():
        # The directory is generated and wholly owned by this script.  Remove only
        # its known generated children, never a broad workspace path.
        if asset_dir.exists():
            shutil.rmtree(asset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_manual_assets(source_root, asset_dir)
    render_architecture_diagrams(asset_dir)

    fragments: dict[str, str] = {}
    source_labels: dict[str, str] = {}
    for _category, slug, _label in PAPER_ORDER:
        if slug in MANUAL_CROPS:
            fragment = manual_fragment(slug)
            source_labels[slug] = "本讲义依据本地原始 PDF 新建完整章节"
        else:
            fragment, source_html = load_imported_fragment(source_root, slug, asset_dir)
            source_labels[slug] = f"复用并校勘原子页面：{source_html.name}"
        fragments[slug] = insert_architecture_lab(fragment, slug)

    toc_groups: list[str] = []
    articles: list[str] = []
    index = 0
    for category in dict.fromkeys(item[0] for item in PAPER_ORDER):
        items = [item for item in PAPER_ORDER if item[0] == category]
        lis: list[str] = []
        for _category, slug, label in items:
            index += 1
            lis.append(f'<li><a href="#paper-{index:02d}-{slug}">{index}. {html.escape(label)}</a></li>')
            articles.append(f"""
<article id="paper-{index:02d}-{slug}" class="paper">
  <div class="paper-kicker">{html.escape(category)} · {index}. {html.escape(label)}</div>
  <div class="paper-source">{html.escape(source_labels[slug])} · <a href="{OFFICIAL_LINKS[slug]}">官方论文</a></div>
  <div class="paper-content">{fragments[slug]}</div>
</article>
""")
        toc_groups.append(f'<section><h2>{html.escape(category)}</h2><ol>{"".join(lis)}</ol></section>')

    total_figures = sum(fragment.count("<figure") for fragment in fragments.values())
    total_images = sum(fragment.count("<img ") for fragment in fragments.values())
    total_explanations = sum(fragment.count('class="fig-explain"') for fragment in fragments.values())
    if not (total_figures == total_images == total_explanations):
        raise RuntimeError(
            f"Figure contract violated: figures={total_figures}, images={total_images}, "
            f"explanations={total_explanations}"
        )
    source_figures = total_figures - len(ARCHITECTURE_DOT_BODIES)

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>八篇 PEFT 与视觉指令微调论文精读讲义</title>
  <style>{CSS}</style>
</head>
<body>
<main>
  <section class="cover">
    <h1>从低秩权重更新到视觉表示转向：八篇论文精读讲义</h1>
    <p class="lead">完整覆盖 LoRA、QLoRA、DoRA、LLaVA-MoLE、ReFT、MoReS、VL-Adapter 与 L2T。按“三类论文 HTML 合并版”的紧凑 A4 风格编排，同时增加统一推导、证据强度、组合关系和校勘注；每篇另有一张结构定位图和一个可从输入复算到输出的完整小维度例子。</p>
    <p>阅读范围：8 份原始 PDF，共 194 页（含附录） · {source_figures} 张原论文图表 + {len(ARCHITECTURE_DOT_BODIES)} 张自绘结构图 · 每图均含“怎么看 / 说明什么 / 为什么重要或证据边界”解释。</p>
    <p class="small">生成脚本：<code>scripts/build_vlm_peft_eight_papers_lecture_html.py</code>。图表版权归各论文作者；本页用于个人研究学习与评论。</p>
  </section>
  <nav class="toc">
    <h1>目录</h1>
    <p><a href="#coordinate-system">导读：统一坐标系与总表</a> · <a href="#synthesis">综合：组合、选型、边界与复习</a></p>
    {"".join(toc_groups)}
  </nav>
  {front_matter()}
  {"".join(articles)}
  {closing_matter()}
</main>
</body>
</html>
"""
    destination = out_dir / "vlm_peft_eight_papers_lecture.html"
    destination.write_text(document, encoding="utf-8")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = build(args.source_root.resolve(), args.out_dir.resolve())
    print(destination)


if __name__ == "__main__":
    main()
