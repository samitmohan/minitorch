"""Generate growing architecture diagrams, one per milestone, as .excalidraw files."""
import json
from pathlib import Path

OUT = Path(__file__).parent / "diagrams"
OUT.mkdir(exist_ok=True)

NEW = ("#2f9e44", "#b2f2bb")
OLD = ("#1971c2", "#d0ebff")
FRAME = "#868e96"

# (milestone, label, x, y, w, h)
FRAMES = [
    (1, "examples/", 20, 40, 560, 110),
    (1, "minitorch/nn", 20, 170, 560, 150),
    (1, "core", 20, 340, 560, 90),
    (1, "backend", 20, 450, 560, 110),
    (2, "serve/  (tinyserve, torch first)", 620, 170, 340, 390),
    (5, "end-to-end apps", 620, 40, 340, 110),
]

# (milestone, id, x, y, w, h, label)
BOXES = [
    # examples
    (1, "ex_gpt", 40, 75, 120, 56, "train_gpt.py\n7M GPT"),
    (3, "ex_qwen", 175, 75, 120, 56, "qwen.py\ngenerate"),
    (3, "ex_lora", 310, 75, 120, 56, "lora.py\nfinetune"),
    (6, "ex_sd", 445, 75, 120, 56, "stable_\ndiffusion.py"),
    # nn row 1
    (1, "nn_rms", 40, 205, 96, 44, "RMSNorm"),
    (1, "nn_rope", 146, 205, 96, 44, "RoPE"),
    (1, "nn_gqa", 252, 205, 96, 44, "GQA +\nKV cache"),
    (1, "nn_swi", 358, 205, 96, 44, "SwiGLU"),
    (1, "nn_adam", 464, 205, 100, 44, "AdamW +\ncosine"),
    # nn row 2
    (1, "nn_bpe", 40, 262, 96, 44, "BPE\ntokenizer"),
    (3, "nn_st", 146, 262, 96, 44, "safe-\ntensors"),
    (3, "nn_lora", 252, 262, 96, 44, "LoRA"),
    (3, "nn_q", 358, 262, 96, 44, "int8 /\nint4"),
    (6, "nn_unet", 464, 262, 100, 44, "UNet /\nVAE"),
    # core
    (1, "core_t", 40, 370, 160, 46, "Tensor\nautograd"),
    (1, "core_m", 215, 370, 180, 46, "mlops.py\none gradient table"),
    (2, "core_d", 410, 370, 155, 46, "dtypes\nfp32 / bf16"),
    # backend
    (1, "be_np", 40, 485, 150, 56, "numpy.py\nreference, CPU"),
    (2, "be_mx", 200, 485, 150, 56, "mlx.py\nMetal GPU"),
    (2, "be_fa", 360, 485, 98, 56, "flash\nattn\nkernel"),
    (4, "be_pa", 467, 485, 98, 56, "paged\nattn\nkernel"),
    # serve
    (4, "sv_api", 640, 205, 300, 50, "FastAPI  /v1/chat/completions"),
    (4, "sv_eng", 640, 280, 300, 50, "Engine.step() + spec decode"),
    (2, "sv_sch", 640, 355, 300, 50, "Scheduler: continuous batching\n+ chunked prefill"),
    (2, "sv_kv", 640, 430, 300, 50, "Paged KV cache\n+ prefix cache"),
    (4, "sv_run", 640, 500, 300, 44, "ModelRunner on minitorch"),
    # apps
    (5, "ap_rag", 640, 75, 140, 56, "RAG"),
    (5, "ap_agent", 800, 75, 140, 56, "agent"),
]

# (milestone, id, x, y, points)
ARROWS = [
    (1, "a_ex_nn", 300, 150, [[0, 0], [0, 20]]),
    (1, "a_nn_core", 300, 320, [[0, 0], [0, 20]]),
    (1, "a_core_be", 300, 430, [[0, 0], [0, 20]]),
    (4, "a_api_eng", 790, 255, [[0, 0], [0, 25]]),
    (4, "a_eng_sch", 790, 330, [[0, 0], [0, 25]]),
    (2, "a_sch_kv", 790, 405, [[0, 0], [0, 25]]),
    (4, "a_kv_run", 790, 480, [[0, 0], [0, 20]]),
    (4, "a_run_nn", 640, 515, [[0, 0], [-58, -225]]),
    (5, "a_ap_api", 790, 131, [[0, 0], [0, 74]]),
]

TITLES = {
    1: "Milestone 1: core, tokenizer, GPT",
    2: "Milestone 2: Metal kernels + start serving",
    3: "Milestone 3: real weights, LoRA, quantization",
    4: "Milestone 4: the engine runs on minitorch",
    5: "Milestone 5: end-to-end apps",
    6: "Milestone 6: Stable Diffusion (optional)",
}


def text(id_, x, y, w, h, s, size=14, color="#1e1e1e", align="center"):
    return {"id": id_, "type": "text", "x": x, "y": y, "width": w, "height": h,
            "text": s, "fontSize": size, "fontFamily": 5, "textAlign": align,
            "verticalAlign": "middle", "strokeColor": color}


def boxtext(id_, x, y, w, h, s, size=13):
    """Excalidraw text ignores verticalAlign outside containers, so center by hand."""
    th = s.count("\n") * size * 1.25 + size * 1.25
    return text(id_, x, y + (h - th) / 2, w, th, s, size)


def scene(k):
    els = [text("title", 20, 0, 940, 30, TITLES[k], 22, align="left")]
    for m, label, x, y, w, h in FRAMES:
        if m > k:
            continue
        color = NEW[0] if m == k else FRAME
        els.append({"id": f"f_{label}", "type": "rectangle", "x": x, "y": y, "width": w,
                    "height": h, "strokeColor": color, "backgroundColor": "transparent",
                    "fillStyle": "solid", "strokeWidth": 1, "strokeStyle": "dashed",
                    "roughness": 0})
        els.append(text(f"fl_{label}", x + 8, y + 4, w - 16, 18, label, 13, color, "left"))
    for m, id_, x, y, w, h, label in BOXES:
        if m > k:
            continue
        stroke, fill = NEW if m == k else OLD
        els.append({"id": id_, "type": "rectangle", "x": x, "y": y, "width": w, "height": h,
                    "strokeColor": stroke, "backgroundColor": fill, "fillStyle": "solid",
                    "strokeWidth": 2, "strokeStyle": "solid", "roughness": 0})
        els.append(boxtext(f"t_{id_}", x, y, w, h, label))
    for m, id_, x, y, pts in ARROWS:
        if m > k:
            continue
        color = NEW[0] if m == k else "#495057"
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        els.append({"id": id_, "type": "arrow", "x": x, "y": y,
                    "width": max(xs) - min(xs), "height": max(ys) - min(ys),
                    "strokeColor": color, "strokeWidth": 2, "roughness": 0, "points": pts,
                    "endArrowhead": "arrow"})
    return {"type": "excalidraw", "version": 2, "source": "claude", "elements": els,
            "appState": {"viewBackgroundColor": "#ffffff"}}


POOL_Y = 170


def kv_scene():
    """Paged KV cache: two requests share prefix blocks through their block tables."""
    els = [text("title", 20, 0, 900, 30, "Paged KV cache with prefix sharing", 22, align="left")]
    els.append(text("pool", 700, POOL_Y + 18, 330, 20, "<- physical KV block pool", 13,
                    FRAME, "left"))
    colors = {0: NEW, 1: NEW, 2: OLD, 3: ("#f08c00", "#ffd8a8"), 4: ("#868e96", "#e9ecef"),
              5: ("#868e96", "#e9ecef")}
    labels = ["blk 0\nsystem", "blk 1\nprompt", "blk 2\nreq A", "blk 3\nreq B", "free", "free"]
    for i in range(6):
        s, f = colors[i]
        x = 20 + i * 110
        els.append({"id": f"b{i}", "type": "rectangle", "x": x, "y": POOL_Y, "width": 95,
                    "height": 56, "strokeColor": s, "backgroundColor": f, "fillStyle": "solid",
                    "strokeWidth": 2, "strokeStyle": "solid", "roughness": 0})
        els.append(boxtext(f"bt{i}", x, POOL_Y, 95, 56, labels[i]))
    for r, (name, blocks, y) in enumerate([("req A", [0, 1, 2], 50), ("req B", [0, 1, 3], 290)]):
        els.append(text(f"rn{r}", 20, y, 90, 40, f"{name}\nblock table", 13, align="left"))
        for j, b in enumerate(blocks):
            x = 130 + j * 70
            els.append({"id": f"r{r}{j}", "type": "rectangle", "x": x, "y": y, "width": 55,
                        "height": 40, "strokeColor": "#495057", "backgroundColor": "#ffffff",
                        "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
                        "roughness": 0})
            els.append(boxtext(f"rt{r}{j}", x, y, 55, 40, str(b), 16))
            tx = 20 + b * 110 + 47
            sy = y + 40 if r == 0 else y
            ty = POOL_Y if r == 0 else POOL_Y + 56
            dx, dy = tx - (x + 27), ty - sy
            els.append({"id": f"ra{r}{j}", "type": "arrow", "x": x + 27, "y": sy,
                        "width": abs(dx), "height": abs(dy), "strokeColor": "#495057",
                        "strokeWidth": 1, "roughness": 0, "points": [[0, 0], [dx, dy]],
                        "endArrowhead": "arrow"})
    els.append(text("note", 700, 40, 330, 90,
                     "Blocks 0 and 1 hold the shared\nprefix. Both requests point at\n"
                     "them, so prefill runs once.\nEach request owns only its tail.",
                     13, "#1e1e1e", "left"))
    return {"type": "excalidraw", "version": 2, "source": "claude", "elements": els,
            "appState": {"viewBackgroundColor": "#ffffff"}}


for k in TITLES:
    (OUT / f"m{k}.excalidraw").write_text(json.dumps(scene(k), indent=1))
(OUT / "kv.excalidraw").write_text(json.dumps(kv_scene(), indent=1))
print("wrote", sorted(p.name for p in OUT.glob("*.excalidraw")))
