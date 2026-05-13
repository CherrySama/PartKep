# PartKep — Part-Aware Semantic Keypoint Manipulation

PartKep is a robot manipulation pipeline that uses **semantic part-aware priors (SAP)** combined with **VLM-driven constraint weighting** to guide a Franka Panda arm through pick-and-place tasks. Given a natural-language instruction and a scene image, the system identifies object parts (e.g. cup handle, rim, body), reasons semantically about how to grasp, and executes a full motion plan in MuJoCo simulation.

---

## Pipeline Overview

```
[Scene Image + Instruction]
         │
         ▼
 test/test_visual_pipeline.py
  GroundingDINO → SAM3 Segmentation → 2D Keypoints
  Output: images/results/pipeline_results.json
         │
         ▼
 Step 2A (local GPU)          Step 2B (remote server)
 test/test_vlm.py             server/vlm_server.py  ←── GPU server
                              server/vlm_client.py  ←── local machine
  Qwen3.5-9B VLM → Constraint Weights (w_grasp_axis, w_safety)
  Output: results/vlm_results.json
         │
         ▼
 demo_mujoco.py
  SAP + ConstraintInstantiator → PoseSolver → IKSolver → MotionPlanner
  Output: MuJoCo simulation execution
```

---

## Requirements

### Hardware
| Component | Requirement |
|-----------|-------------|
| GPU (VLM) | ≥ 12 GB VRAM (4-bit quantised Qwen3.5-9B) |
| GPU (Vision) | CUDA-capable GPU for GroundingDINO + SAM3 |
| CPU / RAM | Standard workstation |

> **Note:** Step 1 (visual pipeline) and Step 3 (MuJoCo demo) can run on a local machine with a modest GPU. Step 2 (VLM) requires ≥ 12 GB VRAM; use the server mode if your local machine does not meet this requirement.

### Software
- Python ≥ 3.10
- CUDA ≥ 12.8 (for PyTorch with GPU)
- MuJoCo ≥ 2.3.3
- `transformers` ≥ 5.0.0 (required for SAM3)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/PartKep.git
cd PartKep
```

### 2. Install dependencies

```bash
pip install -r requirement.txt
```

Install PyTorch with CUDA 12.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 3. Download model weights

```bash
python loadmodels.py
```

This downloads:
- **SAM3** (`facebook/sam3`, ~3.5 GB) → `models/sam3/`
- **GroundingDINO** (`IDEA-Research/grounding-dino-base`, ~1.5 GB) → `models/grounding-dino-base/`

For the VLM (Qwen3.5-9B, ~5–9 GB depending on quantisation), download separately from Hugging Face and place it at `models/Qwen3.5-9B/` (or update `MODEL_PATH` in `test/test_vlm.py`).

---

## Quick Start

### Step 1 — Visual Pipeline

Detects object parts and extracts 2D keypoints from a scene image.

```bash
python test/test_visual_pipeline.py
```

Edit the `TEST_CASES` list inside the script to point to your own images and instructions:

```python
TEST_CASES = [
    {
        "instruction": "pick up the cup",
        "image_path":  "images/cup3.jpg",
    },
    # optionally override GroundingDINO prompt:
    {
        "instruction":      "pick up the bottle",
        "detection_prompt": "brown bottle on the table",
        "image_path":       "images/kitchen.png",
    },
]
```

**Output:** `images/results/pipeline_results.json` and annotated images in `images/results/`.

---

### Step 2 — VLM Constraint Decision

Choose **one** of the two options below.

#### Option A — Local GPU

Run on a machine with ≥ 12 GB VRAM:

```bash
python test/test_vlm.py
```

Configure paths at the top of the file:

```python
MODEL_PATH   = Path("models/Qwen3.5-9B")   # path to local model weights
LOAD_IN_4BIT = True                         # recommended for 12 GB VRAM
RESULTS_JSON = Path("images/results/pipeline_results.json")
OUTPUT_JSON  = Path("results/vlm_results.json")
```

To run only a subset of instructions:

```python
VLM_FILTER = ["pick up the cup"]   # set to None to run all
```

---

#### Option B — Remote Server (SSH Tunnel)

Use this if your local machine lacks sufficient GPU memory.

**On the GPU server**, start the VLM HTTP server:

```bash
python server/vlm_server.py
```

Configure the model path inside `server/vlm_server.py`:

```python
MODEL_PATH   = "/workspace/models/Qwen3.5-9B"
LOAD_IN_4BIT = True
PORT         = 8000
```

**On your local machine**, open an SSH tunnel in a separate terminal:

```bash
ssh -L 8000:localhost:8000 <user>@<server-ip> -p <port> -N
```

Then send the pipeline results to the server:

```bash
python server/vlm_client.py
```

The client reads `images/results/pipeline_results.json`, sends requests to the server via the tunnel, and writes `images/results/vlm_results.json`.

**Output (both options):** `results/vlm_results.json`

---

### Step 3 — MuJoCo Simulation

Runs the full manipulation pipeline in MuJoCo: pose solving → IK → motion planning → execution.

```bash
python demo_mujoco.py
```

Configure the target instruction at the top of the file (must match an entry in `vlm_results.json`):

```python
INSTRUCTION = "pick up the cup"
SCENE_XML   = "assets/franka_emika_panda/scene.xml"
VLM_JSON    = Path("results/vlm_results.json")
```

---

## A Note on Object Realism

PartKep relies on GroundingDINO and SAM3 for part detection, both of which are trained on real-world imagery. For best results, use **photorealistic or scanned objects** in your scene. Simple geometric primitives (plain cylinders, boxes) may not be reliably detected from text prompts alone. If detection fails, try a more descriptive `detection_prompt` (see Step 1 configuration above).

---

## Project Structure

```
PartKep/
├── assets/
│   └── franka_emika_panda/     # Franka Panda MJCF model and scene XML
├── configs/
│   ├── SAP.py                  # Semantic part-aware priors knowledge base
│   ├── part_config.py          # Object → part name + SAM3 prompt mapping
│   ├── groundingdino_cfg.py    # GroundingDINO thresholds and model path
│   ├── sam3_cfg.py             # SAM3 model path and device config
│   └── camera_config.py        # Camera intrinsics / extrinsics
├── modules/
│   ├── taskParser.py           # Natural-language instruction → TaskSpec
│   ├── groundingdino.py        # Object detection (GroundingDINO wrapper)
│   ├── imageprocessor.py       # Bounding-box cropping utility
│   ├── sam3segmenter.py        # Part segmentation + keypoint extraction (SAM3)
│   ├── vlmDecider.py           # VLM constraint weight inference (Qwen3.5-9B)
│   ├── constraintsInst.py      # SAP → geometric constraints instantiation
│   ├── poseSolver.py           # SLSQP-based SE(3) pose optimisation
│   ├── IKSolver.py             # Inverse kinematics (ikpy)
│   └── motionPlanner.py        # Waypoint interpolation and pick-place planning
├── simulation/
│   └── mujoco_env.py           # MuJoCo environment wrapper and executor
├── server/
│   ├── vlm_server.py           # FastAPI VLM server (run on GPU machine)
│   ├── vlm_client.py           # HTTP client (run on local machine)
│   └── requirement.txt         # Server-side dependencies
├── test/
│   ├── test_visual_pipeline.py # Experiment 1: visual pipeline
│   ├── test_vlm.py             # Experiment 2: VLM constraint decision
│   └── test_pick_pose.py       # Unit test: pick pose computation
├── images/                     # Input images and annotated results
├── results/                    # JSON outputs (pipeline_results, vlm_results)
├── models/                     # Downloaded model weights (not tracked by git)
├── demo_mujoco.py              # Full pipeline demo in MuJoCo
├── loadmodels.py               # Model download script
└── requirement.txt             # Project dependencies
```

---

## Configuration Reference

| File | Key Parameters |
|------|---------------|
| `configs/SAP.py` | Per-part approach direction, grasp axis, contact mode, safety margin |
| `configs/part_config.py` | Object → part name and SAM3 text prompt mapping |
| `configs/groundingdino_cfg.py` | `BOX_THRESHOLD`, `TEXT_THRESHOLD`, `NMS_THRESHOLD` |
| `configs/sam3_cfg.py` | `LOCAL_MODEL_PATH`, `DEVICE` |
| `configs/camera_config.py` | Camera intrinsics (`fx`, `fy`, `cx`, `cy`) and extrinsic matrix |

To add support for a new object type, update both `configs/part_config.py` (add the object label and its part prompts) and `configs/SAP.py` (add SAP entries for any new part names).

---

## Citation

If you use PartKep in your research, please cite:

```bibtex
@mastersthesis{ho2026partkep,
  author = {Yinghao He},
  title  = {PartKep: Part-Aware Semantic Keypoint Manipulation for Robot Pick-and-Place},
  school = {University of Nottingham},
  year   = {2026},
}
```

---
