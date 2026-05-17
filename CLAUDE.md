# CLAUDE.md — Contexto pro Claude Code

> Este arquivo é carregado automaticamente pelo Claude Code ao abrir o repo.
> Use pra dar contexto rápido em caso de erro durante a prova.

## Propósito do projeto

Pipeline de detecção de objetos pra **prova prática de Visão Computacional**.
Treina YOLO (ultralytics) ou Detectron2 em CUDA/MPS/CPU, com presets por hardware,
checkpoints automáticos, métricas COCO e plots. **Contexto = prova com tempo limitado**:
priorize soluções rápidas e robustas sobre perfeição.

## Arquitetura em 30 segundos

```
PC-150/                          ← repo git (esse), rodado no lab Ubuntu+RTX 4090
├── setup.sh                     ← menu interativo: docker / uv / pip
├── src/shared/                  ← código compartilhado (8 módulos)
│   ├── device_config.py         ← detecta cuda/mps/cpu → HardwareConfig
│   ├── presets.py               ← fast/balanced/full × hardware
│   ├── dataset.py               ← CocoDetectionDataset + DatasetIndex
│   ├── augmentation.py          ← pipelines off/light/standard/full
│   ├── checkpoints.py           ← RunPaths cria runs/exp_TS_NOME/
│   ├── evaluate.py              ← COCOeval wrapper (mAP, AP/classe)
│   ├── visualize.py             ← matplotlib plots (loss, mAP, PR, CM, samples)
│   └── utils.py                 ← seed, logging, paths
├── src/train_yolo.py            ← treina via ultralytics
├── src/train_detectron2.py      ← treina via Detectron2 (opcional)
├── scripts/
│   ├── smoke_test.py            ← --mode quick (CPU, ~30s) | --mode full (real)
│   ├── prepare_dataset.py       ← COCO/YOLO/VOC → COCO interno
│   ├── inference.py             ← métricas + plots finais
│   └── generate_smoke_data.py   ← regera data/smoke/
├── configs/                     ← YAMLs (presets, yolo, detectron2)
├── data/raw/{train,val,test}/   ← drop zone (gitkeeps versionados)
├── data/processed/              ← saída de prepare_dataset.py
└── data/smoke/                  ← 10 PNGs sintéticos commitados (smoke test)
```

**Repo MAC/** fica ao lado em desenvolvimento, mas **NÃO** está no lab. Lá só roda o PC-150.

## Fluxo padrão do dia da prova

```bash
git clone <repo> && cd PC-150
./setup.sh                                    # escolhe uv (mais rápido)
source .venv/bin/activate                     # ⚠️ ATIVAR SEMPRE
python scripts/smoke_test.py --mode quick     # valida ambiente (~30s)
# (recebe dataset → arrasta pra data/raw/{train,val,test}/ + labels.json)
python scripts/prepare_dataset.py --input data/raw/ --output data/processed/ --format auto
python scripts/smoke_test.py --mode full --data data/processed/
python src/train_yolo.py --model yolov8s --data data/processed/ --preset balanced --name run1
python scripts/inference.py --model-path runs/exp_*_run1/best.pt --data data/processed/test --framework yolo
```

---

## ⚠️ Erros conhecidos e como resolver

### `ModuleNotFoundError: torchvision` (ou qualquer outro pacote)
**Causa:** ambiente não ativado. Os pacotes estão no `.venv/` ou no container, não no Python do sistema.
**Fix:**
- Rota uv/pip: `source .venv/bin/activate`
- Rota Docker: `docker compose exec pipeline bash` (ou prefixe comandos com `docker compose exec pipeline`)

Confirma com `which python` — deve apontar pra `.venv/bin/python` ou `/opt/conda/bin/python` (container).

### `PermissionError: [Errno 13]` em `runs/` ou `data/`
**Causa:** UID do container ≠ UID do host. Acontece quando subiu o container sem `HOST_UID`/`HOST_GID`.
**Fix:**
```bash
docker compose down
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up -d --build
# Se já existem arquivos com ownership errada:
sudo chown -R $(id -u):$(id -g) runs/ data/
```
O `setup.sh` já exporta essas vars automaticamente — esse erro só ocorre se ele foi pulado.

### `ImportError: No module named 'detectron2'`
**Causa:** Detectron2 é **opcional**, não vem no setup padrão.
**Fix (se realmente precisar):**
```bash
uv pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
Compilação leva 5–15 min. Se falhar, **use YOLO** (`src/train_yolo.py`) — cobre os mesmos casos de uso com mAP comparável e zero hassle.

### Conversão de dataset falhou ou splits estranhos
**Sintoma comum:** `test: 0 imagens` ou todas as imagens no `train`.
**Causa:** layout do `data/raw/` não é o esperado pelo `prepare_dataset.py`.
**Debug:**
```bash
# Vê o que tem em raw/
find data/raw -maxdepth 3 -type f | head -20

# Roda com formato explícito ao invés de auto:
python scripts/prepare_dataset.py --input data/raw/ --output data/processed/ --format coco
```
Consulta `data/raw/README.md` pros 5 cenários suportados. Cenários que **funcionam**:
- 1 JSON único + pastas físicas `train/`, `test/` (split por pasta)
- JSONs separados (`train.json`, `val.json`, `test.json`)
- YOLO TXT em `train/labels/`, `train/images/`
- Pascal VOC com `Annotations/`

**Se cair em caso edge:** edite `scripts/prepare_dataset.py` (funções `_convert_coco`,
`_convert_yolo`, `_convert_voc`) ou processe os JSONs manualmente com `jq`/Python e
coloque direto em `data/processed/{train,val,test}/{images/,annotations.json}`.

### CUDA OOM (Out of Memory) no treino
**Causa:** batch grande demais ou modelo grande demais.
**Fix em ordem de tentativa:**
1. `--batch 8` (default é 16 em CUDA)
2. `--batch 4`
3. Troca modelo: `--model yolov8s` → `yolov8n`
4. `--preset fast` (reduz subset junto)

### `cuda` não detectado embora a 4090 esteja na máquina
**Causa provável:** `torch` instalado sem suporte CUDA (versão CPU).
**Verifica:**
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# Esperado: True 12.1
```
**Fix:**
```bash
uv pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Treino interrompido / kernel crash / SSH caiu
**Fix:** retoma do último checkpoint salvo.
```bash
python src/train_yolo.py --resume runs/exp_*/last.pt
# Ou pega o mais recente:
python src/train_yolo.py --resume $(ls -t runs/*/last.pt | head -1)
```
Checkpoints são salvos a cada 3 épocas + sempre o `best.pt` (melhor mAP@0.5) + `last.pt`.

### Smoke test `--mode quick` falha em download dos pesos
**Causa:** primeira execução baixa ResNet50 e YOLO pré-treinados (~100MB cada).
**Fix:** confirme conexão com internet. Em ambientes restritos, pré-baixe:
```bash
python -c "from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.DEFAULT)"
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### MAC: `BrokenPipeError` ou DataLoader trava
**Causa:** `num_workers > 0` causa deadlock no macOS com multiprocessing.
**Fix:** o `device_config.py` já força `num_workers=0` em MPS automaticamente.
Se editou e quebrou: confirme que `HardwareConfig.num_workers == 0` quando device é mps.

---

## Limitações conhecidas e soluções

### 1. Detectron2 não roda em MPS
**Limitação:** o backend MPS do PyTorch não suporta vários ops do Detectron2.
**Solução implementada:** `device_config.py` força `detectron2_device="cpu"` em Macs, com aviso.
**Workaround real:** use YOLO no Mac. Detectron2 no Mac via CPU é lento demais pra ser útil (~50× mais lento).

### 2. `cascade_rcnn` exige máscaras de segmentação
**Limitação:** o único cascade R-CNN no Detectron2 Model Zoo é `cascade_mask_rcnn_R_50_FPN_3x`,
que precisa de campos `segmentation` válidos nas anotações.
**Detecção:** `train_detectron2.py:154` checa e aborta cedo com mensagem clara se faltar.
**Workaround:** use `faster_rcnn` ou `retinanet` se só tem bboxes.

### 3. YOLO precisa converter COCO → YOLO TXT internamente
**Limitação:** `ultralytics` exige layout YOLO (1 .txt por imagem). Nosso dataset interno é COCO.
**Solução implementada:** `train_yolo.py:_coco_to_yolo_data_yaml()` gera `data/processed/_yolo/`
com symlinks (não duplica bytes) e `data.yaml`. Transparente pro usuário.
**Se symlinks falharem** (FAT/NTFS): cai em `shutil.copy2` automaticamente, mas dobra uso de disco.

### 4. Modelos suportados são fixos no `VALID_MODELS`
**Limitação:** `src/train_yolo.py:38` aceita só `yolov8n/s/m`; Detectron2 só 3 do Zoo.
**Workaround:** edite `VALID_MODELS` e adicione mais (`yolov8l`, `yolov8x`, `yolo11n`, etc).
Pesos são baixados automaticamente. Pra Detectron2, adicione entradas em
`configs/detectron2_config.yaml:model_zoo`.

### 5. Não suporta segmentação, pose, tracking, classificação
**Limitação:** pipeline é específica pra detecção de bboxes.
**Workaround:** fora do escopo. Pra outras tarefas, adapte:
- `src/shared/dataset.py` (formato dos targets)
- `src/shared/evaluate.py` (métricas diferentes — DICE, OKS, etc)
- `src/shared/visualize.py` (overlays diferentes)

### 6. `prepare_dataset.py` não cobre todos layouts possíveis
**Limitação:** detecta COCO/YOLO/VOC nos layouts mais comuns; pode falhar com layouts exóticos.
**Casos cobertos:** veja `data/raw/README.md` (5 cenários).
**Workaround pra layout não-padrão:** processe manualmente e coloque em
`data/processed/{train,val,test}/{images/,annotations.json}` no formato COCO.
A pipeline a partir daí funciona normalmente.

### 7. Métricas precision/recall globais usam stats do COCOeval
**Limitação:** `pycocotools` não expõe precision/recall globais isolados; só mAP e AR.
**Solução implementada:** `evaluate.py` usa `stats[8]` (AR@100) como proxy de recall e
`mAP@0.5:0.95` como precision (limitação documentada no código).
**Pra valores exatos:** as PR curves em `inference.py:_build_pr_curves()` calculam
precision/recall verdadeiros por classe — use esses se a métrica global for crítica.

### 8. Subset reprodutível é fração de imagens, não de anotações
**Limitação:** `--subset 0.3` pega 30% das imagens (com seed). Se as classes forem
desbalanceadas, o subset pode pegar zero anotações de uma classe minoritária.
**Workaround:** use `--subset` ≥ 0.5 em datasets com long-tail, ou estratifique manualmente.

### 9. Docker build inclui Detectron2 que pode falhar
**Limitação:** compilação de Detectron2 falha em algumas combinações CUDA/PyTorch.
**Solução implementada:** Dockerfile usa `|| true` — build continua mesmo se Detectron2 falhar.
**Verifica dentro do container:** `python -c "import detectron2; print(detectron2.__version__)"`.
Se faltar, instale manualmente dentro do container.

### 10. Early stopping no Detectron2 lança `StopIteration` no meio do treino
**Limitação:** Detectron2 não tem early stopping nativo; implementamos via hook que
levanta exceção. Funciona mas o stack trace pode parecer assustador.
**Solução implementada:** `train_detectron2.py` captura `StopIteration` em volta de `trainer.train()`.
**Se mexer:** mantenha o `try/except StopIteration`.

---

## Atalhos pra debugging com Claude Code

Quando algo der errado, dê esses comandos ao Claude:

```bash
# Estado do ambiente
which python && python --version && pip list 2>/dev/null | grep -iE "torch|ultralytics|detectron|coco"

# Estado das runs
ls -la runs/ 2>/dev/null | head && find runs -name "metrics.json" -exec echo {} \; -exec cat {} \;

# Estado do dataset
find data/raw data/processed -maxdepth 3 -type d 2>/dev/null
python -c "from src.shared.dataset import DatasetIndex; i = DatasetIndex.from_root('data/processed'); print('classes:', i.class_names(), 'splits:', bool(i.train), bool(i.val), bool(i.test))"

# Hardware visto
python -c "from src.shared.device_config import detect_hardware; print(detect_hardware())"
```

## Princípios pra resolver problemas

1. **Tempo é o recurso escasso.** Prefira workarounds simples a fixes elegantes.
2. **Nunca delete `runs/` durante a prova.** Mesmo runs falhas têm `config.json` que ajuda a debugar.
3. **Smoke test antes de qualquer treino sério.** 30 segundos investidos podem evitar 1h perdida.
4. **YOLO > Detectron2 em emergência.** Mais robusto, menos dependências, instala mais fácil.
5. **`--preset fast` + `--subset 0.1` + `--epochs 3`** é o mínimo viável pra ter QUALQUER resultado.
6. **`--resume` salva vidas.** Treino caiu? Não recomece, retome.
7. **Confie no `device_config`.** Não force device manualmente — ele já trata os edge cases.

## Estrutura de uma run de treino

```
runs/exp_{YYYYMMDD_HHMMSS}_{NOME}/
├── checkpoints/epoch_003.pt, 006.pt, ...   ← a cada 3 épocas
├── best.pt                                  ← melhor mAP@0.5 (usar pra inferência)
├── last.pt                                  ← última época (usar pra --resume)
├── config.json                              ← hiperparâmetros usados
├── metrics.json                             ← métricas finais
├── metrics_per_epoch.csv                    ← loss/mAP por época
├── weights/                                 ← (YOLO) saída raw do ultralytics
└── plots/
    ├── loss_curve.png, map_evolution.png
    ├── precision_recall.png, confusion_matrix.png
    └── inference_samples/sample_*.png       ← verde=pred, vermelho=GT
```

Após `inference.py`, adiciona-se `inference/` dentro da run com `metrics.json`,
`inference_report.json` e mais plots.
