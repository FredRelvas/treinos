# PC-150 — Pipeline de Detecção de Objetos (Cheatsheet da Prova)

Pipeline modular para treino, avaliação e inferência de detectores YOLO
(via `ultralytics`) e Detectron2. Otimizada para Ubuntu 24 + RTX 4090
mas roda em Apple Silicon (MPS, via diretório `../MAC/`) e CPU.

> Todos os comandos abaixo assumem que você está dentro de `PC-150/`.

---

## 1. Setup (executar uma vez ao chegar no lab)

```bash
./setup.sh
```

O script **detecta todas as opções disponíveis** (Docker+NVIDIA, uv, pip) e
abre um menu interativo pra você escolher. Para pular o menu e pegar a
primeira opção viável (docker → uv → pip):

```bash
./setup.sh --auto
```

**Recomendação:** em prova com tempo limitado, prefira **uv** (1–3 min e iteração mais rápida).
Use Docker se o Detectron2 não compilar no host ou quiser isolamento total.

### ⚠️ ATIVAR O AMBIENTE — CRÍTICO

**O setup só prepara o ambiente. Antes de rodar QUALQUER comando, ative-o:**

| Rota escolhida | Como ativar |
|---|---|
| **Docker** | `docker compose exec pipeline bash` (entra no shell do container; todos os comandos abaixo rodam lá dentro) |
| **uv** | `source .venv/bin/activate` (no host, na pasta `PC-150/`) |
| **pip** | `source .venv/bin/activate` (idem) |

Sem ativar, você vai ver erros tipo `ModuleNotFoundError: torchvision` —
o Python do sistema não tem as deps; elas estão no venv/container.

Atalho pro Docker (sem entrar no shell):
```bash
docker compose exec pipeline python scripts/smoke_test.py --mode quick
```

### Detectron2 (opcional)

```bash
# Dentro do container Docker:  já vem instalado (se o build não falhou)
# uv:
uv pip install 'git+https://github.com/facebookresearch/detectron2.git'
# pip:
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

> Pipeline funciona sem Detectron2 — só desabilita `train_detectron2.py`.

---

## 2. Smoke Test rápido (validar ambiente)

```bash
python scripts/smoke_test.py --mode quick
```

Roda em ~30s em CPU, usa imagens sintéticas em `data/smoke/` (já commitadas).
Espera: `✅ Ambiente OK — pipeline pronta para uso`.

Se a saída pedir, regere o dataset sintético:

```bash
python scripts/generate_smoke_data.py
```

---

## 3. Receber dataset do professor

Arraste para `data/raw/` (já existe com subpastas `train/`, `val/`, `test/` prontas).
Formatos aceitos: **COCO JSON**, **YOLO TXT**, **Pascal VOC**.

**Veja `data/raw/README.md` para os 5 cenários típicos com exemplos de onde colocar cada arquivo.**

Cenário mais comum (prof manda 1 JSON único + 2 pastas):
```
data/raw/
├── train/   ← arraste imagens de treino aqui
├── test/    ← arraste imagens de teste aqui
└── labels.json   ← anotações COCO
```

Se vier validação também (`val/` + imagens), o `prepare_dataset.py` respeita o split do prof.
Sem `val/`, ele cria automaticamente 80/20 a partir do train.

---

## 4. Converter dataset → COCO JSON

```bash
python scripts/prepare_dataset.py \
    --input  data/raw/ \
    --output data/processed/ \
    --format auto
```

- `--format auto|coco|yolo|voc`
- Cria automaticamente split 80/20 train/val se `val/` não existir.
- Imprime resumo com contagem por classe.

---

## 5. Smoke Test completo (validar dataset)

```bash
python scripts/smoke_test.py --mode full --data data/processed/
```

Treina YOLOv8n por 3 épocas em 10% do dataset, gera plots e métricas
em `runs/exp_*_smoke_full/`. Falha rápido se algo no dataset estiver quebrado.

---

## 6. Treinar

### YOLO (preferido — rápido e robusto)

```bash
# Padrão balanced (RTX 4090: 50 épocas, batch 16, 70% do dataset)
python src/train_yolo.py --model yolov8s --data data/processed/

# Rápido: 30% do dataset, 15 épocas
python src/train_yolo.py --model yolov8s --data data/processed/ --preset fast --subset 0.3

# Full: 100 épocas, dataset inteiro, augmentation completa (mosaic + mixup)
python src/train_yolo.py --model yolov8m --data data/processed/ --preset full --name experimento_final

# Retomar treino interrompido
python src/train_yolo.py --resume runs/exp_20260517_143022_yolo_yolov8s/last.pt --epochs 30
```

### Detectron2 (se instalado)

```bash
python src/train_detectron2.py --model faster_rcnn --data data/processed/ --preset balanced
python src/train_detectron2.py --model retinanet   --data data/processed/ --preset full
```

> Modelos válidos: `faster_rcnn`, `retinanet`, `cascade_rcnn` (último exige máscaras).

---

## 7. Avaliar e gerar métricas

```bash
python scripts/inference.py \
    --model-path runs/exp_.../best.pt \
    --data       data/processed/test \
    --framework  yolo
```

Saídas em `runs/exp_.../inference/`:
- `metrics.json` (mAP@0.5, mAP@0.5:0.95, AP por classe, precision, recall, ms/imagem)
- `inference_report.json` (predição + GT por imagem)
- `plots/precision_recall.png`, `confusion_matrix.png`
- `plots/inference_samples/` (≥10 amostras anotadas: verde=pred, vermelho=GT)

---

## Problemas comuns

### `ModuleNotFoundError: torchvision` (ou qualquer outro pacote)
Você esqueceu de ativar o venv ou entrar no container. Veja a seção 1.

### `PermissionError: [Errno 13]` em `runs/` ou `data/`
Acontece no Docker quando o UID do container difere do UID do host. O
`docker-compose.yml` agora usa `user: "${HOST_UID}:${HOST_GID}"` e o
`setup.sh` exporta esses vars automaticamente. Se você subiu o container
**sem** passar pelo `setup.sh`, suba assim:

```bash
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up -d --build
```

Se já existem arquivos com ownership errada em `runs/`:
```bash
sudo chown -R $(id -u):$(id -g) runs/ data/
```

### Detectron2 não importa
Não foi instalado. É opcional — só afeta `train_detectron2.py`. Veja seção 1.

---

## Comandos de emergência

```bash
# Falta de tempo: subset minúsculo, 5 épocas
python src/train_yolo.py --model yolov8n --data data/processed/ --preset fast --subset 0.05 --epochs 5

# Sem GPU: força CPU (detect_hardware já faz isso, mas pode reduzir)
python src/train_yolo.py --model yolov8n --data data/processed/ --preset fast --batch 2 --epochs 3

# Retomar do último checkpoint salvo
python src/train_yolo.py --resume $(ls -t runs/*/last.pt | head -1)

# Desabilitar augmentation (debug)
python src/train_yolo.py --model yolov8n --data data/processed/ --no-augment --epochs 3

# Reinstalar Detectron2 do zero (CUDA)
uv pip install --force-reinstall 'git+https://github.com/facebookresearch/detectron2.git'
```

---

## Estrutura de saída de cada run

```
runs/exp_{TIMESTAMP}_{NOME}/
├── checkpoints/epoch_003.pt, epoch_006.pt, ...   ← a cada 3 épocas
├── best.pt                                       ← melhor mAP@0.5
├── last.pt                                       ← última época
├── config.json                                   ← hiperparâmetros usados
├── metrics.json                                  ← métricas finais
├── metrics_per_epoch.csv                         ← loss/mAP por época
└── plots/
    ├── loss_curve.png
    ├── map_evolution.png
    ├── precision_recall.png
    └── inference_samples/sample_001.png ...
```

---

## Presets disponíveis

| Preset    | CUDA              | MPS               | CPU              |
|-----------|-------------------|-------------------|------------------|
| fast      | 15 ep / batch 16  | 5 ep / batch 4    | 3 ep / batch 2   |
| balanced  | 50 ep / batch 16  | 20 ep / batch 8   | 10 ep / batch 2  |
| full      | 100 ep / batch 16 | 50 ep / batch 8   | 20 ep / batch 2  |

CLI sempre sobrescreve: `--epochs`, `--batch`, `--subset`, `--no-augment`.
