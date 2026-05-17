# data/raw/ — Drop zone do dataset do professor

Arraste aqui o que o prof enviar. Depois rode:

```bash
python scripts/prepare_dataset.py --input data/raw/ --output data/processed/ --format auto
```

O script detecta o formato automaticamente, valida, e escreve em
`data/processed/` no layout interno (sempre COCO JSON em `train/val/test/`).

---

## Cenários comuns — onde colocar cada arquivo

### Cenário 1 — Prof manda 1 JSON único + pastas físicas (mais comum)

```
data/raw/
├── train/img001.jpg, img002.jpg, ...
├── test/img050.jpg, img051.jpg, ...
└── labels.json    ← 1 arquivo com TODAS as anotações
```

O `prepare_dataset.py` usa as pastas físicas pra determinar o split.
Imagens em `train/` viram split train, em `test/` viram split test.

### Cenário 2 — Prof manda JSONs separados por split

```
data/raw/
├── train/imagens.jpg... + train.json
├── val/imagens.jpg...   + val.json     (opcional)
└── test/imagens.jpg...  + test.json
```

Cada JSON descreve seu split. Não precisa do `labels.json` raiz.

### Cenário 3 — Sem validação (só train + test)

```
data/raw/
├── train/...
├── test/...
└── labels.json   (ou train.json + test.json)
```

`prepare_dataset.py` cria automaticamente um split val sintético tirando
20% do train (com seed reprodutível). O test fica intocado.

### Cenário 4 — Formato YOLO TXT

```
data/raw/
├── train/images/*.jpg
├── train/labels/*.txt   ← 1 .txt por imagem (cls cx cy w h normalizados)
├── val/images/, val/labels/
└── classes.txt          (ou data.yaml com names: [...])
```

### Cenário 5 — Formato Pascal VOC

```
data/raw/
├── train/*.jpg + *.xml  (ou train/images/ + train/Annotations/)
└── test/...
```

---

## Dica: se não tiver certeza do formato

Roda o script — ele tem `--format auto` e imprime no final um resumo com
contagem de imagens por split e anotações por classe. Se a contagem
estiver estranha (ex: test=0), volta aqui e confere o layout.

```bash
python scripts/prepare_dataset.py --input data/raw/ --output data/processed/ --format auto
```

Output esperado:
```
============================================================
Dataset convertido em data/processed
------------------------------------------------------------
  treino:    240 imagens, 612 anotações
  validação: 60  imagens, 153 anotações
  teste:     75  imagens, 189 anotações
------------------------------------------------------------
  classes:
    [  1] cat                       312 anotações
    [  2] dog                       298 anotações
    [  3] bird                      344 anotações
============================================================
```
