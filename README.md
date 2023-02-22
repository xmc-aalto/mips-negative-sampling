
Meta-classifier free negative sampling for extreme multilabel classification.

The code is adapted from the source codes of LightXML [1].

## Requirements

* tokenizers==0.7.0
* numpy==1.18.5
* pandas==1.0.4
* tqdm==4.46.1
* scipy==1.4.1
* transformers==2.11.0
* scikit_learn==0.23.2
* torch==1.5.1
* faiss-gpu
* apex

## Datasets

The datasets can be downloaded from the following links:
* [Amazon-670K](https://drive.google.com/open?id=1Xd4BPFy1RPmE7MEXMu77E2_xWOhR1pHW)
* [Wikipedia-500K](https://drive.google.com/open?id=1bGEcCagh8zaDV0ZNGsgF0QtwjcAm0Afk)

Please place the datasets in the data folder.

## Train and evaluation

Run the following commands for training and evaluation using MIPS-s method on Amazon-670 and Wikipedia-500K:
```bash
python src/main.py --epoch 20 --dataset amazon670k --swa --batch 16 --max_len 128 --hidden_dim 400 --model_type mips --num_neg_mips 5 --nlist 818 --nprobe_eval 350

python src/main.py --epoch 10 --dataset wiki500k --swa --batch 32 --max_len 128 --hidden_dim 500 --model_type mips --num_neg_mips 5 --nlist 707 --nprobe_eval 256
```

## References
[1] Jiang, Ting, et al., [Lightxml: Transformer with dynamic negative sampling for high-performance extreme multi-label text classification](https://ojs.aaai.org/index.php/AAAI/article/view/16974/16781), AAAI, 2021.
