# Urdu → Roman Urdu NMT (BiLSTM, PyTorch)

## How to run (local)
1) Install deps
   pip install -r code/requirements.txt
2) Execute the notebook end-to-end (non-interactive):
   jupyter nbconvert --to notebook --execute code/urdu2roman_nmt.ipynb --output results/executed.ipynb
   # outputs (metrics, examples, checkpoints) are saved under results/

## How to run (Colab)
- Open: code/urdu2roman_nmt.ipynb → Runtime > Run all.
- GPU: T4/A100 (whichever available). Mentioned in results.pdf.

## Notes
- BPE/WordPiece tokenizer implemented from scratch (no external trainer).
- Model: BiLSTM encoder (2 layers), LSTM decoder (4 layers), attention.
- Metrics reported: BLEU, Perplexity, CER.

## Demo
- Public Streamlit: <PUT_URL_HERE>
