Sentiment Analysis with BERT and Gradio
This project uses BERT for sentiment analysis on reviews, trained on a CSV dataset, and deployed via Gradio for interactive use. It predicts Positive/Negative sentiment and offers actionable insights.

Specifications

Hardware
CPU: Multi-core (e.g., Intel i5/i7)
GPU: NVIDIA with CUDA (optional)
RAM: 16 GB min (32 GB recommended)
Storage: 10 GB free

Software
OS: Windows, macOS, Linux
Python: 3.8+
Dependencies: torch (CUDA if GPU), transformers, pandas, scikit-learn, openvino-dev[onnx], gradio, plotly, numpy, tqdm
Dataset: CSV with Text column (e.g., Reviews.csv)

Installation

Get Code

Save as sentiment_analysis.py or use Jupyter.

Set Up Environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

Install Libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
pip install transformers pandas scikit-learn openvino-dev[onnx] gradio plotly numpy tqdm --upgrade -qAdd Dataset

Place Reviews.csv in the directory or upload via Gradio.

Execution
Run
python sentiment_analysis.py

Train (Optional)
Trains BERT on Reviews.csv, saves to best_model/.

Convert Model (Optional)
mo --input_model sentiment_model.onnx --output_dir ./openvino_model --data_type FP16

Use Gradio
Launches interface:
Set sample size (100–10,000).
Upload CSV or use default.
Click Analyze for plot and report.

View Results

Bar graph: Positive vs. Negative.
Report: Metrics + recommendations.

Notes

GPU auto-detected; CPU fallback.
Gradio: share=True for public link, debug=True for logs
























