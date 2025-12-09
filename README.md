# Threat-Driven Frameworks for Privacy-Preserving Machine Learning (2017–2025)

**Author:** Mohammed Saad Shareef, Syed Ahsan Ahmed  
Lords Institute of Engineering and Technology, Hyderabad, India  

📄 **One-Page Research Summary**  
For a quick overview, reviewers and professors can read the concise 1-page summary:  
👉 **[Download Summary (PDF)](paper/Research-Summary.pdf)**


---

##  Overview

This repository accompanies the preprint:

**_Benchmarking the Practical Adoption of Privacy-Preserving Machine Learning:  
A Tool-Centric and Framework-Oriented Review (2017–2025)_**

The repository contains all supplemental material required to **reproduce the DP-SGD experiment (Appendix A)**, generate figures, and reference framework-selection tools such as the **practitioner decision tree** and **threat→defense map**.

It is designed for:

- Researchers evaluating PPML tools (Opacus, TF-Privacy, FATE, FLARE, CrypTen, SEAL, Flower)  
- Practitioners comparing DP, FL, SMPC, HE, and hybrid frameworks  
- Students studying privacy–utility tradeoffs  
- Reviewers verifying reproducibility

---

## Repository Structure

| Path | Description |
|------|-------------|
| `run_privacy_accounting.py` | Canonical DP-SGD MNIST experiment (Appendix A). |
| `plot_epsilon_vs_accuracy.py` | Minimal IEEE-style privacy–utility curve generator. |
| `notebooks/privacy_accounting.ipynb` | Interactive demo (quick mode) calling the script. |
| `figures/` | All final paper figures (threat→defense map, decision tree, Appendix plot). |
| `results/` | Processed CSV results used in figures (`eps_results_final.csv` or demo CSV). |
| `paper/` | LaTeX source + preprint version (replace with final PDF). |
| `decision_matrix.csv` | Practitioner-oriented comparison of PPML frameworks (DP, FL, HE, SMPC). |
| `requirements.txt` | Reproducible environment for all scripts and notebooks. |

Raw datasets (MNIST) are **not** committed and are downloaded automatically via `torchvision`.

---

## Quick Start (Reproduce Appendix A)

### 1 Install dependencies
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

### 2 Run the DP-SGD experiment (quick demo)
This runs 1 epoch, 1 seed, sigma=1.0 → fast reproducibility check.

python run_privacy_accounting.py --quick --out results/eps_results_demo.csv

### 3 Generate the Privacy–Utility Curve
python plot_epsilon_vs_accuracy.py

Output saved to:
figures/epsilon_vs_accuracy.png
figures/epsilon_vs_accuracy.svg

### 4 Full reproduction (slower; full Appendix A)
python run_privacy_accounting.py --epochs 15 --batch 128 --out results/eps_results_final.csv
python plot_epsilon_vs_accuracy.py


## Decision Matrix (Practitioner Tool)

decision_matrix.csv compares major PPML frameworks across:

- Threat coverage (MIAs, inversion, extraction, leakage, poisoning)
- Techniques (DP, FL, SMPC, HE, hybrid)
- Maturity + ecosystem support
- Performance + scalability
- Real-world adoption
- Documentation & ease-of-use

Each entry includes a provenance column clarifying whether values come from:

- paper experiments (paper_run)
- literature surveys (literature)
- framework documentation (tool_doc)
- external references (repo_link)


## Experimental Reproducibility

This repository includes:

- Deterministic seeds for all runs
- CSV logs for DP-SGD experiments
- Full plotting script (IEEE-style)
- A lightweight demo notebook
- Clear environment + version pinning

Hardware used for primary results:

- GPU: Tesla T4 / RTX series
- PyTorch ≥ 2.0
- Opacus ≥ 1.1

## Citation

If you use this repository, please cite:

@article{shareef2025ppmlsurvey,
  title={Benchmarking the Practical Adoption of Privacy-Preserving ML: A Tool-Centric and Framework-Oriented Review (2017--2025)},
  author={Ahmed,Syed Ahsan and Shareef,Mohammed Saad},
  year={2025},
  journal={Preprint/ Workshop Submission},
}

CITATION.cff is also provided.

## License

This project is released under the MIT License.
You are free to reuse, modify, and distribute with attribution.

## Contributing

Pull requests are welcome for:

- Extending the decision matrix
- Adding new PPML frameworks
- Improving plotting or reproducibility tools
- Reporting errors in the threat→defense mappings

## Contact

For collaboration or research inquiries:
- [**Syed Ahsan Ahmed**](https://www.linkedin.com/in/syed-ahsan-ahmed-17475a290/)
- [**Mohammed Saad Shareef**](https://www.linkedin.com/in/mohammed-saad-shareef-019397265/)
