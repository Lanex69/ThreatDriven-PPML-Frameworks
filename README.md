# Threat-Driven Frameworks for Privacy-Preserving Machine Learning (2017–2025)

*Authors:*  
Ayed Ahsan Ahmed, Mohammed Saad Shareef 
Lords Institute of Engineering and Technology, Hyderabad, India  

---

### 🧠 Overview
This repository accompanies the survey paper:

> *Threat-Driven Frameworks for Privacy-Preserving Machine Learning: A Practitioner’s Guide (2017–2025)*  
> Focused on mapping privacy threats (membership inference, model inversion, data extraction, poisoning, gradient leakage) to practical mitigation frameworks and decision matrices.

The repository provides all supplementary materials—benchmark tables, figures, framework metrics, and extended notes—to support reproducibility and practitioner reference.

---

### 📁 Repository Contents

| Folder | Description |
|--------|--------------|
| /paper | Paper outline, figures (flowchart, decision tree), and draft PDF. |
| /data | Benchmark data: attack-defense mapping and framework decision matrices. |
| /case-studies | Short applied notes (Healthcare, Edge, Finance, Smart City). |
| /extended-bibliography | Scholarcy-based summaries of 6 canonical threat papers. |

---

### 📊 Key Files

| File | Description |
|------|--------------|
| data/attack_defense_mapping.csv | Core dataset mapping threats → impacts → mitigations. |
| data/decision_matrix.csv | Comparative matrix of framework maturity, usability, and adoption. |
| paper/tables/Table1_Attack_Defense.md | Table used in Section 3 of the paper. |
| paper/figures/figure1_flowchart.svg | Visual mapping of threats to defense techniques. |
| case-studies/healthcare_FED-EHR.md | Privacy-preserving FL deployment in EHR settings. |

---

### 🧩 Methodology Summary
- *Data collection:* Canonical threat papers (2015–2024) via Scholarcy summarization.  
- *Framework mapping:* Evaluated frameworks include Opacus, TensorFlow Privacy, Flower, FATE, CrypTen, TenSEAL, Microsoft SEAL, and NVIDIA FLARE.  
- *Synthesis:* Mapped each framework’s defensive scope (DP, FL, SMPC, HE) to corresponding threat vectors.  
- *Decision matrix:* Built based on criteria—usability, scalability, and real-world adoption.

---

### 📈 How to Cite
If you use this repository, please cite:
@article{shareef2025threatdrivenppml,
title={Threat-Driven Frameworks for Privacy-Preserving Machine Learning: A Practitioner’s Guide (2017–2025)},
author={ Ahmed, Syed Ahsan and Shareef, Mohammed Saad},
year={2025},
journal={Preprint / Workshop Submission}
}

---

### 🧱 Future Work
This repository will expand with the experimental follow-up paper:
> *Secure and Private Fine-Tuning of LLMs with Parameter-Efficient Methods (DP + LoRA)*

That companion repo (tentative name: Privtune-DP-LoRA-Benchmark) will benchmark Opacus + Flower for differential-privacy fine-tuning in non-IID settings.

---

### 📜 License
This project is licensed under the *MIT* license.  
You are free to reuse, modify, or distribute the material with appropriate credit.

---

### 🤝 Contributing
Pull requests are welcome for:
- Adding new frameworks or metrics  
- Extending case studies  
- Correcting references or figures  

To contribute:
1. Fork this repository.  
2. Create a branch (add-framework-xyz).  
3. Submit a pull request with a short description.

---

### 🔗 Contact
For correspondence or collaboration:  
📧 [insert your academic email]
🌐 [LinkedIn: Mohammed Saad Shareef](https://www.linkedin.com/in/syed-ahsan-ahmed-17475a290/)
🌐 [LinkedIn: Mohammed Saad Shareef](https://linkedin.com/in/yourprofile)
