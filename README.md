# Wazuh Sysmon Alert Triage with Machine Learning

This repository contains the full reproducible pipeline used in the study on reducing alert fatigue in Wazuh-based SOC environments using supervised machine learning.

## Overview
The project demonstrates how to:
- Build a unified and leakage‑resistant dataset from Wazuh Sysmon alerts and public Mordor datasets.
- Apply masking and preprocessing to avoid environment‑specific bias.
- Train and evaluate Logistic Regression, Random Forest, and XGBoost triage models.
- Integrate the trained model into a live Wazuh deployment using a lightweight enrichment service.
- Visualize model-assisted alerts inside OpenSearch dashboards.

## Repository Structure
```
src/                # Data prep, training, evaluation, integration scripts
data/               # Downloaded + processed dataset (or scripts to fetch/prepare it)
artifacts/          # Trained models, vectorizers, unified schema
configs/            # Wazuh, Filebeat, and OpenSearch configuration examples

## Integration into Wazuh
The `src/integration/ml_triage.py` script can be deployed on a Wazuh manager to enrich incoming Sysmon alerts with model predictions. Example Filebeat and OpenSearch configurations are provided in `configs/`.

## License
This project is released under an open-source license. Mordor-derived data follows original licensing terms.

## Citation
If you use this work, please cite the accompanying study.
