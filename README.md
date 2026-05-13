# ventilator_pdm

Reference implementation accompanying the MSc thesis _Predictive Maintenance for Critical Care Ventilators Using Multivariate Anomaly Detection_ (Erik Hafnor, University of Stavanger, 2026).

This repository contains the source code for the data-ingestion pipeline, feature engineering, anomaly detection models, evaluation framework, explainability layer, and deployment service described in the thesis. The thesis manuscript is delivered separately through the university's formal submission system.

## Installation

```
pip install -e .
```

Requires Python 3.10 or newer.

## Repository layout

| Path | Contents |
|---|---|
| `ventilator_pdm/ingestion/` | HL7 MLLP receiver, parser, and QuestDB writer for live ingestion of Elisa 800 telemetry |
| `ventilator_pdm/features.py` | Feature pipeline: resample-then-pivot, derived features, sliding windows |
| `ventilator_pdm/extract.py` | QuestDB data extraction to Parquet fleet files |
| `ventilator_pdm/variables.py` | Variable-ID constants, BITFIELDS, and feature column lists |
| `ventilator_pdm/registry.py` | Fleet registry: CMMS-to-telemetry serial mapping, known failure events, PV events |
| `ventilator_pdm/models/` | Isolation Forest, One-Class SVM, and CNN-LSTM autoencoder implementations |
| `ventilator_pdm/evaluation.py` | Detection metrics and lead-time computation against known failure events |
| `ventilator_pdm/evaluation_pv.py` | Preventive-maintenance negative-control evaluation |
| `ventilator_pdm/xai.py` | SHAP TreeExplainer utilities for the Isolation Forest |
| `ventilator_pdm/inference.py` | Operational inference engine: score latest telemetry and generate alerts |
| `ventilator_pdm/service/` | FastAPI deployment service: scheduled inference, CMMS feedback adapter, SQLite alert store, web dashboard |
| `figures/` | Figure-generator scripts (Cypresses design system) used to produce the figures in the thesis |
| `examples/sample-hl7.txt` | One synthetic HL7 v2.x message illustrating the OBX structure consumed by the parser |
| `tests/` | Pytest suite for feature engineering, models, registry, and service layer |

## Running the tests

```
pip install -e ".[test]"
pytest
```

## Data

The original telemetry data analysed in the thesis is covered by a confidentiality agreement with Helse Stavanger HF and is not redistributed in this repository. The HL7 OBX-field data is technical-only per the gateway vendor's documentation and contains no patient-identifying information.

The C-MAPSS turbofan benchmark used for cross-domain validation in the thesis is publicly available from the [NASA Prognostics Center of Excellence Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

## License

MIT. See [LICENSE](LICENSE).
