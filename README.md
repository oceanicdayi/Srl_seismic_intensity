# Srl_seismic_intensity

Manuscript repository for **"Observation-Driven Second-by-Second Seismic-Intensity Forecasting as a Low-False-Alarm Complement to Source-Parameter Earthquake Early Warning in Taiwan"**, prepared for submission to *Seismological Research Letters*.

**Authors:** Da-Yi Chen and Yu-Hsuan Chang (Central Weather Administration, Taipei, Taiwan)

## Overview

This repository contains the manuscript describing the **Second-by-second Intensity Transformer Forecaster (SITF)**, an observation-driven model that forecasts a seismic station's final peak intensity directly from incomplete one-second intensity sequences — without using hypocenter, magnitude, rupture geometry, or a ground-motion prediction equation as model input.

SITF is proposed as a low-false-alarm confirmation and refinement layer that complements Taiwan's operational source-parameter earthquake early warning system (eBEAR) and observation-propagation methods such as PLUM, rather than replacing them.

## Key Results

- Trained on 198 Taiwan earthquakes; evaluated on 161 strictly nonoverlapping earthquakes (97,354 station-event samples per early window).
- Exact class accuracy rises from 56.4% (5 s window) to 79.4% (40 s window); off-by-one accuracy rises from 83.8% to 98.2%.
- At the national public-warning threshold of intensity ≥ 4:
  - **EW20:** 91.5% precision, 49.5% POD, F1 = 0.642, FAR = 0.38%
  - **EW40:** 96.4% precision, 82.5% POD, F1 = 0.889, FAR = 0.26%
- A propagation-only PLUM-like baseline achieves higher detection but substantially higher false-alarm rates (9.84–13.87%); target-station persistence is nearly perfectly precise but more conservative.
- An event-level sensitivity analysis suggests combining SITF with the operational eBEAR pathway can recover events missed by source-parameter alerting.

## Repository Contents

| File | Description |
|---|---|
| `SRL_manuscript_observation_driven_second_by_second_intensity.md` | Full manuscript (abstract, methods, results, discussion, tables, references) |
| `LICENSE` | Apache License 2.0 |

## Code and Data

The associated model implementation is maintained in a separate repository: [`oceanicdayi/Transformer_seismic_intensity`](https://github.com/oceanicdayi/Transformer_seismic_intensity).

Event-level raw files, operational eBEAR logs, and trained weights may be subject to Central Weather Administration data-distribution and operational-security policies.

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.
