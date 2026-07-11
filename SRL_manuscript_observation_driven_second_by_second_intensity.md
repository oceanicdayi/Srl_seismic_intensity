# Observation-Driven Second-by-Second Seismic-Intensity Forecasting as a Low-False-Alarm Complement to Source-Parameter Earthquake Early Warning in Taiwan

**Running title:** Observation-driven intensity forecasting for Taiwan EEW  
**Authors:** Da-Yi Chen^1,* and Yu-Hsuan Chang^1  
**Affiliation:** ^1 Central Weather Administration, Taipei, Taiwan  
**Corresponding author:** Da-Yi Chen  
**Target journal:** *Seismological Research Letters*  
**Article type:** Regular Article

## Abstract

Earthquake early warning (EEW) systems commonly estimate origin time, hypocenter, and magnitude before converting those source parameters into expected shaking. This architecture provides rapid regional coverage, but errors in phase association, location, magnitude, or rupture representation can propagate into alert decisions, particularly for offshore or complex earthquakes. We evaluate an observation-driven complement that forecasts final station-level seismic intensity directly from incomplete one-second intensity sequences, without using hypocenter, magnitude, source model, or ground-motion prediction equation as model input. The Second-by-second Intensity Transformer Forecaster (SITF) ingests the first 5–40 s of intensity observations at each station and predicts the final peak intensity class and continuous intensity value; separate checkpoints are trained for each window length. Trained on 198 Taiwan earthquakes and evaluated on 161 strictly nonoverlapping earthquakes (97,354 station-event samples per early window), SITF reached 91.5% precision at 20 s with a false-alarm rate of 0.38% at the national public-warning threshold of intensity 4 or greater, and 96.4% precision, 82.5% probability of detection, and an F1 score of 0.889 at 40 s. A strict propagation-only PLUM-like baseline that excluded the target station attained higher probability of detection but substantially lower precision (35.2–35.7%) and higher false-alarm rates (9.84% and 13.87% at 20 and 40 s, respectively). Persistence of the observed target-station intensity was nearly perfectly precise but detected fewer eventual exceedances. An event-level sensitivity analysis of 160 matched earthquakes further shows that combining SITF with the operational eBEAR pathway recovers events missed by source-parameter alerting alone. SITF is best interpreted not as a replacement for source-parameter EEW or for multi-station deep-learning alerting models, but as a low-false-alarm, source-parameter-independent confirmation and refinement layer suited to a layered warning architecture alongside eBEAR and PLUM-like propagation.

**Keywords:** earthquake early warning; seismic intensity; Transformer; observation-driven forecasting; PLUM; eBEAR; Taiwan

## Introduction

Earthquake early warning (EEW) is an inference problem under extreme time constraints. Operational systems must decide which locations are likely to experience damaging shaking before the strongest motion arrives, while using observations that are incomplete, noisy, and spatially uneven. Many regional EEW systems address this task through a source-parameter-first workflow: detect an event, associate phase picks, estimate origin time and hypocenter, estimate magnitude, and then convert the evolving source solution into predicted ground motion or seismic intensity. This architecture is indispensable because it can provide broad-area information before strong shaking is observed at distant sites, and it underlies mature operational systems worldwide, including the west-coast U.S. ShakeAlert system (Kohler et al., 2020).

Taiwan's Earthworm-Based Earthquake Alarm Reporting system (eBEAR) represents this mature operational paradigm. It combines real-time waveform acquisition, P-wave picking and association, rapid source estimation, alert filtering, and dissemination. Its principal strength is speed: once a stable event solution is available, warnings can be issued to areas that have not yet recorded strong motion. However, the earliest source solution is necessarily based on only the initial portion of the rupture. Offshore earthquakes, extended or multi-segment ruptures, magnitude saturation and scaling error (Kuyuk and Allen, 2013), location errors, and local site amplification can therefore cause the intensity inferred from early source parameters to differ from the shaking that is ultimately observed. These limitations do not invalidate source-parameter EEW; rather, they motivate independent observational pathways that can confirm, correct, or refine the alert as the shaking field develops.

Observation-based EEW methods reduce dependence on source parameters. The Propagation of Local Undamped Motion (PLUM) family, for example, estimates the likely shaking at a target site from strong motion already observed at neighboring stations. Because it uses the wavefield directly, PLUM is robust to source-model errors. Its unavoidable limitation is latency: a nearby station must first experience and report sufficiently strong shaking, and aggressive spatial propagation can lead to over-warning. A still more conservative observational baseline is target-station persistence, in which the final intensity is assumed to equal the maximum intensity already observed at the target. Persistence has very few false alarms, but it cannot warn a station before that station itself has experienced strong shaking.

A parallel line of work applies deep learning directly to real-time ground-motion estimation, and this literature must be situated relative to the present study. Convolutional networks have been trained to map raw multi-station waveforms to gridded ground-motion intensity (Jozinovic et al., 2020), and transformer-based multi-station architectures such as the Transformer Earthquake Alerting Model (TEAM) use cross-station attention to jointly estimate location, magnitude, and a per-station warning decision directly from waveforms (Münchmeyer et al., 2021), in some settings improving on the speed and accuracy of traditional source-parameter pipelines (Meier, 2017). These approaches remain source-centric in the sense that they estimate, or implicitly encode, the earthquake source so that a warning can be generalized to stations that have not yet recorded any shaking. SITF occupies a different point in this design space. It is single-station rather than multi-station, it does not attempt to recover location, magnitude, or any other source description even implicitly, and it operates on a compact one-second intensity sequence rather than raw multi-channel waveforms. This narrower scope trades away the ability to warn a station before it has recorded any shaking — an ability that requires some form of source or cross-station information — in exchange for a forecast whose errors cannot be traced to a misestimated source, and for a substantially simpler single-channel input representation.

This study investigates an intermediate operating point. We ask whether a learned temporal model can use the partial evolution of one-second seismic intensity at a station to forecast the station's eventual peak intensity. The proposed Second-by-second Intensity Transformer Forecaster (SITF) is source-parameter independent at inference: hypocenter, magnitude, rupture geometry, and ground-motion prediction equations are not model inputs. It also does not require a separate station-level amplitude threshold before producing an estimate; zero-intensity background samples and weak early observations remain valid input. In the present retrospective experiment, windows are referenced to catalog origin time, so SITF should not be described as an autonomous event detector. Its trigger independence is limited to the forecasting layer: once a continuous event-relative sequence is available, the model can update a forecast every second without waiting for the target station or a neighboring station to cross a strong-motion threshold.

The scientific objective is not to show that SITF uniformly outperforms eBEAR, PLUM, persistence, or multi-station deep-learning architectures such as TEAM. These methods use different information and occupy different positions on the warning-time–accuracy frontier. Instead, we test three operationally relevant hypotheses: (1) incomplete one-second intensity sequences contain information about final station shaking beyond the current observed maximum; (2) at the national public-warning threshold of intensity 4 or greater, SITF can provide a useful low-false-alarm operating point; and (3) an observation-driven forecast can complement source-parameter EEW by reducing vulnerability to source-estimation errors and by identifying events or station outcomes that a single pathway may miss.

## Data and Methods

### Data Set and Experimental Design

The data consist of event-level files containing one-second seismic-intensity sequences for stations in Taiwan. Each event record includes time samples, station identifiers, intensity sequences, epicentral-distance metadata, and earthquake information. Station metadata contain 639 station records with station code, latitude, longitude, elevation, administrative area, and station name.

The training archive contains 198 earthquake files. Among the 197 events with complete earthquake metadata, magnitudes range from 3.02 to 7.20 and focal depths range from 4.26 to 250.40 km. The original evaluation archive contains 208 events. To prevent event-level leakage, 47 event identifiers appearing in both archives were removed, leaving 161 strictly nonoverlapping earthquakes for the primary independent evaluation. Each early-window evaluation contains 97,354 station-event samples before method-specific common-valid filtering. Common-valid filtering restricts a given comparison to the subset of station-event samples for which every method being compared can produce a value; for example, PLUM-R30 requires at least one non-target station within 30 km to report an epicentral distance and intensity, and the eBEAR sensitivity analysis requires the event to appear in the online eBEAR log. Sample counts therefore vary slightly across comparisons and are reported with the corresponding table or figure.

Each supervised sample is an `(earthquake, station)` pair. The input is the target station's one-second intensity sequence during an early window of length 5, 10, 15, 20, 25, 30, 35, or 40 s, denoted EW05–EW40. The prediction target is the maximum station intensity over the complete event sequence. The ordered classes are 0, 1, 2, 3, 4, 5−, 5+, 6−, 6+, and 7. Separate model checkpoints are trained for each window length. This formulation is a station-level temporal forecasting problem; the present implementation does not use simultaneous multi-station input, cross-station attention, station coordinates, hypocentral distance, azimuth, geology, or source parameters during the forward pass.

The primary population includes all valid stations, including stations with no measurable shaking during the early window. This choice is intentionally stringent. Removing stations that have not yet received shaking would eliminate the most difficult cases and would overstate performance for a system intended to operate continuously across a dense network.

### Model Architecture and Training

SITF uses a compact temporal convolutional embedding followed by a Transformer encoder. The architecture is inspired by general temporal-encoder designs but does not use pretrained speech weights or a speech-specific front end. A representative configuration uses a hidden size of 192, four Transformer encoder layers, four attention heads, an intermediate size of 384, and dropout of 0.1. EW05 uses a one-layer convolutional stem with kernel size 3 and unit stride. EW10–EW40 use three unit-stride convolutional layers with kernel sizes 3, 3, and 3. The valid encoded frames are pooled and passed to two heads: a 10-class classification head and a continuous-intensity regression head.

The multitask objective is

`L = 0.3 L_CE + 0.7 L_SmoothL1`,

where `L_CE` is cross entropy for the ordered intensity class and `L_SmoothL1` is the regression loss for continuous intensity. The representative training configuration uses AdamW, a learning rate of 3 × 10−4, weight decay of 10−2, warmup ratio 0.1, batch size 16, 30 epochs, and random seed 42. Checkpoints are selected using validation mean absolute error. Events, rather than individual station samples, are separated among training, validation, and test subsets to reduce leakage between samples from the same earthquake.

### Benchmark Methods

Three complementary benchmark pathways are considered.

**Persistence-cummax.** The predicted final intensity is the maximum target-station intensity already observed in the early window. This is a completely observation-based lower-complexity baseline. It has no ability to anticipate intensity growth beyond the current target-station maximum.

**PLUM-R30-cummax-exclude-self.** The predicted intensity at a target is the maximum observed intensity among stations within 30 km, with the target station excluded. Excluding the target creates a strict propagation-only comparison and prevents the baseline from degenerating into target persistence.

**eBEAR-related operational sensitivity.** eBEAR is treated as the source-parameter operational reference. Station-level comparison is not claimed because the mapping between eBEAR dissemination sites and strong-motion stations remains provisional. Instead, a separate event-level sensitivity analysis uses 160 events matched between the evaluation archive and an online eBEAR log at an alert definition equivalent to intensity 4 or PGA ≥ 25 cm/s².

### Evaluation Metrics

Regression performance is measured using mean absolute error (MAE), root mean square error (RMSE), and bias. Ordered-class performance is measured using exact accuracy and off-by-one accuracy. Threshold performance is evaluated at intensity `I ≥ 3`, `I ≥ 4`, and `I ≥ 5−` using precision, probability of detection (POD or recall), F1 score, false-alarm rate (FAR), critical success index, and miss rate. The principal operational threshold is `I ≥ 4`, corresponding to the threshold used for Taiwan's national public earthquake warning.

Because early windows are referenced to catalog origin time, the reported window length is not equal to delivered public warning time. A prospective system would need to subtract telemetry, event recognition or window alignment, inference, decision, and dissemination latency. The present analysis therefore evaluates forecasting skill and the time–accuracy frontier, not end-to-end public lead time.

Point estimates in the tables below are computed over 97,354 station-event samples per window before filtering, so sampling variability at the level of individual precision or POD values is small. We nonetheless report exact numerators and denominators (true and false positives, true and false negatives) wherever available, such as for the eBEAR sensitivity analysis, so that readers can reconstruct exact or approximate confidence intervals. Event-block bootstrap confidence intervals for the full SITF-versus-baseline comparison have not yet been computed and are identified as a specific target for the revised analysis in Limitations.

## Results

### Overall Forecast Accuracy

Forecast accuracy improves as more of the intensity sequence becomes available. Across the 161-event independent evaluation, exact class accuracy rises from 56.4% at EW05 to 79.4% at EW40, while off-by-one accuracy rises from 83.8% to 98.2%. Continuous-intensity MAE decreases from 0.755 to 0.261 and RMSE decreases from 1.095 to 0.521. These trends show that the early intensity trajectory contains increasing information about final shaking and that the model progressively converges as the waveform-derived intensity sequence develops.

Average accuracy alone is insufficient for evaluating an EEW system. Most station-event samples are below the public-warning threshold, so apparently high overall accuracy can coexist with poor warning performance. We therefore focus on threshold discrimination at `I ≥ 4`.

### Performance at the National Public-Warning Threshold

Threshold performance is strongly window dependent. At EW05, SITF is precise but detects only a small fraction of stations that eventually reach intensity 4. Performance deteriorates around EW15: precision falls to 51.2% and FAR rises to 2.15%. Precision then recovers sharply at EW20, reaching 91.5% with POD 49.5%, F1 0.642, and FAR 0.38% on the matched common-valid population. EW30 provides the most conservative learned operating point, with 96.2% precision and FAR 0.20%. At EW40, precision remains 96.4%, POD reaches 82.5%, and F1 reaches 0.889, with FAR 0.26%.

The EW15 trough is a substantive result rather than measurement noise: it demonstrates that more elapsed time does not automatically produce monotonically better threshold behavior. The physical interpretation may involve the transition between weak onset observations and more diagnostic S-wave motion, but travel-time analysis is required before assigning a definitive mechanism. Operationally, the result argues against selecting a single checkpoint solely by average accuracy. A warning policy should instead be calibrated by window and should privilege precision or POD according to the intended action.

### Comparison with Direct Observation Baselines

At EW20, persistence-cummax is perfectly precise but conservative: POD is 47.4% and F1 is 0.643. SITF produces nearly the same F1 (0.642), slightly higher POD (49.5%), and a small FAR of 0.38%. The result indicates that SITF has begun to anticipate some future intensity growth while remaining close to the conservative behavior of persistence.

The propagation-only PLUM-R30 baseline reaches a higher POD of 65.9% at EW20, but precision is only 35.7% and FAR is 9.84%. At EW40, PLUM reaches POD 91.1%, but precision remains 35.2% and FAR increases to 13.87%. In contrast, SITF reaches POD 82.5% with precision 96.4% and FAR 0.26%. Thus, PLUM and SITF occupy distinctly different operating points. PLUM aggressively extends observed strong shaking to nearby sites and favors detection, whereas SITF favors reliable station-specific confirmation with far fewer false alarms.

At EW40, persistence and SITF converge. Persistence has F1 0.891 and POD 80.3%, while SITF has F1 0.889 and POD 82.5%. The learned model does not materially improve F1 at this late window, but it detects slightly more threshold exceedances at the cost of a small number of false alarms. This convergence is expected because by 40 s the target station has often recorded much of the damaging motion.

### Complementarity with eBEAR

The event-level eBEAR sensitivity analysis includes 160 matched events. Under the event-alert definition, eBEAR produces 126 true positives, 2 false positives, 30 false negatives, and 2 true negatives, corresponding to precision 98.4%, POD 80.8%, and F1 0.887. An event-level SITF rule based on EW20 or later identifies all 156 alert-positive events in this retrospective matched set with one false positive. Combining the eBEAR and SITF event triggers removes the 30 eBEAR misses while retaining precision of 98.1%.

This result must be interpreted at the event level. It does not establish that SITF provides more accurate site-specific warning than eBEAR, because matched station-level eBEAR predictions are not yet available. It does, however, support the central ensemble argument: source-parameter and observation-driven pathways fail for different reasons, and their union can reduce misses relative to either pathway alone.

### Spatial Heterogeneity

Performance varies geographically. At EW20, lower county-level exact accuracy occurs in parts of western Taiwan, including Changhua, Yunlin, and Chiayi, whereas higher accuracy occurs in Penghu, Kaohsiung, Pingtung, and Taitung. Two Yunlin stations exhibit intensity-4 false-alarm rates much larger than the network-wide average. These outliers indicate that a single network-wide threshold may be inadequate. Station-aware probability calibration, site-specific thresholds, or multi-station corroboration should be considered before deployment.

## Discussion

### Scope of Trigger Independence

The term “trigger-free” requires precise definition. SITF does not wait for the target station or a neighboring station to exceed an amplitude or intensity threshold before generating an estimate. Zero and weak early values are legitimate inputs, and the model can issue an updated forecast at each second. In that station-level sense, the forecasting layer is trigger independent.

The current experiment nevertheless uses windows aligned to catalog origin time. SITF does not autonomously detect earthquakes or determine the origin of an event stream. An operational implementation would need an external timing reference, a continuous sliding-window detector, or integration with eBEAR. The strongest deployment concept is therefore not a standalone trigger-free system, but an always-running observation-driven module activated or time-aligned by the existing EEW infrastructure while remaining independent of the source parameters used by that infrastructure.

### Relationship to Prior Machine-Learning Approaches to Earthquake Early Warning

SITF's single-station, source-parameter-free design should be read as a deliberate complement to, rather than a competitor with, recent multi-station deep-learning alerting models. Grid-based convolutional models (Jozinovic et al., 2020) and cross-station transformer models such as TEAM (Münchmeyer et al., 2021) are designed to extrapolate shaking to stations that have not yet recorded any motion, which requires them to model or implicitly encode the source and the wavefield jointly across the network. That capability is valuable precisely where SITF cannot help: a station with no early signal has no sequence for SITF to forecast from. Conversely, because SITF requires neither cross-station input nor any source-derived quantity, it could in principle be layered on top of either a traditional source-parameter pipeline such as eBEAR or a learned multi-station architecture such as TEAM, providing a source-independent, station-specific check on whichever upstream pathway is used to generate the initial broad-area alert. We view this as a design-space distinction rather than a performance claim: the present study does not benchmark SITF against TEAM or comparable multi-station networks on a shared dataset, and doing so is identified as future work.

### Independence from Source-Parameter Uncertainty

Source-parameter EEW and SITF are affected by different uncertainties. eBEAR must infer an evolving earthquake from early P-wave observations and then map that source estimate into expected intensity. Errors can arise from pick association, hypocenter, magnitude, finite rupture, or the ground-motion model. SITF bypasses that chain at inference and forecasts directly from the observed local intensity trajectory. It is therefore less exposed to errors caused specifically by an incorrect source model.

This independence does not imply immunity to all earthquake complexity. A local station sequence can still be ambiguous before strong motion arrives, rare large events may be poorly represented in training, and site noise can produce false predictions. The proper claim is narrower: the forecast is not conditioned on an estimated point source, so point-source errors cannot directly propagate through the SITF inference path.

### Warning Time versus Reliability

SITF's principal cost is short warning time. A source-parameter system may alert distant sites before they record any shaking, whereas a station-local intensity model requires the early motion at that station to contain predictive information. EW20 provides a high-precision confirmation mode but only about half of eventual intensity-4 exceedances are detected at that time. EW40 detects more than 80%, but the remaining usable lead time may be small or zero at many sites.

This tradeoff does not eliminate operational value. Many protective actions are sensitive to false alarms and can benefit from a highly reliable secondary confirmation, even when it arrives later than the first source-based alert. SITF may also refine the estimated warning area, sustain or cancel an earlier alert, support situational awareness during a prolonged rupture, and provide redundancy when the source solution is unstable. The system is most defensible as a layered warning architecture: eBEAR provides the earliest broad-area alert, SITF supplies observation-driven confirmation and correction, and PLUM-like propagation supplies a robust high-detection pathway where false-alarm tolerance is greater.

### Operational Significance of the Intensity-4 Threshold Result

The national warning threshold gives the analysis a clear decision context, and it is here that the results have the most direct bearing on practice. The result is not that SITF has the highest overall exact accuracy, nor that it dominates every baseline; average accuracy is dominated by the many below-threshold samples and is not, by itself, an operational criterion. Rather, after sufficient early observation, a source-independent, single-station intensity-sequence model identifies intensity-4 exceedances with very high precision and very low false-alarm rate across a large independent station-event population: at EW20, precision exceeds 90%; at EW30–EW40, it exceeds 96%, while FAR remains below 0.3%. This is the regime in which a secondary confirmation channel can plausibly add value to an operational ensemble without materially increasing nuisance alerts.

## Limitations

First, the primary target is final peak intensity, not intensity at a fixed future horizon. A separate lightweight `t+h` diagnostic shows that skill degrades with horizon, but a production Transformer trained explicitly for `t+1`, `t+3`, `t+5`, and `t+10 s` remains future work. Second, the analysis is retrospective and origin-time referenced; true public lead time and alert latency are not measured. Third, strong-intensity samples are sparse, and class imbalance can make station-level reliability unstable. Fourth, the present model is station local and cannot use the spatial coherence of a dense network. Fifth, the eBEAR comparison is event-level sensitivity rather than a matched station-level benchmark, and no shared-dataset comparison against multi-station deep-learning alerting architectures (e.g., TEAM) has been performed. Sixth, the evaluation does not yet quantify the 2024 Hualien earthquake as part of the independent test set; that event should be presented as an illustrative case unless a leakage-free special-event protocol is established. Seventh, event-block bootstrap confidence intervals for the primary precision, POD, F1, and FAR comparisons have not yet been computed; although the underlying sample sizes are large (97,354 station-event pairs per window before filtering), a revised analysis should report resampling-based uncertainty rather than point estimates alone. Eighth, all training and evaluation events are Taiwanese earthquakes recorded on Taiwan's relatively dense strong-motion network; performance on sparser networks, different regional seismicity, or non-Taiwan intensity-scale conventions has not been tested and should not be assumed.

## Operational Path Forward

A prospective shadow-mode experiment should run SITF alongside eBEAR using live one-second intensity messages. The experiment should record the timestamp of every input, forecast, threshold crossing, eBEAR message, and final observed intensity. Predefined decision rules should be evaluated, including an eBEAR-first/SITF-confirm policy, a union policy designed to reduce misses, and a precision-constrained policy requiring temporal or neighboring-station consistency. The primary outcomes should be delivered lead time, precision, POD, FAR, alert stability, and station- or county-level calibration. A shared-dataset benchmark against multi-station deep-learning alerting architectures such as TEAM would also clarify whether SITF's station-local design point is dominated on Taiwan data or genuinely complementary in the accuracy–latency–complexity space.

## Conclusions

We developed and evaluated a station-level Transformer that forecasts final seismic intensity from incomplete one-second intensity sequences without using magnitude, hypocenter, rupture geometry, or a ground-motion prediction equation as model input. On 161 earthquakes excluded from the training archive, SITF's exact accuracy increased from 56.4% at 5 s to 79.4% at 40 s, and off-by-one accuracy increased from 83.8% to 98.2%.

At Taiwan's national public-warning threshold of intensity 4, the model provided a high-reliability operating regime. Precision reached 91.5% at EW20 and exceeded 96% at EW30–EW40, while FAR remained below 0.4%. A strict PLUM-like propagation baseline achieved greater detection but generated substantially more false alarms, whereas persistence was nearly perfectly precise but more conservative. These complementary behaviors show that no single method dominates the warning-time–accuracy tradeoff.

The evidence supports SITF as a low-false-alarm observation-driven layer that complements eBEAR and, in principle, other source-centric pathways including multi-station deep-learning alerting architectures. eBEAR remains responsible for the earliest detection, source characterization, and broad-area warning; SITF can provide independent confirmation and refinement that is not directly affected by source-parameter errors. The next requirements are prospective shadow-mode testing with true telemetry and dissemination latency, and a shared-dataset comparison against multi-station learned alerting models. Until such testing is completed, SITF should be presented as a retrospective forecasting and ensemble-confirmation method, not as an operational standalone warning system.

## Data and Resources

The analysis uses Central Weather Administration station-intensity sequences and operational research products. Event-level raw files, operational eBEAR logs, and trained weights may be subject to Central Weather Administration data-distribution and operational-security policies. Reproducibility materials should include the nonoverlap event manifest, preprocessing and evaluation scripts, model configuration, derived metric tables, and synthetic or small example files. The public code repository currently associated with the project is `https://github.com/oceanicdayi/Transformer_seismic_intensity`.

## Declaration of Competing Interests

The authors declare no competing interests.

## Acknowledgments

We thank the Central Weather Administration for maintaining Taiwan's seismic and strong-motion monitoring networks and the operational eBEAR workflow. We also thank colleagues who contributed to data preparation, operational interpretation, and review of the analysis workflow.

## References

- Allen, R. M., and D. Melgar (2019). Earthquake early warning: Advances, scientific challenges, and societal needs, Annual Review of Earth and Planetary Sciences 47, 361–388.
- Allen, R. M., P. Gasparini, O. Kamigaichi, and M. Böse (2009). The status of earthquake early warning around the world: An introductory overview, Seismological Research Letters 80, 682–693.
- Hoshiba, M., and S. Aoki (2015). Numerical shake prediction for earthquake early warning: Data assimilation, real-time shake mapping, and simulation of wave propagation, Bulletin of the Seismological Society of America 105, 1324–1338.
- Johnson, C. E., A. Bittenbinder, B. Bogaert, L. Dietz, and W. Kohler (1995). Earthworm: A flexible approach to seismic network processing, IRIS Newsletter 14, 1–4.
- Jozinovic, D., A. Lomax, I. Stajduhar, and A. Michelini (2020). Rapid prediction of earthquake ground shaking intensity using raw waveform data and a convolutional neural network, Geophysical Journal International 222, 1379–1389.
- Kodera, Y., Y. Yamada, K. Hirano, K. Tamaribuchi, S. Adachi, N. Hayashimoto, M. Morimoto, M. Nakamura, and M. Hoshiba (2018). The Propagation of Local Undamped Motion (PLUM) method: A simple and robust seismic wavefield estimation approach for earthquake early warning, Bulletin of the Seismological Society of America 108, 983–1003.
- Kohler, M. D., et al. (2020). Earthquake early warning ShakeAlert system: West Coast wide production prototype, Seismological Research Letters 91, 1763–1775.
- Kuyuk, H. S., and R. M. Allen (2013). A global approach to provide magnitude estimates for earthquake early warning alerts, Geophysical Research Letters 40, 6329–6333.
- Meier, M.-A. (2017). How "good" are real-time ground motion predictions from earthquake early warning systems?, Journal of Geophysical Research: Solid Earth 122, 5561–5577.
- Münchmeyer, J., D. Bindi, U. Leser, and F. Tilmann (2021). The transformer earthquake alerting model: A new versatile approach to earthquake early warning, Geophysical Journal International 225, 646–656.
- Vaswani, A., N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin (2017). Attention is all you need, Advances in Neural Information Processing Systems 30.
- Wu, Y.-M., et al. (2013). Progress on earthquake early warning in Taiwan: From prototype to real-time operation, Seismological Research Letters 84, 635–644.

## Table 1. Data summary

| Item | Value |
|---|---:|
| Training archive | 198 earthquakes |
| Internal split | 158 train / 20 validation / 20 test earthquakes |
| Original evaluation archive | 208 earthquakes |
| Events removed for overlap | 47 |
| Independent evaluation set | 161 earthquakes |
| Station-event samples per early window | 97,354 |
| Station metadata | 639 stations |

## Table 2. Intensity-4 benchmark comparison

| Method | Window | Precision | POD/Recall | F1 | FAR |
|---|---:|---:|---:|---:|---:|
| Persistence-cummax | EW20 | 100.00% | 47.39% | 0.643 | 0.00% |
| PLUM-R30, exclude target | EW20 | 35.71% | 65.90% | 0.463 | 9.84% |
| SITF | EW20 | 91.50% | 49.48% | 0.642 | 0.38% |
| Persistence-cummax | EW30 | 100.00% | 62.67% | 0.770 | 0.00% |
| PLUM-R30, exclude target | EW30 | 35.55% | 80.53% | 0.493 | 12.10% |
| SITF | EW30 | 96.15% | 59.52% | 0.735 | 0.20% |
| Persistence-cummax | EW40 | 100.00% | 80.32% | 0.891 | 0.00% |
| PLUM-R30, exclude target | EW40 | 35.23% | 91.12% | 0.508 | 13.87% |
| SITF | EW40 | 96.39% | 82.52% | 0.889 | 0.26% |

## Recommended Figure Set

1. Taiwan events and station distribution, with training and independent evaluation events distinguished.
2. Operational concept: eBEAR source-parameter pathway, PLUM observation-propagation pathway, and SITF station-sequence forecasting pathway.
3. Example one-second intensity sequences with EW05–EW40 windows and final peak labels.
4. SITF architecture.
5. Exact accuracy, off-by-one accuracy, MAE, and RMSE versus window.
6. Intensity-4 precision, POD, F1, FAR, and confusion counts versus window.
7. Matched EW20/EW30/EW40 comparison among persistence, PLUM-R30, and SITF.
8. Station-level spatial maps and high-FAR outliers.
9. Event-level eBEAR/SITF complementarity confusion diagram.
10. Illustrative 2024 Hualien case study, explicitly labeled as outside the primary independent test set.
11. Conceptual design-space comparison distinguishing SITF (single-station, source-parameter-free forecasting) from multi-station deep-learning alerting architectures such as TEAM (cross-station, source-estimating).
