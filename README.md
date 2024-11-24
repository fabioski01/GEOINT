# INFERNO Project - 8th CASSINI Hackathons

This repository contains the Python code for two key phases of **Project INFERNO**, which is aimed at detecting and analyzing military activities using satellite imagery and AI-driven analysis. Below, we outline the workflow and the purpose of each phase.

---

## Phase 1: Satellite Data Extraction via OpenEO

In the first step, satellite imagery data is extracted from Sentinel-2 using the **OpenEO** platform. The area of interest (AOI) is defined by a shapefile stored locally, which specifies the geographical region to analyze. For example, in this case, the Donetsk region in Ukraine was used as the test AOI.

Key features of this phase include:
- **Input:** AOI is defined through a shapefile in the local folder.
- **Output:** Satellite data as images (rectangles of a few square kilometers each) within a specified **time frame**.
- **Data Format:** Extracted images are processed for further AI-based analysis.

---

## Phase 2: Morphological Change Detection and Object Classification

The extracted Sentinel-2 images will undergo **AI-driven analysis** to identify significant morphological changes, such as:
- Mechanized movements.
- Modifications in pre-identified military infrastructures.

Upon detecting changes, high-resolution satellite data from paid providers (e.g., Maxar, Airbus) are requested to zoom into specific areas. The higher resolution data is then used to:
1. **Detect vehicles:** Identify the presence of vehicles or structures of interest.
2. **Classify objects:** Recognize and categorize vehicles or military equipment.

This phase leverages a neural network trained on a dataset of various vehicles, including military vehicles. The dataset is extendable, allowing the inclusion of additional vehicles to improve classification accuracy.

---

## Current Capabilities

- **Sentinel-2 Data Extraction:** Fully automated pipeline for image extraction.
- **AI Detection and Classification:**
  - Initial detection and classification implemented using a neural network.
  - Trained dataset includes various civilian vehicles.
  - Expandable dataset for broader recognition capabilities of military vehicles.

---

## Future Development

Planned improvements and features include:
1. **Enhanced Morphological Change Detection:** Incorporating advanced AI techniques for detecting subtle changes inside Copernicus' Datacubes.
2. **Integration of High-Resolution Satellite Data Providers:** Automating the process of acquiring high-resolution data upon detecting changes.
3. **Vehicle Dataset Expansion:** Adding more labeled examples for both military and civilian vehicles to improve classification precision.
