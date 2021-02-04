# fingerprint-matching

### Fingerprint Features
*Ridges* - where the skin has a higher profile than its surroundings **(BLACK)** \
*Valleys* - where the skin has a lower profile than its surroundings **(WHITE)** \
*Directional field* (DF) - local orientation of the ridge-valley structures **(VECTOR FIELD)** \
*The singular points* (SPs) - discontinuities in the directional field **(CORE, DELTA)** \
*Minutiae* - details of the ridge-valley structures **(RIDGE ENDINGS, BIFURCATIONS)** \
*Directional field* is used for **enhancement** of the fingerprint \
*Directional field* + *The singular points* is used for **classification** \
*Minutiae* is used for **matching**

### Steps

#### 1. Feature Extraction
1. - [x] *Directional field* estimation

<div align="center">
  <img src="https://github.com/ViktorBusk/fingerprint-matching/blob/main/recourses/orientation_field_beta_1_0_0.png" height="40%" width="40%">
</div>

2. - [ ] *Singular point* Extraction

3. - [ ] Segmentation of fingerprint
4. - [ ] *Minutiae* Extraction

#### 2. Matching

#### 3. Database Search
