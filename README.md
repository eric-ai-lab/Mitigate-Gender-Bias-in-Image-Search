# Mitigate Gender Bias in Image Search

Internet search affects people's cognition of the world, so mitigating biases 
in search results and learning fair models is imperative for social good. We 
study a unique gender bias in image search in this work: the search images are 
often gender-imbalanced for gender-neutral natural language queries. We diagnose 
and mitigate two typical image search models, the specialized model trained on 
in-domain datasets and the generalized representation model pre-trained on 
massive image and text data across the internet.For more details, please see 
our EMNLP 2021 Oral paper [Are Gender-Neutral Queries Really Gender-Neutral? 
Mitigating Gender Bias in Image Search](https://arxiv.org/abs/2109.05433).

## Data

The image search data for occupations can be acquired [here](https://github.com/mjskay/gender-in-image-search). The gender-neutral captions used as MS-COCO and Flicker30K queries are available [here](https://drive.google.com/drive/folders/1VawbfSChqT2skCsCD8kPEwrwYuf4ckN6?usp=sharing).

## Running the code

This repo contains the following code:

- `clip_coco.py` : quantify and mitigate the gender bias by post-processing
- `fairsample.py` : mitigate the gender bias with fair sampling
- `occupational_gender_bias.py` : evaluate and mitigate the similarity bias in realistic image search results

## Reference

If you find this repository helpful for your research, please cite our publication:

```
@InProceedings{Wang2021MitigateGenderBiasInImageSearch,
  author={Wang, Jialu and Liu, Yang and Wang, Xin Eric},
  title={Are Gender-Neutral Queries Really Gender-Neutral? Mitigating Gender Bias in Image Search},
  booktitle={EMNLP},
  year={2021}
}
```
