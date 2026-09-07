# Step-rule provenance — source check before diagnostic01

The whole-sequence finite-difference rule came from this repository's `src/local_llm_lab/pipeline/jlens.py::jacobian_vector_product`, explicitly required by requirements§3.3 and the requirements' existing-building-blocks table. Implementation SOURCE-NOTES-01/02 already identify this source. It was not derived from the hosted weights or from an inspected hosted fitting run.

The local hosted config names Neuronpedia's `fit_lens.py` and credits the Anthropic implementation. The [public Neuronpedia driver](https://github.com/hijohnnylin/neuronpedia/blob/main/utils/neuronpedia-utils/neuronpedia_utils/jlens/fit_lens.py) delegates fitting to `jlens.fit`. The [public Anthropic fitter](https://github.com/anthropics/jacobian-lens/blob/main/jlens/fitting.py), inspected on7 September2026, computes backward derivatives through `torch.autograd.grad` with output cotangents. That code uses autodifferentiation, not a finite-difference epsilon. The whole-sequence step therefore has no demonstrated precedent in that fitter.

Historical limit: the saved config has no fitter commit or dependency lock, and CREDIT.md separately credits the n1000 artifact. Current upstream source does not prove the exact historical n1000 execution. We can identify the local rule's actual origin and the public fitter's method; we cannot certify a historical run from metadata it does not contain. No claim of exact arithmetic is made for autodifferentiation in finite precision.

The initial archived credit URL `anthropics/jlens` returned404; the public driver points to `anthropics/jacobian-lens`. No new checkpoint, corpus or dependency was downloaded.

Local evidence SHA256:

- `models/jlens/source/config.yaml`: `af842b924a20d264d6a3793e01918be49cb2601a3b7ea300a8ecf6eb88822739`
- `models/jlens/source/CREDIT.md`: `abc4178006201a8e96073788cd83db32142e139964e94db0c24c8dbfe10218bf`
- `models/jlens/Qwen3.5-4B_jacobian_lens_n1000.json`: `b019e8c9cef34b1f8f46f1486721565658bcaa5f7e20a261645b6ebc6f94f112`
