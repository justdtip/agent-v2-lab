# Chart contract

Four panels show observed native-bf16 map relative error and cosine by repository layer, then native error and cosine versus the finite-difference step scale at layers 1 and 33. Lines connect observed points; they are not fitted response models. Step axes are logarithmic and the relative-error sweep is logarithmic. No finite-difference/autograd match is inferred from cosine alone. The eight promoted-path dtype errors are shown as a status, not plotted as numerical failures or successes. Measurements are the card's, one held row with two selected positions at 128 tokens; runtime is not compared.

A separate tabulated mathematical model computes a mixed-error halving-ratio counterexample and the step/global-RMS identity. These are labelled illustrative/derived rather than fitted parameters or measurements of Gemma. The script pins source hashes, checks complete finished logs and writes CSV/JSON plus PNG/SVG. Sources and original records stay unchanged.
