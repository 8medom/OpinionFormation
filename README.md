# OpinionFormation

Basic simulations of opinion formation on synthetic signed relation network (Erdős–Rényi model). For a given number of subjects (N), mean degree (z), structural noise (beta), and number of seed opinions (N_S), the program outputs opinion consistency (C) and opinion stability (S) (their mean, standard deviation, and 10th and 90th percentile).

The simulations are run on num_realizations = 100 independent network realizations. On each realization, num_realizations_for_stability = 100 model realizations are run to estimate opinion stability (which is then averaged over all network realizations). All model realizations contribute equally to the estimation of opinion consistency.

See the related paper published in Communications Physics (2021): https://www.nature.com/articles/s42005-021-00579-3.
