# RON_topology
Example script

```bash
python sMNIST.py --n_hid 100 --dt 0.042 --gamma 2 --epsilon 20 --gamma_range 2 --epsilon_range 2 --inp_scaling 1 --rho 9 --ron --trials 1 --topology orthogonal --sparsity 0.8
```

Model assessment can be triggered via `--use_test` flag and by setting `--trials 5` (to compute mean and standard deviation).

Please, note that, due to its structure, the sparsity level of the `lower` matrix is different from all the others.  
To get a matrix with a sparsity of 0.5, 0.6, 0.7, 0.8, 0.9 (as reported in the paper), you need to set the `--sparsity` flag to 0, 0.2, 0.4, 0.6, 0.8 respectively.