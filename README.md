# RON_topology
Example script

```bash
python sMNIST.py --n_hid 100 --dt 0.042 --gamma 2 --epsilon 20 --gamma_range 2 --epsilon_range 2 --inp_scaling 1 --rho 9 --ron --trials 1 --topology lower --sparsity 0.8
```

Model assessment can be triggered via `--use_test` flag and by setting `--trials 5` (to compute mean and standard deviation).