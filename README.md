# eqx_marl

PPO / IPPO / MAPPO in JAX + equinox on brax environments.

`ff_PPO_kan.py` uses Kolmogorov-Arnold networks from [kaneqx](https://github.com/johnviljoen/kaneqx)
as actor and critic, with staged grid extension and Adam state transition; install kaneqx from its
repo (`pip install -e /path/to/kaneqx`). `distreqx` 0.0.3 requires a newer equinox than 0.11.x; use
`distreqx==0.0.1` with equinox 0.11.
