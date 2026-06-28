import subprocess

N_ENVS = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128, 256]

for n_envs in N_ENVS:
    print(f"Running {n_envs} envs...")

    subprocess.run(
        [
            "python",
            "-m",
            "Agent.train",
            "--n_envs",
            str(n_envs),
        ],
        check=True,
    )

print("Benchmark finished.")


  # Paste into train.py, so the n_envs is correctly parsed.
  # import argparse

  # parser = argparse.ArgumentParser()
  # parser.add_argument("--n_envs", type=int, default=1)
  # args = parser.parse_args()

  # sim_config.n_envs = args.n_envs