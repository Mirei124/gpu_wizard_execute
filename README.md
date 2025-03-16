# gpu_wizard_execute

The program monitors GPU resource availability and executes the specified command if resources are sufficient.

## How to use

First, download the binary file from releases and create an initial config. You can use `./gpu_wizard_execute -p` to view the default config, and save the config to a json file (i.e. `.plan.json`).

```
{
  "core_count": 1,
  "memory_per_core": 1,
  "gpu_percent": 50,
  "check_times": 1,
  "check_interval": 15,
  "gpu_env": "CUDA_VISIBLE_DEVICES",
  "set_envs": [],
  "unset_envs": []
}
```

The above config means the program will check the GPU resource `core_count` times, waiting `check_interval` seconds for each check. If there is one GPU with more than `memory_per_core` GB free memory and GPU core usage is lower than `100 - gpu_percent`, the program will execute the giving CMD and set `gpu_env` to the available GPU index (i.e. CUDA_VISIBLE_DEVICES=0), while setting environment variables in `set_envs` and unsetting environment variables in `unset_envs`. For more details, you can use `./gpu_wizard_execute --help`.

```
Usage: gpu_wizard_execute [OPTIONS] [CMD]...

Arguments:
  [CMD]...  Specify the command to execute.

Options:
  -n, --core-count <CORE_COUNT>
          Specify the GPU core count.
  -m, --memory-per-core <MEMORY_PER_CORE>
          Set the memory (GB) available on each GPU.
  -g, --gpu-percent <GPU_PERCENT>
          Define the percentage of free GPU cores.
  -k, --check-times <CHECK_TIMES>
          Set the number of checks to perform.
  -t, --check-interval <CHECK_INTERVAL>
          Specify the interval for checks in seconds.
  -e, --gpu-env <GPU_ENV>
          Set <gpu_env> to the available GPU index.
  -s, --set-envs <SET_ENVS>
          Append environment variables for command execution.
  -u, --unset-envs <UNSET_ENVS>
          Remove specified environment variables for command execution.
  -c, --config-path <FILE>
          Read configuration from the specified file path.
  -p, --print-default-config
          Print the default configuration.
  -w, --save-config
          Save the current configuration to a file.
  -h, --help
          Print help
  -V, --version
          Print version
```
