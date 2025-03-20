use clap::CommandFactory;
use clap::Parser;
use env_logger::Builder;
use log::{LevelFilter, info, warn};
use serde::{Deserialize, Serialize};
use serde_json;
use std::env;
use std::fmt::Debug;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::process::exit;
use std::thread::sleep;
use std::time::Duration;

struct GPUInfo {
    gpu_free: usize,
    index: usize,
    memory_free: u32, // GiB
}

impl Debug for GPUInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GPUInfo {{index: {}, memory_free: {} G, gpu_free: {} %}}\n",
            self.index, self.memory_free, self.gpu_free
        )
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct Config {
    core_count: usize,
    memory_per_core: u32,
    gpu_percent: usize,
    check_times: usize,
    check_interval: u64,
    gpu_env: String,
    set_envs: Vec<String>,
    unset_envs: Vec<String>,
}

#[derive(Parser, Debug)]
#[command(version, about = "The program monitors GPU resource availability\nand executes the specified command if resources are sufficient.", long_about = None)]
struct Cli {
    #[arg(short = 'n', long, help = "Specify the GPU core count.")]
    core_count: Option<usize>,

    #[arg(short, long, help = "Set the memory (GB) available on each GPU.")]
    memory_per_core: Option<u32>,

    #[arg(short, long, help = "Set the GPU usage available on each GPU.")]
    gpu_percent: Option<usize>,

    #[arg(short = 'k', long, help = "Set the number of checks to perform.")]
    check_times: Option<usize>,

    #[arg(
        short = 't',
        long,
        help = "Specify the interval for checks in seconds."
    )]
    check_interval: Option<u64>,

    #[arg(short = 'e', long, help = "Set <gpu_env> to the available GPU index.")]
    gpu_env: Option<String>,

    #[arg(
        short,
        long,
        help = "Append environment variables for command execution."
    )]
    set_envs: Option<Vec<String>>,

    #[arg(
        short,
        long,
        help = "Remove specified environment variables for command execution."
    )]
    unset_envs: Option<Vec<String>>,

    #[arg(
        short,
        long,
        value_name = "FILE",
        help = "Read configuration from the specified file path."
    )]
    config_path: Option<PathBuf>,

    #[arg(short, long, help = "Print the current configuration.")]
    print_config: bool,

    #[arg(short = 'w', long, help = "Save the current configuration to a file.")]
    save_config: bool,

    #[arg(help = "Specify the command to execute.")]
    cmd: Vec<String>,

    #[arg(short, long, help = "Increase output verbosity.")]
    verbose: bool,
}

fn main() {
    let cli = Cli::parse();
    if cli.verbose {
        Builder::new().filter_level(LevelFilter::Info).init();
    } else {
        match env::var("RUST_LOG") {
            Ok(_) => {
                Builder::from_default_env().init();
            }
            Err(_) => {
                Builder::new().filter_level(LevelFilter::Warn).init();
            }
        }
    }

    let (config_path, strict) = match cli.config_path.as_deref() {
        Some(v) => (v, true),
        None => (Path::new(".plan.json"), false),
    };
    let mut config = read_config_from_file(config_path, strict);

    if let Some(v) = cli.core_count {
        config.core_count = v;
    }
    if let Some(v) = cli.memory_per_core {
        config.memory_per_core = v;
    }
    if let Some(v) = cli.gpu_percent {
        config.gpu_percent = v;
    }
    if let Some(v) = cli.check_times {
        config.check_times = v;
    }
    if let Some(v) = cli.check_interval {
        config.check_interval = v;
    }
    if let Some(v) = cli.gpu_env {
        config.gpu_env = v;
    }
    if let Some(v) = cli.set_envs {
        if v.len() > 0 {
            config.set_envs = v;
        }
    }
    if let Some(v) = cli.unset_envs {
        if v.len() > 0 {
            config.unset_envs = v;
        }
    }

    if cli.print_config {
        println!(
            "Current config:\n{}",
            serde_json::to_string_pretty(&config).unwrap()
        );
        return;
    } else {
        info!(
            "Current config:\n{}",
            serde_json::to_string_pretty(&config).unwrap()
        );
    }

    if cli.save_config {
        save_config(&config, config_path);
    }

    if cli.cmd.len() == 0 {
        Cli::command().print_help().unwrap();
        return;
    }

    let gpus = wait_for_resource(
        config.core_count,
        config.memory_per_core,
        config.gpu_percent,
        config.check_times,
        config.check_interval,
    );

    run_command(
        &cli.cmd.join(" "),
        &gpus.join(","),
        &config.gpu_env,
        &config.set_envs,
        &config.unset_envs,
    );
}

fn parse_cuda_info() -> Vec<GPUInfo> {
    let mut gpu_info_list = vec![];
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu",
            "index,utilization.gpu,memory.free",
            "--format",
            "csv,noheader",
        ])
        .output()
        .expect("nvidia-smi execute failed");
    for line in String::from_utf8(output.stdout).unwrap().split("\n") {
        if line.len() == 0 {
            break;
        }
        let mut field_it = line.split(", ");
        let index = field_it.next().unwrap().parse::<usize>().unwrap();
        let gpu_percent = field_it.next().unwrap();
        let gpu_percent = gpu_percent[..gpu_percent.len() - 2]
            .parse::<usize>()
            .unwrap();
        let memory_free = field_it.next().unwrap();
        let memory_free = memory_free[..memory_free.len() - 4].parse::<u32>().unwrap();
        let gpu_info = GPUInfo {
            index,
            gpu_free: 100 - gpu_percent,
            memory_free: memory_free / 1024,
        };
        gpu_info_list.push(gpu_info);
    }
    info!("{:?}", gpu_info_list);
    gpu_info_list
}

fn check_resource_enough(
    gpu_info_list: &Vec<GPUInfo>,
    core_count: usize,
    memory_per_core: u32,
    gpu_percent: usize,
) -> Option<Vec<String>> {
    let mut available_gpu = vec![];
    for gpu_info in gpu_info_list {
        if gpu_info.memory_free >= memory_per_core && gpu_info.gpu_free >= gpu_percent {
            available_gpu.push((gpu_info.index, gpu_info.gpu_free));
        }
    }
    if available_gpu.len() >= core_count {
        available_gpu.sort_by_key(|x| 100 - x.1);
        let gpus = available_gpu
            .iter()
            .map(|x| x.0.to_string())
            .collect::<Vec<String>>();

        return Some(gpus[0..core_count].to_vec());
    }
    return None;
}

fn wait_for_resource(
    core_count: usize,
    memory_per_core: u32,
    gpu_percent: usize,
    cum_count: usize,
    interval_sec: u64,
) -> Vec<String> {
    let mut cur_count = 0;
    loop {
        let gpu_info_list = parse_cuda_info();
        match check_resource_enough(&gpu_info_list, core_count, memory_per_core, gpu_percent) {
            Some(gpus) => {
                cur_count += 1;
                info!("Resource is enough: {}", cur_count);
                if cur_count >= cum_count {
                    return gpus;
                }
            }
            None => {
                cur_count = 0;
                info!("Resource isn't enough.");
            }
        }
        sleep(Duration::from_secs(interval_sec));
    }
}

fn run_command(
    cmd: &str,
    gpus: &str,
    gpu_env: &String,
    env: &Vec<String>,
    env_clear: &Vec<String>,
) {
    println!(r"*** Start run `{}` ***", &cmd);
    println!(r"*** Using GPU `{}` ***", &gpus);
    let mut command = Command::new("sh");
    command.arg("-c").arg(&cmd).env(gpu_env, gpus);

    for s in env {
        let kvs: Vec<&str> = s.splitn(2, "=").collect();
        if kvs.len() != 2 {
            warn!("set_envs parse error: {}", s);
            continue;
        }
        command.env(kvs[0], kvs[1]);
    }
    for s in env_clear {
        command.env_remove(s);
    }

    let mut child = command.spawn().expect("Execute cmd failed");
    let status = child.wait().unwrap();
    println!(r"*** Stop run ***");
    exit(status.code().unwrap());
}

fn default_config() -> Config {
    Config {
        core_count: 1,
        memory_per_core: 1,
        gpu_percent: 50,
        check_times: 1,
        check_interval: 15,
        gpu_env: "CUDA_VISIBLE_DEVICES".to_string(),
        set_envs: vec![],
        unset_envs: vec![],
    }
}

fn read_config_from_file(file_path: &Path, strict: bool) -> Config {
    if !fs::exists(file_path).unwrap() {
        if strict {
            panic!("Config file isn't exist: {}", file_path.to_str().unwrap());
        } else {
            return default_config();
        }
    } else {
        let mut file = fs::File::open(file_path).unwrap();
        let mut content = String::new();
        file.read_to_string(&mut content).unwrap();
        let config: Config = serde_json::from_str(&content[..]).expect("Read config failed");
        info!(
            "Read config from {}:\n{}",
            file_path.to_str().unwrap(),
            serde_json::to_string_pretty(&config).unwrap()
        );
        config
    }
}

fn save_config(config: &Config, file_path: &Path) {
    let mut file = fs::File::create(file_path).unwrap();
    file.write_all(serde_json::to_string_pretty(&config).unwrap().as_bytes())
        .unwrap();
    file.flush().unwrap();
    info!("Config is saved to {}", file_path.to_str().unwrap());
}
