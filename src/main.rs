mod gpu;

use clap::{Parser, Subcommand};
use wgpu::Backends;

#[derive(Parser)]
#[command(name = "forge", about = "ForgeEngine — GPU compute engine")]
struct Cli {
    /// Force a specific GPU backend (vulkan, metal, dx12, gl)
    #[arg(long, global = true)]
    backend: Option<String>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run GPU round-trip test
    Test,
    /// Verify GPU availability
    Verify,
    /// Run a compute job
    Run,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    let backends = cli
        .backend
        .as_deref()
        .map(gpu::parse_backend)
        .unwrap_or(Backends::all());

    match cli.command {
        Some(Command::Test) | None => {
            println!("forge: GPU round-trip test");
            let (device, queue) = gpu::init_device(backends);
            let results = gpu::run_test_dispatch(&device, &queue);
            if gpu::verify_results(&results) {
                println!("PASS — all 256 values correct (output[i] == i * 2)");
            } else {
                eprintln!("FAIL — GPU round-trip verification failed");
                std::process::exit(1);
            }
        }
        Some(Command::Verify) => {
            println!("forge verify: not yet implemented");
        }
        Some(Command::Run) => {
            println!("forge run: not yet implemented");
        }
    }
}
