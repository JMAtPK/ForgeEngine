mod bundle;
mod contract;
mod dag;
mod dag_runner;
mod gpu;
mod harness;
mod registry;
mod schema;

use clap::{Parser, Subcommand};
use std::path::PathBuf;
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
    /// Verify a kernel bundle
    Verify {
        /// Path to the bundle JSON file
        bundle: PathBuf,
        /// Registry directory (default: .forge/registry)
        #[arg(long, default_value = ".forge/registry")]
        registry: PathBuf,
    },
    /// Validate a DAG manifest
    ValidateDag {
        /// Path to the DAG manifest JSON file
        manifest: PathBuf,
        /// Registry directory (default: .forge/registry)
        #[arg(long, default_value = ".forge/registry")]
        registry: PathBuf,
    },
    /// Run a game in windowed mode
    Run {
        /// Path to the game manifest JSON file
        manifest: PathBuf,
        /// Registry directory (default: .forge/registry)
        #[arg(long, default_value = ".forge/registry")]
        registry: PathBuf,
        /// Enable per-dispatch postcondition verification
        #[arg(long)]
        verify: bool,
        /// PRNG seed for deterministic replay
        #[arg(long)]
        seed: Option<u64>,
    },
    /// Run a DAG pipeline from a game manifest
    RunDag {
        /// Path to the game manifest JSON file
        manifest: PathBuf,
        /// Registry directory (default: .forge/registry)
        #[arg(long, default_value = ".forge/registry")]
        registry: PathBuf,
        /// Number of frames to execute
        #[arg(long, default_value = "1")]
        frames: usize,
        /// Enable per-dispatch postcondition verification
        #[arg(long)]
        verify: bool,
        /// Dump final buffer state as JSON to this file
        #[arg(long)]
        output: Option<PathBuf>,
    },
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
        Some(Command::Verify {
            bundle: bundle_path,
            registry: registry_path,
        }) => {
            let bun = match bundle::load_bundle(&bundle_path) {
                Ok(b) => b,
                Err(e) => {
                    let output = bundle::VerifyOutput {
                        accepted: false,
                        kernel: String::new(),
                        registry_id: None,
                        errors: vec![e],
                        steps: vec![],
                    };
                    println!("{}", serde_json::to_string_pretty(&output).unwrap());
                    std::process::exit(1);
                }
            };

            let (device, queue) = gpu::init_device(backends);
            let mut output = bundle::run_full_pipeline(&device, &queue, &bun);

            if output.accepted {
                // Read bundle content for hash
                let content = std::fs::read_to_string(&bundle_path).unwrap_or_default();
                let hash = bundle::bundle_hash(&content);
                let registry_id = format!("{}_{}", output.kernel, hash);

                match registry::register_kernel(
                    &registry_path,
                    &output.kernel,
                    &bun.contract,
                    &hash,
                ) {
                    Ok(_) => {
                        output.registry_id = Some(registry_id);
                    }
                    Err(e) => {
                        eprintln!("Warning: failed to register kernel: {}", e);
                    }
                }
            }

            println!("{}", serde_json::to_string_pretty(&output).unwrap());
            if !output.accepted {
                std::process::exit(1);
            }
        }
        Some(Command::ValidateDag {
            manifest,
            registry: registry_path,
        }) => {
            let dag = match dag::load_dag_manifest(&manifest) {
                Ok(d) => d,
                Err(e) => {
                    let result = dag::DagValidationResult {
                        valid: false,
                        errors: vec![e],
                    };
                    println!("{}", serde_json::to_string_pretty(&result).unwrap());
                    std::process::exit(1);
                }
            };

            let result = dag::validate_dag(&dag, &registry_path);
            println!("{}", serde_json::to_string_pretty(&result).unwrap());
            if !result.valid {
                std::process::exit(1);
            }
        }
        Some(Command::Run {
            manifest,
            registry,
            verify,
            seed,
        }) => {
            harness::run_harness(&manifest, &registry, backends, verify, seed);
        }
        Some(Command::RunDag {
            manifest,
            registry,
            frames,
            verify,
            output,
        }) => {
            let game = match dag_runner::load_game_manifest(&manifest) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            };

            let (device, queue) = gpu::init_device(backends);
            let mut runner = match dag_runner::DagRunner::new(device, queue, &game, &registry) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            };

            for frame in 1..=frames {
                match runner.run_frame(verify) {
                    Ok(result) => {
                        if !result.passed {
                            eprintln!("Frame {}: FAIL", result.frame);
                            for e in &result.errors {
                                eprintln!("  {}", e);
                            }
                            std::process::exit(1);
                        }
                    }
                    Err(e) => {
                        eprintln!("Frame {}: error: {}", frame, e);
                        std::process::exit(1);
                    }
                }
            }

            if let Some(output_path) = output {
                match runner.dump_buffers() {
                    Ok(json) => {
                        let content = serde_json::to_string_pretty(&json).unwrap();
                        if let Err(e) = std::fs::write(&output_path, content) {
                            eprintln!("Failed to write output: {}", e);
                            std::process::exit(1);
                        }
                        println!("Output written to {}", output_path.display());
                    }
                    Err(e) => {
                        eprintln!("Failed to dump buffers: {}", e);
                        std::process::exit(1);
                    }
                }
            }

            println!("forge run-dag: {} frames completed", frames);
        }
    }
}
