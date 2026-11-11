//! `gpss-viewer`: the desktop viewer, or `gpss-viewer record` for headless videos.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use gpss_viewer::record::{self, Options};
use gpss_viewer::render::{self, Renderer};

#[derive(Parser)]
#[command(version, about = "3D viewer for gpss cuboids")]
struct Cli {
    #[command(subcommand)]
    command: Option<Cmd>,
    /// Puzzle dimensions (R-L width x F-B depth x U-D height).
    #[arg(long, default_value = "3x3x3")]
    dims: String,
    /// Start from a random scramble of this many moves.
    #[arg(long)]
    scramble: Option<usize>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Render scenario scripts headlessly to captioned MP4s plus logs.
    Record {
        /// Scenario files.
        scripts: Vec<PathBuf>,
        /// Output directory.
        #[arg(long, default_value = "target/viewer-videos")]
        out: PathBuf,
        /// Frames per second.
        #[arg(long, default_value_t = 30)]
        fps: u32,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Some(Cmd::Record { scripts, out, fps }) => {
            let (device, queue, adapter) = match render::headless_device() {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::from(2);
                }
            };
            eprintln!("rendering on {adapter}");
            let mut renderer = Renderer::new(&device);
            let opts = Options { out_dir: out, fps };
            let mut failed = false;
            for s in &scripts {
                match record::record(&device, &queue, &mut renderer, s, &opts) {
                    Ok(r) => {
                        println!(
                            "{}: {}/{} expectations passed -> {}",
                            r.name,
                            r.checked - r.failures.len(),
                            r.checked,
                            r.video.display()
                        );
                        for f in &r.failures {
                            println!("  FAIL {f}");
                        }
                        failed |= !r.failures.is_empty();
                    }
                    Err(e) => {
                        println!("ERROR {e}");
                        failed = true;
                    }
                }
            }
            if failed {
                ExitCode::from(1)
            } else {
                ExitCode::SUCCESS
            }
        }
        None => gpss_viewer::app::run(&cli.dims, cli.scramble),
    }
}
