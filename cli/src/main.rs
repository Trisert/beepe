//! Beepe CLI - Command-line interface for the BPE tokenizer.
//!
//! This is the main entry point for the `beepe` command-line tool.

mod commands;

use clap::{Parser, Subcommand};
use commands::{BenchmarkCommand, DecodeCommand, EncodeCommand, TrainCommand};

#[derive(Parser)]
#[command(name = "beepe")]
#[command(about = "A high-performance BPE tokenizer", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a new tokenizer from text data
    Train(TrainCommand),
    /// Encode text to token IDs
    Encode(EncodeCommand),
    /// Decode token IDs back to text
    Decode(DecodeCommand),
    /// Benchmark encoding performance
    Benchmark(BenchmarkCommand),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train(cmd) => commands::train::run(cmd)?,
        Commands::Encode(cmd) => commands::encode::run(cmd)?,
        Commands::Decode(cmd) => commands::decode::run(cmd)?,
        Commands::Benchmark(cmd) => commands::benchmark::run(cmd)?,
    }

    Ok(())
}
