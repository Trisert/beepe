//! Benchmark command implementation.

use clap::Parser;

/// Benchmark command arguments.
#[derive(Parser)]
pub struct BenchmarkCommand {
    /// Path to the trained tokenizer model
    #[arg(short, long)]
    pub tokenizer: String,

    /// Path to input text file for benchmarking
    #[arg(short, long)]
    pub input: String,

    /// Number of iterations to run
    #[arg(short, long, default_value_t = 100)]
    pub iterations: usize,
}

use anyhow::Result as AnyhowResult;
use beepe_tokenizer::Tokenizer;
use std::fs;
use std::path::Path;
use std::time::Instant;

pub fn run(cmd: BenchmarkCommand) -> AnyhowResult<()> {
    // Load tokenizer
    let tokenizer_path = Path::new(&cmd.tokenizer);
    let tokenizer = Tokenizer::load(tokenizer_path)?;

    // Read input text
    let text = fs::read_to_string(&cmd.input)?;

    println!("Benchmarking encoding...");
    println!("  Text length: {} bytes", text.len());
    println!("  Iterations: {}", cmd.iterations);
    println!();

    // Warmup
    let _ = tokenizer.encode(&text, false);

    // Benchmark
    let start = Instant::now();
    for _ in 0..cmd.iterations {
        let _ = tokenizer.encode(&text, false);
    }
    let elapsed = start.elapsed();

    let avg_time_ns = elapsed.as_nanos() / cmd.iterations as u128;
    let avg_time_ms = avg_time_ns as f64 / 1_000_000.0;

    println!("Results:");
    println!("  Total time: {:.2}s", elapsed.as_secs_f64());
    println!("  Average time: {:.3}ms", avg_time_ms);
    println!("  Throughput: {:.0} tokens/s", 1000.0 / avg_time_ms * 100.0);

    Ok(())
}
