//! Train command implementation.

use clap::Parser;

/// Train command arguments.
#[derive(Parser)]
pub struct TrainCommand {
    /// Path to the training data file
    #[arg(short, long)]
    pub input: String,

    /// Output directory for the trained model
    #[arg(short, long)]
    pub output: String,

    /// Target vocabulary size
    #[arg(short, long, default_value_t = 30_000)]
    pub vocab_size: usize,

    /// Minimum frequency for merges
    #[arg(short, long, default_value_t = 2)]
    pub min_frequency: u64,

    /// Enable parallel training
    #[arg(short, long, default_value_t = true)]
    pub parallel: bool,
}

use anyhow::Result as AnyhowResult;
use beepe::Tokenizer;
use std::fs;
use std::path::Path;
use std::time::Instant;

pub fn run(cmd: TrainCommand) -> AnyhowResult<()> {
    println!("Training tokenizer...");
    println!("  Input: {}", cmd.input);
    println!("  Output: {}", cmd.output);
    println!("  Vocab size: {}", cmd.vocab_size);
    println!("  Min frequency: {}", cmd.min_frequency);
    println!("  Parallel: {}", cmd.parallel);
    println!();

    // Read training data
    let start = Instant::now();
    let data = fs::read_to_string(&cmd.input)?;
    println!("Read {} bytes in {:.2}s", data.len(), start.elapsed().as_secs_f64());
    println!();

    // Create tokenizer
    let mut tokenizer = Tokenizer::builder()
        .vocab_size(cmd.vocab_size)
        .min_frequency(cmd.min_frequency)
        .build()?;

    // Train
    let start = Instant::now();
    tokenizer.train(&data)?;
    println!("Training completed in {:.2}s", start.elapsed().as_secs_f64());
    println!("Final vocab size: {}", tokenizer.vocab_size());
    println!();

    // Save model
    let output_path = Path::new(&cmd.output);
    let start = Instant::now();
    tokenizer.save(output_path)?;
    println!("Model saved to {} in {:.2}s", cmd.output, start.elapsed().as_secs_f64());

    Ok(())
}
