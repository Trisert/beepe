//! Decode command implementation.

use clap::Parser;

/// Decode command arguments.
#[derive(Parser)]
pub struct DecodeCommand {
    /// Path to the trained tokenizer model
    #[arg(short, long)]
    pub tokenizer: String,

    /// Token IDs to decode (comma-separated)
    #[arg(short, long)]
    pub tokens: String,

    /// Skip special tokens during decoding
    #[arg(short, long, default_value_t = false)]
    pub skip_special: bool,
}

use anyhow::Result as AnyhowResult;
use beepe::Tokenizer;
use std::path::Path;

pub fn run(cmd: DecodeCommand) -> AnyhowResult<()> {
    // Load tokenizer
    let tokenizer_path = Path::new(&cmd.tokenizer);
    let tokenizer = Tokenizer::load(tokenizer_path)?;

    // Parse token IDs
    let ids: Vec<u32> = cmd
        .tokens
        .split(',')
        .map(|s| s.trim().parse::<u32>())
        .collect::<Result<Vec<_>, _>>()?;

    // Decode
    let text = tokenizer.decode(&ids, cmd.skip_special);

    println!("{}", text);

    Ok(())
}
