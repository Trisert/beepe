//! Encode command implementation.

use clap::Parser;

/// Encode command arguments.
#[derive(Parser)]
pub struct EncodeCommand {
    /// Path to the trained tokenizer model
    #[arg(short, long)]
    pub tokenizer: String,

    /// Text to encode
    #[arg(short, long)]
    pub input: String,

    /// Add special tokens (BOS, EOS)
    #[arg(short, long, default_value_t = false)]
    pub special_tokens: bool,

    /// Output file (stdout if not specified)
    #[arg(short, long)]
    pub output: Option<String>,
}

use anyhow::Result as AnyhowResult;
use beepe_tokenizer::Tokenizer;
use std::path::Path;

pub fn run(cmd: EncodeCommand) -> AnyhowResult<()> {
    // Load tokenizer
    let tokenizer_path = Path::new(&cmd.tokenizer);
    let tokenizer = Tokenizer::load(tokenizer_path)?;

    // Read input text (from stdin if "-")
    let input_text = if cmd.input == "-" {
        use std::io::Read;
        let mut buffer = String::new();
        std::io::stdin().read_to_string(&mut buffer)?;
        buffer
    } else {
        cmd.input
    };

    // Encode text
    let encoding = tokenizer.encode(&input_text, cmd.special_tokens)?;

    // Output
    let ids_str: Vec<String> = encoding.ids.iter().map(|id| id.to_string()).collect();
    let output = ids_str.join(" ");

    match &cmd.output {
        Some(path) => {
            std::fs::write(path, &output)?;
            println!("Encoded {} tokens to {}", encoding.len(), path);
        }
        None => {
            println!("{}", output);
        }
    }

    Ok(())
}
