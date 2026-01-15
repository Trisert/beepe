//! CLI commands for the beepe tokenizer.

pub mod encode;
pub mod decode;
pub mod train;
pub mod benchmark;

pub use train::TrainCommand;
pub use encode::EncodeCommand;
pub use decode::DecodeCommand;
pub use benchmark::BenchmarkCommand;

