//! CLI commands for the beepe tokenizer.

pub mod benchmark;
pub mod decode;
pub mod encode;
pub mod train;

pub use benchmark::BenchmarkCommand;
pub use decode::DecodeCommand;
pub use encode::EncodeCommand;
pub use train::TrainCommand;
