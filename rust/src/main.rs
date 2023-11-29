use clap::{Arg, Command};
use gptx::nn;
use gptx::ops;
use gptx::tensor::Tensor;
use rayon::ThreadPoolBuilder;
use std::time::Instant;
use tokenizers::{Result, Tokenizer};

fn main() -> Result<()> {
    let matches = Command::new("GPT2 Inference in Rust")
        .version("0.1.0")
        .author("Arun Patro, Olivia")
        .about("GPT2 Inference in Rust")
        .arg(
            Arg::new("max_tokens")
                .short('n')
                .long("tokens")
                .help("Sets the max tokens")
                .default_value("20"),
        )
        .arg(
            Arg::new("max_threads")
                .short('t')
                .long("threads")
                .help("Sets the number of threads")
                .required(false)
                .default_value("1"),
        )
        .arg(
            Arg::new("model_path")
                .short('m')
                .long("model_path")
                .help("The path to the model weights in json")
                .required(false)
                .default_value("model_weights.json"),
        )
        .get_matches();

    let max_tokens_str = matches
        .get_one::<String>("max_tokens")
        .expect("Argument max_tokens missing");
    let max_tokens = max_tokens_str
        .parse::<usize>()
        .expect("Invalid input for max_tokens");

    let max_threads_str = matches
        .get_one::<String>("max_threads")
        .expect("Argument max_threads missing");
    let max_threads = max_threads_str
        .parse::<usize>()
        .expect("Invalid input for max_threads");

    let model_path = matches.get_one::<String>("model_path").unwrap();

    // configs
    println!("num threads: {:?}", max_threads);
    println!("num tokens: {:?}", max_tokens);
    println!("==========================");

    // setup
    ThreadPoolBuilder::new()
        .num_threads(max_threads)
        .build_global()
        .unwrap();

    // setup
    let tokenizer = Tokenizer::from_file("gpt2_tokenizer.json")?;
    println!("Loading model from: {:?}", model_path);
    let start = Instant::now();
    let model = nn::GPT2::from_json(model_path);
    let model_load_time = start.elapsed();

    let prompt = "The answer to life, the universe, and everything is";
    // let encoding = tokenizer.encode(prompt, false)?;
    let tokens = vec![464, 3280, 284, 1204, 11, 262, 6881, 11, 290, 2279, 318];
    println!("[input]: {:?}", prompt);
    // println!("logits: {:?}", model.forward(&tokens));

    // inference
    let start = Instant::now();
    let output_tokens = model.generate(&tokens, max_tokens);
    let model_inference_time = start.elapsed();

    let output_text = tokenizer.decode(
        output_tokens
            .iter()
            .map(|&token| token as u32)
            .collect::<Vec<u32>>()
            .as_slice(),
        false,
    )?;
    println!("[output]: {:?}", output_text);

    // print stats
    println!("==========================");
    println!("load time: {:.2}", model_load_time.as_secs_f32());
    println!("inference time: {:.2}", model_inference_time.as_secs_f32());
    println!(
        "s / token: {:.2}",
        model_inference_time.as_secs_f32() / max_tokens as f32
    );
    println!(
        "token / s: {:.2}",
        max_tokens as f32 / model_inference_time.as_secs_f32()
    );
    Ok(())
}

mod tests {
    use super::*;

    #[test]
    fn test_tokenizer() -> Result<()> {
        let prompt = "The answer to life, the universe, and everything is";
        let tokenizer = Tokenizer::from_file("gpt2_tokenizer.json")?;
        assert_eq!(
            tokenizer.encode(prompt, false)?.get_ids(),
            vec![464, 3280, 284, 1204, 11, 262, 6881, 11, 290, 2279, 318]
        );
        Ok(())
    }
}
