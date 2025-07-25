import argparse
import os
import re
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

def load_tokenizer(model_name_or_path: str) -> Optional[PreTrainedTokenizer]:
    """Loads a Hugging Face tokenizer and handles potential errors."""
    try:
        print(f"Loading tokenizer for model: '{model_name_or_path}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        print("Tokenizer loaded successfully.")
        return tokenizer
    except OSError:
        print(f"Error: Tokenizer not found for '{model_name_or_path}'. Please check the model ID or path.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading the tokenizer: {e}")
        return None

def count_tokens_in_file(filepath: Path, tokenizer: PreTrainedTokenizer) -> int:
    """Reads a file and returns its token count."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return len(tokenizer.encode(text))
    except FileNotFoundError:
        print(f"Warning: File not found during token counting: {filepath}")
        return 0
    except Exception as e:
        print(f"Warning: Could not read or tokenize file {filepath}: {e}")
        return 0

def process_directory_statistics(base_dir: Path, tokenizer: PreTrainedTokenizer) -> Optional[pd.DataFrame]:
    """
    Walks the directory, counts tokens for each generation, and returns a DataFrame.
    """
    # Regex to capture module name and generation ID from the primary file
    file_pattern = re.compile(r"gen_(\w+)_(\d+)\.v")
    
    generation_data = []
    
    module_paths = [p for p in base_dir.iterdir() if p.is_dir()]
    if not module_paths:
        print(f"Error: No module directories found in '{base_dir}'.")
        return None

    print(f"Found {len(module_paths)} module directories. Processing files...")

    # Use tqdm for a progress bar over modules
    for module_path in tqdm(module_paths, desc="Processing Modules"):
        module_name = module_path.name
        
        # Find all primary generation files (those not ending in _trace.txt)
        primary_files = [f for f in module_path.glob("gen_*.v") if not f.name.endswith("_trace.txt")]

        for primary_file in primary_files:
            match = file_pattern.match(primary_file.name)
            if not match:
                continue

            # The regex ensures the module name from the file matches the directory
            file_module_name, gen_id_str = match.groups()
            if file_module_name != module_name:
                print(f"Warning: File module name '{file_module_name}' does not match directory '{module_name}'. Skipping {primary_file.name}")
                continue
            
            gen_id = int(gen_id_str)
            trace_file = primary_file.with_name(f"gen_{module_name}_{gen_id}_trace.txt")

            output_tokens = count_tokens_in_file(primary_file, tokenizer)
            reasoning_tokens = 0
            
            if trace_file.exists():
                reasoning_tokens = count_tokens_in_file(trace_file, tokenizer)

            total_tokens = output_tokens + reasoning_tokens
            
            generation_data.append({
                "module": module_name,
                "generation_id": gen_id,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": total_tokens,
                "has_reasoning": trace_file.exists()
            })

    if not generation_data:
        print("Error: No valid generation files were found to process.")
        return None

    return pd.DataFrame(generation_data)

def generate_console_report(df: pd.DataFrame, verbose: bool = False) -> None:
    """Prints a formatted summary of the token statistics to the console."""
    
    # --- Module-Level Analysis ---
    module_stats = df.groupby('module')['total_tokens'].agg(['count', 'mean', 'min', 'max'])
    module_stats = module_stats.rename(columns={'count': 'num_generations', 'mean': 'avg_tokens', 'min': 'min_tokens', 'max': 'max_tokens'})
    
    # --- Overall Analysis ---
    num_modules = df['module'].nunique()
    total_generations = len(df)
    avg_gens_per_module = module_stats['num_generations'].mean()
    all_modules_same_gens = module_stats['num_generations'].nunique() == 1
    
    # Overall token stats
    overall_avg = df['total_tokens'].mean()
    overall_min = df['total_tokens'].min()
    overall_max = df['total_tokens'].max()
    
    # Check if any reasoning tokens were found
    has_reasoning_data = df['has_reasoning'].any()

    print("\n" + "="*80)
    print(" " * 25 + "Overall Tokenization Report")
    print("="*80)
    
    print("\n--- Overall Statistics ---")
    print(f"Total Number of Tokens Generated: {df['total_tokens'].sum():,.2f}")
    print(f"Total Number of Modules: {num_modules}")
    print(f"Average Generations per Module: {avg_gens_per_module:.2f}")
    print(f"All modules have same number of generations: {'Yes' if all_modules_same_gens else 'No'}")
    
    print("\n--- Overall Token Counts (per generation) ---")
    print(f"Average: {overall_avg:,.2f} tokens")
    print(f"Minimum: {overall_min:,.0f} tokens")
    print(f"Maximum: {overall_max:,.0f} tokens")

    if has_reasoning_data:
        # Filter for generations that actually have reasoning to avoid skewing averages
        reasoning_df = df[df['has_reasoning']]
        output_avg = reasoning_df['output_tokens'].mean()
        output_min = reasoning_df['output_tokens'].min()
        output_max = reasoning_df['output_tokens'].max()
        
        reasoning_avg = reasoning_df['reasoning_tokens'].mean()
        reasoning_min = reasoning_df['reasoning_tokens'].min()
        reasoning_max = reasoning_df['reasoning_tokens'].max()

        print("\n--- Breakdown for Generations with Trace Files ---")
        print(f"           {'Average':>15} {'Minimum':>15} {'Maximum':>15}")
        print(f"Output:    {output_avg:>15,.2f} {output_min:>15,.0f} {output_max:>15,.0f}")
        print(f"Reasoning: {reasoning_avg:>15,.2f} {reasoning_min:>15,.0f} {reasoning_max:>15,.0f}")

    print("\n" + "="*80)
    if verbose:
        print(" " * 28 + "Module-Level Statistics")
        print("="*80)
        print(module_stats.to_string(float_format="%.2f"))
        print("\n" + "="*80)


def save_histogram_plot(df: pd.DataFrame, output_filename: str):
    """Generates and saves a histogram of the token count distribution."""
    print(f"\nGenerating histogram...")
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax1 = plt.subplots(figsize=(12, 7))

        total_generations = len(df)
        has_reasoning_data = df['has_reasoning'].any()
        # Determine the number of bins dynamically for better visualization
        bins = min(max(10, len(df) // 2), 50)

        if has_reasoning_data:
            # Plot multiple distributions if reasoning data exists
            ax1.hist(df['total_tokens'], bins='auto', alpha=0.6, label='Total Tokens')
            #ax1.hist(df['output_tokens'], bins=bins, alpha=0.8, label='Output Tokens')
            # Only plot reasoning tokens for generations that have them
            #reasoning_tokens = df[df['reasoning_tokens'] > 0]['reasoning_tokens']
            #if not reasoning_tokens.empty:
            #    ax1.hist(reasoning_tokens, bins=bins, alpha=0.8, label='Reasoning Tokens')
            ax1.legend()
        else:
            # Plot a single distribution
            ax1.hist(df['total_tokens'], bins=bins, edgecolor='black', alpha=0.8)

        ax1.set_title('Distribution of Token Counts per Generation', fontsize=16)
        ax1.set_xlabel('Token Count', fontsize=12)
        ax1.set_ylabel('Number of Generations (Frequency)', fontsize=12, color='C0')
        ax1.tick_params(axis='y', labelcolor='C0')


        # Get the current upper limit of the primary y-axis.
        y_max = ax1.get_ylim()[1]
        
        # Round this limit up to the nearest "nice" number (e.g., 10, 50, 100).
        # This gives matplotlib a clean range to work with, preventing odd limits.
        if y_max < 10:
            y_max_rounded = np.ceil(y_max)
        elif y_max < 50:
             y_max_rounded = np.ceil(y_max / 5) * 5
        else:
            y_max_rounded = np.ceil(y_max / 10) * 10
        
        # Explicitly set the y-axis limits for the primary axis.
        ax1.set_ylim(0, y_max_rounded)

        # --- Create and configure the secondary y-axis (Percentage) ---
        ax2 = ax1.twinx()  # Create a second y-axis that shares the same x-axis
        # Set the secondary axis limits to be perfectly proportional to the primary's.
        # Because the ranges are now a clean ratio, matplotlib's automatic
        # tick locator will generate aligned ticks and grids.
        ax2.set_ylim(0, (y_max_rounded / total_generations) * 100)

        # Turn off the grid for the secondary axis to use the primary's grid.
        ax2.grid(False)

        ax1_ticks = ax1.get_yticks()  # Get the ticks from the primary axis
        # Set the secondary axis ticks to be the same as the primary's.
        ax2_ticks = [tick / total_generations * 100 for tick in ax1_ticks if tick <= y_max_rounded]
        ax2.set_yticks(ax2_ticks)  # Set the secondary axis ticks

        
        # Format the tick labels as percentages.
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
        
        ax2.set_ylabel('Percentage of Total Generations', fontsize=12, color='C1')
        ax2.tick_params(axis='y', labelcolor='C1')

        
        # Save the figure to a file
        fig.tight_layout()
        fig.savefig(output_filename, dpi=150)
        plt.close(fig) # Close the plot to free up memory
        print(f"Histogram saved successfully to '{output_filename}'")
    except Exception as e:
        print(f"Error: Could not generate or save histogram plot: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze token counts in a structured directory of generated files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", help="Hugging Face model ID or local path (e.g., 'gpt2').")
    parser.add_argument("--base_dir", help="The base directory containing module subdirectories.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output (module-level stats) for debugging.")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    if not base_dir.is_dir():
        print(f"Error: Base directory not found at '{args.base_dir}'")
        return

    tokenizer = load_tokenizer(args.model)
    if not tokenizer:
        return
        
    # Process files and get the granular DataFrame
    granular_df = process_directory_statistics(base_dir, tokenizer)
    
    if granular_df is None or granular_df.empty:
        print("Processing finished. No data to report.")
        return

    # Generate console report from the DataFrame
    generate_console_report(granular_df, verbose=args.verbose)

    # Save the detailed CSV report
    csv_filename = f"{base_dir.name}_gen_token_statistics.csv"
    histogram_filename = f"{base_dir.name}_token_distribution.png"

    # Save the histogram
    save_histogram_plot(granular_df, histogram_filename)
    print(f"\nSaving histogram plot to '{histogram_filename}'...")
    try:
        granular_df.to_csv(csv_filename, index=False)
        print(f"\nDetailed statistics saved to '{csv_filename}'")
    except Exception as e:
        print(f"\nError: Could not save CSV file: {e}")


if __name__ == "__main__":
    main()