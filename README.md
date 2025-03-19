# Using Large Language Models (LLMs) for Text Summarization
## Introduction

This assignment explores text summarization using Large Language Models (LLMs) available through Hugging Face Transformers. The goal is to implement and evaluate an LLM-based summarization model for generating concise summaries from provided text data.

---

## Objective

Utilize a generative AI model to create succinct summaries of given texts with Hugging Face Transformers.

### Methodology

1. Identify and select a summarization model from Hugging Face.
2. Implement a Python script using Hugging Face's pipeline to:
   - Accept a lengthy text input.
   - Generate a concise summary using the chosen AI model.
   - Display both original and summarized texts.

---

## Selected Model

The chosen summarization models are:

- **Model Name:** [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)
    - **Why?:** Highest popularity on Hugging Face


- **Model Name:** [sshleifer/distilbart-cnn-12-6](https://huggingface.co/sshleifer/distilbart-cnn-12-6)
    - **Why?:** Distilled version of BART and interesting to see comparison


- **Model Name:** [google/pegasus-xsum](https://huggingface.co/google/pegasus-xsum)
    - **Why?:** Google is another big player in the AI space.

---

## Code Implementation

Below is the Python script used for generating summaries and benchmarking:

```python
import time
import tracemalloc
from transformers import pipeline

# List of models to test
MODEL_NAMES = [
    "google/pegasus-xsum",
    "facebook/bart-large-cnn",
    "sshleifer/distilbart-cnn-12-6"
]

def benchmark_summarization(text, max_length=100, min_length=25, batch_size=1):
    """
    Benchmarks the execution time and memory usage of different summarization models.

    Parameters:
        text (str): The input text to summarize.
        max_length (int): Maximum length of summary.
        min_length (int): Minimum length of summary.
        batch_size (int): Batch size for the pipeline.

    Prints:
        A summary of performance metrics for each model.
    """
    results = []

    for model_name in MODEL_NAMES:
        print(f"\nBenchmarking model: {model_name}")
        
        # Load the summarization pipeline
        summarizer = pipeline("summarization", model=model_name, device=0)  # Use GPU if available

        # Start memory tracking
        tracemalloc.start()
        
        # Start time tracking
        start_time = time.perf_counter()

        # Run summarization
        summary = summarizer(text, max_length=max_length, min_length=min_length, batch_size=batch_size)

        # Stop time tracking
        end_time = time.perf_counter()
        
        # Stop memory tracking
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Store results
        results.append({
            "Model": model_name,
            "Execution Time (s)": end_time - start_time,
            "Memory Usage (MB)": peak / 10**6,
            "Summary": summary[0]['summary_text']
        })

    # Print results
    print("\n===== Benchmarking Results =====")
    for result in results:
        print(f"Model: {result['Model']}")
        print(f"Execution Time: {result['Execution Time (s)']:.4f} seconds")
        print(f"Memory Usage: {result['Memory Usage (MB)']:.2f} MB")
        print(f"Summary Output: {result['Summary']}\n")

```
---
## Benchmarking Results

### Case 1: [News article](https://huggingface.co/facebook/bart-large-cnn)

| Model                           | Execution Time (s) | Memory Usage (MB) | Summary Output |
|---------------------------------|--------------------|-------------------|----------------|
| google/pegasus-xsum             | 7.5134             | 0.12              | "A New York woman who has been married 10 times is facing criminal charges for allegedly lying on a marriage license application, prosecutors say." |
| facebook/bart-large-cnn         | 6.6450             | 0.16              | "Liana Barrientos, 39, is charged with two counts of \"offering a false instrument for filing in the first degree\" In total, she has been married 10 times, with nine of her marriages occurring between 1999 and 2002. She is believed to still be married to four men." |
| sshleifer/distilbart-cnn-12-6   | 4.6523             | 0.10              | "Liana Barrientos, 39, is charged with two counts of \"offering a false instrument for filing in the first degree\" In total, she has been married 10 times, with nine of her marriages occurring between 1999 and 2002. At one time, she was married to eight men at once, prosecutors say." |

### Case 2: First paragraph of [History of artificial intelligence (AI)](https://www.britannica.com/science/history-of-artificial-intelligenc )

| Model                           | Execution Time (s) | Memory Usage (MB) | Summary Output |
|---------------------------------|--------------------|-------------------|----------------|
| google/pegasus-xsum             | 6.4421             | 0.11              | "Alan Turing was one of the founding fathers of artificial intelligence (AI), and his ideas on machine intelligence are still relevant today." |
| facebook/bart-large-cnn         | 6.7917             | 0.17              | "During World War II Turing was a leading cryptanalyst at the Government Code and Cypher School in Bletchley Park, Buckinghamshire, England. During the war he gave considerable thought to the issue of machine intelligence. In 1948 he introduced many of the central concepts of AI in a report entitled “Intelligent Machinery”." |
| sshleifer/distilbart-cnn-12-6   | 3.9668             | 0.13              | "During World War II Turing was a leading cryptanalyst at the Government Code and Cypher School in Bletchley Park, Buckinghamshire, England. Turing could not turn to the project of building a stored-program electronic computing machine until the cessation of hostilities in Europe in 1945." |

### Case 3: First paragraph of [Evolutionary Medicine](https://elifesciences.org/articles/69398)

| Model                           | Execution Time (s) | Memory Usage (MB) | Summary Output |
|---------------------------------|--------------------|-------------------|----------------|
| google/pegasus-xsum             | 7.0905             | 0.12              | "Evolutionary medicine aims to understand the forces that shape human health and disease, and how they might be affected in the future." |
| facebook/bart-large-cnn         | 4.9632             | 0.13              | "Our individual and collective health is shaped and affected by many factors. These factors include our environment, our inherited and somatic genetic variants, our diets and lifestyles. Human pathogens and parasites continually adapt to our biology and to cultural innovations." |
| sshleifer/distilbart-cnn-12-6   | 5.3459             | 0.13              | "Our individual and collective health is shaped and affected by many factors. These include our environment, our inherited and somatic genetic variants, our variable exposure to pathogens, our diets and lifestyles, our social systems, and our cultural innovations. Human genetic adaptations to our past environments, disease burdens, and cultural practices can affect disease risks today. Meanwhile, human pathogens and parasites continually adapt to our biology." |

---

## Conclusions 

We observed relatively long execution times across all three models, primarily due to running benchmarks on a CPU (Ryzen 7). However, memory usage remained low. The `google/pegasus-xsum` model consistently showed the longest execution time yet produced the shortest summaries. Conversely, both `facebook/bart-large-cnn` and its distilled version, `sshleifer/distilbart-cnn-12-6`, were generally faster and produced similar output. This similarity is expected since the distilled version is a compressed variant of the original model.

Given its shorter execution times and comparable output quality, the `sshleifer/distilbart-cnn-12-6` model appears preferable for CPU-based scenarios. Additional considerations for selecting models include their overall size and performance when executed on GPUs.

**Additional Note:**  
It's uncertain how effectively the current memory benchmarking results translate to GPU-based VRAM usage. Further GPU-based benchmarks would provide better insight into this aspect.

## Included Prompts

ChatGPT 4o was used for writing and coding. With GPT initialized with the following prompts. 


### Initial Documentation Prompt

The initial prompt for documenting this project was:

```markdown
You are an expert in Markdown formatting, technical summarization, and documentation. Your task is to generate a well-structured Markdown report for a small code assignment on using Large Language Models (LLMs) for text summarization.

Instructions:

Input: I will provide:
- The task description
- Code snippets
- Results
- Additional prompts that should be included in the documentation

Processing:
- Summarize the purpose and methodology of the assignment.
- Clearly document the provided code, explaining its functionality.
- Summarize the results, highlighting key observations.
- Incorporate any provided prompts into the Markdown output under a dedicated section.
- Format the report in Markdown, using proper headings, code blocks, and bullet points.

Output: Return a well-structured Markdown document containing:
- A title and brief introduction
- Code explanation with proper formatting
- A summary of results and conclusions
- A section for included prompts and their purpose
- Any relevant insights or next steps

Ensure that the output is clear, concise, and formatted for easy readability. Use fenced code blocks (```) for code snippets and proper Markdown structure for clarity.
```

**Purpose of Included Prompt:** To ensure clear, structured, and comprehensive documentation of the summarization task.

### Python Command-Line Expert Prompt

The initial prompt for code writing in this project was:

```markdown
*"You are an expert in Python 3.11 with deep knowledge of performance benchmarking for both execution time and memory usage. You are also proficient in the Hugging Face `pipeline` package for NLP tasks. Your task is to analyze, optimize, and benchmark Python functions, ensuring efficient execution in terms of both speed and memory consumption.  

Given a Python function, provide:  
1. A clear breakdown of potential performance bottlenecks.  
2. Efficient benchmarking using modules like `timeit`, `perf_counter`, and `memory_profiler`.  
3. Optimized alternatives for improving speed and memory efficiency.  
4. Best practices for integrating Hugging Face `pipeline` efficiently within performance-critical applications.  

Ensure that your responses include clear, well-commented Python code, explanations of optimizations, and practical insights on real-world applications."*  
```

**Purpose of Included Prompt:** To ensure structured, efficient, and professional guidance when developing Python command-line programs.

---
