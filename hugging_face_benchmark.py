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


ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""

history_of_AI = """
During World War II Turing was a leading cryptanalyst at the Government Code and Cypher School in Bletchley Park, Buckinghamshire, England. Turing could not turn to the project of building a stored-program electronic computing machine until the cessation of hostilities in Europe in 1945. Nevertheless, during the war he gave considerable thought to the issue of machine intelligence. One of Turing’s colleagues at Bletchley Park, Donald Michie (who later founded the Department of Machine Intelligence and Perception at the University of Edinburgh), later recalled that Turing often discussed how computers could learn from experience as well as solve new problems through the use of guiding principles—a process now known as heuristic problem solving. Turing gave quite possibly the earliest public lecture (London, 1947) to mention computer intelligence, saying, “What we want is a machine that can learn from experience,” and that the “possibility of letting the machine alter its own instructions provides the mechanism for this.” In 1948 he introduced many of the central concepts of AI in a report entitled “Intelligent Machinery.” However, Turing did not publish this paper, and many of his ideas were later reinvented by others. For instance, one of Turing’s original ideas was to train a network of artificial neurons to perform specific tasks, an approach described in the section Connectionism.
Chess At Bletchley Park Turing illustrated his ideas on machine intelligence by reference to chess—a useful source of challenging and clearly defined problems against which proposed methods for problem solving could be tested. In principle, a chess-playing computer could play by searching exhaustively through all the available moves, but in practice this is impossible because it would involve examining an astronomically large number of moves. Heuristics are necessary to guide a narrower, more discriminative search. Although Turing experimented with designing chess programs, he had to content himself with theory in the absence of a computer to run his chess program. The first true AI programs had to await the arrival of stored-program electronic digital computers.
"""

paper_on_evolution = """
Our individual and collective health is shaped and affected by many factors. These factors include our environment, our inherited and somatic genetic variants, our variable exposure to pathogens, our diets and lifestyles, our social systems, and our cultural innovations.
None of these factors are static, and they all interact with each other. Human genetic adaptations to our past environments, disease burdens, and cultural practices can affect disease risks today, especially if any of the underlying environmental, disease, or cultural factors have changed in the interim. Meanwhile, human pathogens and parasites continually adapt to our biology and to cultural innovations, including advances in medicine, the development of new drugs, and infrastructure improvements (such water-treatment plants or the availability of mosquito nets). The progression of cancer within an individual is also often viewed as an evolutionary process (Merlo et al., 2006).
Growing numbers of scientists are applying evolutionary theory to study these interactions across different timescales and their impacts on modern human health, including with predictions of how our health might be affected by these processes in the future and how we can take informed action. This field of study is known as evolutionary medicine (Stearns and Medzhitov, 2015). To help set the stage for a special issue of eLife on evolutionary medicine, I will highlight a subset of concepts and research approaches in this field.
One area of particularly active research is studying how various pathogens and parasites evolve within and among human hosts, between human and non-human hosts and/or vectors, and in response to drug treatments. Work on the evolution of bacterial pathogen resistance to antibiotics (Bakkeren et al., 2020), virus resistance to antivirals (Irwin et al., 2016), fungal resistance to antifungals (Robbins et al., 2017), and parasite resistance to other antimicrobials (e.g., Haldar et al., 2018) is understandably prominent. Yet the evolutionary forces of mutation, genetic drift and gene flow (including that mediated by human host, animal host, and insect vectors movement and behavior), and the interplay between these forces and human immunity, are also critical components of human infectious disease dynamics.
"""

# I could not run all at the same time
benchmark_summarization(ARTICLE)
# benchmark_summarization(history_of_AI)
# benchmark_summarization(paper_on_evolution)
