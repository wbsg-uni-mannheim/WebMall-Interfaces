

def calculation_results(benchmark_solutions, model_solution):
    """
    Calculate task completion, precision, and recall metrics.

    Args:
        benchmark_solutions: List of strings containing benchmark solutions
        model_solution: List of strings containing model solution

    Returns:
        dict: Contains task_completion_rate, avg_precision, avg_recall, f1_score
    """
    # Convert lists to sets of complete strings, not individual characters
    if isinstance(benchmark_solutions, list):
        benchmark_set = set(benchmark_solutions)
    elif isinstance(benchmark_solutions, set):
        benchmark_set = benchmark_solutions
    else:
        benchmark_set = set([str(benchmark_solutions)])

    if isinstance(model_solution, list):
        model_set = set(model_solution)
    elif isinstance(model_solution, set):
        model_set = model_solution
    else:
        model_set = set([str(model_solution)])

    # Task completion: 1 if exact match, 0 otherwise
    task_completion = 1 if benchmark_set == model_set else 0

    # Precision: intersection / model_set size
    if len(model_set) > 0:
        precision = len(benchmark_set.intersection(model_set)) / len(model_set)
    else:
        precision = 0.0

    # Recall: intersection / benchmark_set size
    if len(benchmark_set) > 0:
        recall = len(benchmark_set.intersection(
            model_set)) / len(benchmark_set)
    else:
        recall = 0.0

    # Calculate F1 score with zero division protection
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return {
        'task_completion_rate': task_completion,
        'avg_precision': precision,
        'avg_recall': recall,
        'f1_score': f1_score
    }
