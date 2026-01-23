class OptimizationSuggester:

    def __init__(self, analysis_results):
        self.analysis_results = analysis_results

    def suggest_optimizations(self):
        """Suggests optimizations based on the analysis results.

        Returns:
            dict: A dictionary where keys are file paths and values are lists
                  of optimization suggestions.
        """
        suggestions = {}
        for result in self.analysis_results:
            file_path = result['file']
            suggestions[file_path] = []

            if result['potential_bottlenecks']:
                for item in result['large_data_structures']:
                    line_content = item['line_content']
                    if 'pd.read_csv' in line_content:
                        suggestions[file_path].append(
                            "Consider using chunking or dtypes to reduce memory usage when reading the CSV file."
                        )
                    elif 'np.load' in line_content:
                        suggestions[file_path].append(
                            "Consider memory mapping or using smaller data types when loading the NumPy array."
                        )
                    elif '[]' in line_content or '{}' in line_content:
                        suggestions[file_path].append(
                            "Review the creation and usage of this list/dictionary; consider using generators, iterators, or more memory-efficient data structures."
                        )

            if result['memory_usage'] > 100 * 1024 * 1024:  # Example threshold: 100MB
                suggestions[file_path].append(
                    "High memory usage detected. Profile the code to identify specific memory-intensive operations."
                )
        return suggestions

# Example usage (add this to a main function or test script):
if __name__ == '__main__':
    from memory_analyzer import MemoryAnalyzer # Assuming memory_analyzer.py is in the same directory

    # Assuming the script is run from the project root
    analyzer = MemoryAnalyzer('.')  # '.' represents the current directory (project root)
    analysis_results = analyzer.analyze_project()

    suggester = OptimizationSuggester(analysis_results)
    optimization_suggestions = suggester.suggest_optimizations()

    for file_path, suggestions in optimization_suggestions.items():
        print(f"Optimization suggestions for: {file_path}")
        if suggestions:
            for suggestion in suggestions:
                print(f"  - {suggestion}")
        else:
            print("  No specific memory optimization suggestions found.")
        print("\n")