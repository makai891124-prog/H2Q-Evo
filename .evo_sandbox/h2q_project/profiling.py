import cProfile
import pstats
import io


def profile_function(func, *args, sort_by='cumulative', output_file='profile_output.txt'):
    """Profiles a function and saves the results to a file.

    Args:
        func: The function to profile.
        *args: The arguments to pass to the function.
        sort_by: The sorting criteria for the profiling results.
                 Valid options are 'cumulative', 'time', 'calls'.
        output_file: The name of the file to save the profiling results to.
    """
    pr = cProfile.Profile()
    pr.enable()
    func(*args)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(sort_by)
    ps.print_stats()

    with open(output_file, 'w') as f:
        f.write(s.getvalue())

    print(f"Profiling results saved to {output_file}")


if __name__ == '__main__':
    # Example Usage (replace with your actual functions)
    def example_quaternion_calculation():
        # Replace with your actual quaternion calculation code
        result = 1 + 2
        return result

    def example_fractal_generation():
        # Replace with your actual fractal generation code
        result = 3 + 4
        return result

    profile_function(example_quaternion_calculation, sort_by='cumulative', output_file='quaternion_profile.txt')
    profile_function(example_fractal_generation, sort_by='cumulative', output_file='fractal_profile.txt')
