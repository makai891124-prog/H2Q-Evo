from h2q_project.optimization.memory_optimizer import MemoryOptimizer

def my_function(data):
    my_list = [i for i in range(data)]
    return my_list

if __name__ == '__main__':
    optimizer = MemoryOptimizer(my_function, 1000000)
    memory_usage, return_value = optimizer.run_and_profile()
    print(f"Memory Usage: {memory_usage} MB")
    print(f"Return Value: {return_value[:10]}...") #print first 10 elements