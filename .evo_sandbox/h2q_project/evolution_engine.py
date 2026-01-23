import logging

# Configure logging (basic example, can be extended)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EvolutionEngine:
    def __init__(self, initial_population, fitness_function, mutation_rate=0.01):
        self.population = initial_population
        self.fitness_function = fitness_function
        self.mutation_rate = mutation_rate

    def evolve(self, generations=10):
        for generation in range(generations):
            logging.info(f"Starting generation {generation + 1}")

            # 1. Evaluate fitness
            fitness_scores = [self.fitness_function(individual) for individual in self.population]

            # 2. Selection (example: select top 50%)
            num_to_select = len(self.population) // 2
            selected_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:num_to_select]
            selected_population = [self.population[i] for i in selected_indices]

            logging.debug(f"Selected {len(selected_population)} individuals for breeding")

            # 3. Crossover (example: pair up selected individuals and create offspring)
            offspring = []
            for i in range(0, len(selected_population), 2):
                if i + 1 < len(selected_population):
                    parent1 = selected_population[i]
                    parent2 = selected_population[i + 1]
                    # Simple crossover: average the parents (assuming they are numeric lists/arrays)
                    offspring.append([(p1 + p2) / 2 for p1, p2 in zip(parent1, parent2)])

            logging.debug(f"Created {len(offspring)} offspring through crossover")

            # 4. Mutation
            for i in range(len(offspring)):
                for j in range(len(offspring[i])):
                    if random.random() < self.mutation_rate:
                        offspring[i][j] += random.uniform(-1, 1) # Example mutation: add a random value

            logging.debug(f"Mutated offspring")

            # 5. Replacement (replace the old population with the new generation)
            self.population = selected_population + offspring

            logging.info(f"Finished generation {generation + 1}")

        logging.info("Evolution complete.")
        return self.population

import random
if __name__ == '__main__':
    # Example usage (replace with your actual problem and data)
    initial_population = [[random.random() for _ in range(5)] for _ in range(10)]  # 10 individuals with 5 genes each

    def fitness_function(individual):
        # Example fitness: sum of squares (higher is better)
        return sum(x**2 for x in individual)

    engine = EvolutionEngine(initial_population, fitness_function, mutation_rate=0.05)
    final_population = engine.evolve(generations=20)

    print("Final Population:", final_population)
