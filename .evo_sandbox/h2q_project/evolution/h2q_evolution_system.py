from h2q_project.evolution.base_evolution_system import BaseEvolutionSystem
import gc

class H2QEvolutionSystem(BaseEvolutionSystem):
    def __init__(self, initial_population, evaluator, selector, crossover, mutation, generations):
        super().__init__(initial_population, evaluator, selector, crossover, mutation, generations)

    def run(self):
        population = self.initial_population

        for generation in range(self.generations):
            print(f"Generation {generation + 1}/{self.generations}")

            # Evaluate the population
            evaluated_population = self.evaluator.evaluate(population)

            # Select the best individuals
            selected_population = self.selector.select(evaluated_population)

            # Create the next generation through crossover and mutation
            new_population = []
            for i in range(0, len(selected_population), 2):
                parent1 = selected_population[i]
                if i + 1 < len(selected_population):
                    parent2 = selected_population[i + 1]
                    children = self.crossover.crossover(parent1, parent2)
                    new_population.extend(children)
                else:
                    new_population.append(parent1)

            mutated_population = [self.mutation.mutate(individual) for individual in new_population]

            population = mutated_population

            # Explicitly trigger garbage collection to free up memory
            gc.collect()

        # Evaluate the final population
        evaluated_population = self.evaluator.evaluate(population)

        # Select the best individual from the final population
        best_individual = self.selector.select(evaluated_population)[0]

        return best_individual
