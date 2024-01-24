use rand::{thread_rng, Rng};
use rand::prelude::*;
use rand::distributions::{Distribution, WeightedIndex};
use std::io;

#[derive(Debug, Clone)]
struct Operation {
    job_id: usize,
    machine_id: usize,
    processing_time: usize,
    start_time: Option<usize>,
}

struct ConflictGraph {
    edges: Vec<(usize, usize)>,
}

#[derive(Debug, Clone)]
struct Chromosome {
    operations: Vec<Operation>,
    makespan: usize,
}

#[derive(Debug)]
struct Population {
    chromosomes: Vec<Chromosome>,
}

#[derive(Debug)]
struct GAParameters {
    mutation_probability: f64,
    max_iterations: usize,
}

impl GAParameters {
    pub fn new(mutation_probability: f64, max_iterations: usize) -> Self {
        GAParameters {
            mutation_probability,
            max_iterations,
        }
    }
}

struct GeneticAlgorithm {
    population: Population,
    conflict_graph: ConflictGraph,
    params: GAParameters,
}

impl Chromosome {
    // Initialize with a sequential (inefficient) schedule
    fn initialize(operations: Vec<Operation>) -> Chromosome {
        let mut operations = operations.clone();
        operations.sort_by(|a, b| a.job_id.cmp(&b.job_id).then(a.machine_id.cmp(&b.machine_id)));

        let mut current_time = 0;
        for operation in &mut operations {
            operation.start_time = Some(current_time);
            current_time += operation.processing_time;
        }

        let makespan = operations.last().map(|op| op.start_time.unwrap() + op.processing_time).unwrap_or(0);

        Chromosome {
            operations,
            makespan,
        }
    }

    // Implementations for One_point_crossover, linear_order_crossover, order_crossover, swap_mutation, and move_mutation
    fn one_point_crossover(parent1: &Chromosome, parent2: &Chromosome) -> (Chromosome, Chromosome) {
        let mut rng = thread_rng();
        let crossover_point = rng.gen_range(0..parent1.operations.len());
        
        let child1_operations = parent1.operations[..crossover_point]
            .to_vec()
            .into_iter()
            .chain(parent2.operations[crossover_point..].to_vec().into_iter())
            .collect::<Vec<Operation>>();

        let child2_operations = parent2.operations[..crossover_point]
            .to_vec()
            .into_iter()
            .chain(parent1.operations[crossover_point..].to_vec().into_iter())
            .collect::<Vec<Operation>>();

        // Assuming we have a function to calculate the makespan
        let child1_makespan = calculate_makespan(&child1_operations);
        let child2_makespan = calculate_makespan(&child2_operations);

        (
            Chromosome { operations: child1_operations, makespan: child1_makespan },
            Chromosome { operations: child2_operations, makespan: child2_makespan },
        )
    }

    fn linear_order_crossover(parent1: &Chromosome, parent2: &Chromosome) -> Chromosome {
        let mut rng = thread_rng();
        let length = parent1.operations.len();
    
        let start = rng.gen_range(0..length);
        let end = rng.gen_range(start..length);
    
        let mut child_operations = vec![None; length];
        let mut parent2_operations = parent2.operations.clone();
    
        for i in start..end {
            child_operations[i] = Some(parent1.operations[i].clone());
            parent2_operations.retain(|op| !(op.job_id == parent1.operations[i].job_id && op.machine_id == parent1.operations[i].machine_id));
        }
    
        let mut parent2_index: usize = 0;
        for i in 0..length {
            if child_operations[i].is_none() {
                child_operations[i] = Some(parent2_operations[parent2_index].clone());
                parent2_index += 1;
            }
        }
    
        let operations = child_operations.into_iter().map(|op| op.unwrap()).collect();
        let makespan = calculate_makespan(&operations);
    
        Chromosome { operations, makespan }
    }
    
    fn order_crossover(parent1: &Chromosome, parent2: &Chromosome) -> Chromosome {
        let mut rng = thread_rng();
        let length = parent1.operations.len();
    
        let start = rng.gen_range(0..length);
        let end = rng.gen_range(start..length);
    
        let mut child_operations = vec![None; length];
        let mut parent2_operations = parent2.operations.clone();
    
        for i in start..end {
            child_operations[i] = Some(parent1.operations[i].clone());
            parent2_operations.retain(|op| !(op.job_id == parent1.operations[i].job_id && op.machine_id == parent1.operations[i].machine_id));
        }
    
        let mut parent2_index = 0;
        for i in end..end+length {
            let idx = i % length;
            if child_operations[idx].is_none() {
                child_operations[idx] = Some(parent2_operations[parent2_index].clone());
                parent2_index += 1;
            }
        }
    
        let operations = child_operations.into_iter().map(|op| op.unwrap()).collect();
        let makespan = calculate_makespan(&operations);
    
        Chromosome { operations, makespan }
    }

    fn swap_mutation(&mut self) {
        let mut rng = thread_rng();
        let length = self.operations.len();
        let swap_indices = (rng.gen_range(0..length), rng.gen_range(0..length));

        let temp = self.operations[swap_indices.0].start_time;
        self.operations[swap_indices.0].start_time = self.operations[swap_indices.1].start_time;
        self.operations[swap_indices.1].start_time = temp;

        for i in swap_indices.0.min(swap_indices.1)+1..swap_indices.0.max(swap_indices.1) {
            if self.operations[i].start_time.is_some() {
                self.operations[i].start_time = Some(self.operations[i-1].start_time.unwrap() + self.operations[i-1].processing_time);
            }
        }
        self.operations.swap(swap_indices.0, swap_indices.1);
        self.makespan = calculate_makespan(&self.operations);
    }

    fn move_mutation(&mut self) {
        let mut rng = thread_rng();
        let length = self.operations.len();
        
        let from_index = rng.gen_range(0..length);
        let to_index = rng.gen_range(0..length);
        
        if from_index != to_index {

            self.operations[from_index].start_time = self.operations[to_index].start_time;
            let operation = self.operations.remove(from_index);
            self.operations.insert(to_index, operation);

            for i in from_index.min(to_index)+1..from_index.max(to_index) {
                if self.operations[i].start_time.is_some() {
                    self.operations[i].start_time = Some(self.operations[i-1].start_time.unwrap() + self.operations[i-1].processing_time);
                }
            }
        }
    
        self.makespan = calculate_makespan(&self.operations);
    }
    

    // Function to check if a chromosome is redundant
    fn is_redundant(&self, conflict_graph: &ConflictGraph) -> bool {
        // Check for repeated operations
        let mut operation_set = std::collections::HashSet::new();
        for operation in &self.operations {
            if !operation_set.insert((operation.job_id, operation.machine_id)) {
                return true;
            }
        }

        // Check for conflicting operations according to the conflict graph
        for &(op1, op2) in &conflict_graph.edges {
            let mut times = self.operations.iter()
                .filter(|&op| op.machine_id == op1 || op.machine_id == op2)
                .map(|op| (op.start_time.unwrap(), op.start_time.unwrap() + op.processing_time))
                .collect::<Vec<_>>();
            times.sort_by_key(|k| k.0);
            for pair in times.windows(2) {
                if let &[(_, end_previous), (start_next, _)] = pair {
                    if start_next < end_previous {
                        return true;
                    }
                }
            }
        }

        // Check for overlapping operations on machines
        let mut machine_timelines = std::collections::HashMap::new();
        for operation in &self.operations {
            machine_timelines.entry(operation.machine_id)
                .or_insert_with(Vec::new)
                .push((operation.start_time.unwrap(), operation.start_time.unwrap() + operation.processing_time));
        }

        for (_, times) in machine_timelines {
            let mut times = times;
            times.sort_by_key(|k| k.0);
            for pair in times.windows(2) {
                if let &[(_, end_previous), (start_next, _)] = pair {
                    if start_next < end_previous {
                        return true;
                    }
                }
            }
        }

        // If none of the above checks indicate redundancy, the chromosome is not redundant
        false
    }


}

impl Population {
    fn sort_by_fitness(&mut self) {
        self.chromosomes.sort_by(|a, b| a.makespan.cmp(&b.makespan));
    }

    fn select_parent(&self) -> &Chromosome {
        let mut rng = thread_rng();
        let weights: Vec<_> = (1..=self.chromosomes.len()).map(|rank| 2.0 / (rank as f64)).collect();
        let dist = WeightedIndex::new(&weights).unwrap();
        &self.chromosomes[dist.sample(&mut rng)]
    }

    fn replace_chromosome(&mut self, new_chromosome: Chromosome) {
        let mut rng = thread_rng();
        let median_rank = self.chromosomes.len() / 2;
        let replace_index = rng.gen_range(median_rank..self.chromosomes.len());
        self.chromosomes[replace_index] = new_chromosome;
    }
}

impl GeneticAlgorithm {
    pub fn new(initial_population: Population, conflict_graph: ConflictGraph, params: GAParameters) -> Self {
        GeneticAlgorithm {
            population: initial_population,
            conflict_graph,
            params,
        }
    }

    pub fn run(&mut self) {
        for _ in 0..self.params.max_iterations {
            self.population.sort_by_fitness();
    
            // Genetic algorithm operations such as crossover, mutation, etc.
            let parent1 = self.population.select_parent();
            let parent2 = self.population.select_parent();
    
            let (child1,child2) = Chromosome::one_point_crossover(&parent1, &parent2);
            let (child3,child4) = (Chromosome::linear_order_crossover(&parent1, &parent2),Chromosome::linear_order_crossover(&parent2, &parent1));
            let (child5,child6) = (Chromosome::order_crossover(&parent1, &parent2),Chromosome::order_crossover(&parent2, &parent1));
    
            let children = vec![child1, child2, child3, child4, child5, child6];
    
            let mut non_redundant_children = children.into_iter()
                .filter(|child| !child.is_redundant(&self.conflict_graph))
                .collect::<Vec<_>>();
            let mut rng = rand::thread_rng();
    
            if non_redundant_children.len() >= 1 {
                let idx1 = (0..non_redundant_children.len()).choose(&mut rng).unwrap();
                
                if rng.gen::<f64>() < self.params.mutation_probability {
                    let mut child = non_redundant_children[idx1].clone();
                    child.swap_mutation();
                    if !child.is_redundant(&self.conflict_graph) {
                        non_redundant_children[idx1] = child;
                    }
                }
                
                    if non_redundant_children.len() >= 2 {
                        let idx2 = (0..non_redundant_children.len()).choose(&mut rng).unwrap();
                        if rng.gen::<f64>() < self.params.mutation_probability {
                            let mut child = non_redundant_children[idx2].clone();
                            child.move_mutation();
                            if !child.is_redundant(&self.conflict_graph) {
                                non_redundant_children[idx2] = child;
                            }
                        }
                    }
            }
    
            for non_redundant_child in non_redundant_children {
                self.population.replace_chromosome(non_redundant_child);
            }

        }
    }
}

fn calculate_makespan(operations: &Vec<Operation>) -> usize {
    let mut machine_end_times = std::collections::HashMap::new();

    for operation in operations {
        let machine_id = operation.machine_id;
        let start_time = machine_end_times.get(&machine_id).cloned().unwrap_or(0).max(operation.start_time.unwrap_or(0));
        let end_time = start_time + operation.processing_time;

        machine_end_times.insert(machine_id, end_time);
    }

    *machine_end_times.values().max().unwrap_or(&0)
}

fn generate_operations(num_jobs: usize, num_machines: usize, max_processing_time: usize) -> Vec<Operation> {
    let mut rng = thread_rng();
    let mut operations = Vec::new();

    for job_id in 1..=num_jobs {
        for machine_id in 1..=num_machines {
            let processing_time = rng.gen_range(1..=max_processing_time);
            operations.push(Operation { job_id, machine_id, processing_time, start_time: None });
        }
    }

    operations
}

fn generate_conflict_graph(num_jobs: usize, num_conflicts: usize) -> ConflictGraph {
    let mut rng = thread_rng();
    let mut conflicts = Vec::new();

    for _ in 0..num_conflicts {
        let job1 = rng.gen_range(1..=num_jobs);
        let mut job2 = rng.gen_range(1..=num_jobs);
        while job1 == job2 {
            job2 = rng.gen_range(1..=num_jobs);
        }
        conflicts.push((job1, job2));
    }

    ConflictGraph { edges: conflicts }
}

fn main() {
    // Function to read user input and parse it into a number
    fn read_input<T: std::str::FromStr>() -> T where <T as std::str::FromStr>::Err: std::fmt::Debug {
        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Failed to read line");
        input.trim().parse().expect("Please enter a valid number")
    }

    println!("Enter the number of jobs (e.g. 3):");
    let num_jobs: usize = read_input();

    println!("Enter the number of machines (e.g. 3):");
    let num_machines: usize = read_input();

    println!("Enter the maximum processing time (e.g. 3):");
    let max_processing_time: usize = read_input();

    println!("Enter the number of conflicts (e.g. 2):");
    let num_conflicts: usize = read_input();

    println!("Enter the mutation probability (0.0 to 1.0):");
    let mutation_probability: f64 = read_input();

    println!("Enter the number of iterations for the genetic algorithm (e.g. 10000):");
    let max_iterations: usize = read_input();

    println!("Enter the population size (e.g. 50):");
    let population_size: usize = read_input();

    let operations = generate_operations(num_jobs, num_machines, max_processing_time);
    let conflict_graph = generate_conflict_graph(num_jobs, num_conflicts);

    let ga_params = GAParameters::new(mutation_probability, max_iterations);

    let mut initial_population = Population { chromosomes: Vec::new() };

    for _ in 0..population_size {
        let chromosome = Chromosome::initialize(operations.clone());
        initial_population.chromosomes.push(chromosome);
    }

    let mut ga = GeneticAlgorithm::new(initial_population, conflict_graph, ga_params);

    let start_solution = ga.population.chromosomes.iter().min_by_key(|c| c.makespan);
    if let Some(start_best) = start_solution {
        println!("Worst Solution: {:?}", start_best.makespan)
    }

    println!("running!");
    ga.run();

    let best_solution = ga.population.chromosomes.iter().min_by_key(|c| c.makespan);
    if let Some(best) = best_solution {
        println!("Best GA Solution: {:?}", best.makespan);
    }
}


