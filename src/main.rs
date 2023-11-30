use rand::seq::SliceRandom;  // Import the trait required for the shuffle method
use rand::{thread_rng, Rng};        // Import the thread_rng function for a random number generator
use rand::prelude::*;
use rand::distributions::{Distribution, WeightedIndex}; // Import the WeightedIndex function for weighted random selection

// Define the structure for an Operation
#[derive(Debug, Clone)]
struct Operation {
    job_id: usize,
    machine_id: usize,
    processing_time: usize,
    start_time: Option<usize>, // Added to track when the operation starts
}

struct ConflictGraph {
    edges: Vec<(usize, usize)>, // Represents conflicts between machine_id
}

// Define the structure for a Chromosome
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
    // ... other parameters
}

// Genetic algorithm functions
impl Chromosome {

    // Function to initialize a chromosome with a randomised schedule with aid
    fn initialize(operations: Vec<Operation>, conflict_graph: &ConflictGraph) -> Chromosome {
        let mut shuffled_operations = operations.clone();
        // Shuffle operations based on a simple heuristic (e.g., job_id)
        shuffled_operations.shuffle(&mut thread_rng());

        let mut machine_end_times = std::collections::HashMap::new();
        let mut job_end_times = std::collections::HashMap::new();
        
        for operation in &mut shuffled_operations {
            let machine_id = operation.machine_id;
            let job_id = operation.job_id;

            // Determine the earliest start time for this operation, based on machine availability
            let mut start_time = machine_end_times.get(&machine_id).cloned().unwrap_or(0);

            // Check for conflicts and adjust start time if necessary
            for &(preceding_machine_id, conflicting_machine_id) in &conflict_graph.edges {
                if preceding_machine_id == machine_id || conflicting_machine_id == machine_id {
                    // The start time for the operation must be after the end time of the job it conflicts with
                    let end_time_of_preceding_machine = machine_end_times.get(&preceding_machine_id).cloned().unwrap_or(0);
                    let end_time_of_conflicting_machine = machine_end_times.get(&conflicting_machine_id).cloned().unwrap_or(0);

                    if start_time < end_time_of_preceding_machine || start_time < end_time_of_conflicting_machine {
                        // Adjust start time to after the preceding job has finished
                        operation.start_time = std::cmp::max(Some(end_time_of_preceding_machine),Some(end_time_of_conflicting_machine));
                        start_time = operation.start_time.unwrap();
                    }
                }
            }

            // If the start time has not been adjusted due to conflict, set it now
            if operation.start_time.is_none() {
                operation.start_time = Some(start_time);
            }

            // Update the end time for this job and machine
            let end_time = operation.start_time.unwrap() + operation.processing_time;
            machine_end_times.insert(machine_id, end_time);
            job_end_times.insert(job_id, end_time);
        }

        // The makespan is the maximum end time among all jobs
        let makespan = job_end_times.values().max().cloned().unwrap_or(0);
        Chromosome {
            operations: shuffled_operations,
            makespan,
        }
    }

    // One-point crossover (X1)
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

    // Linear order crossover (LOX)
    fn linear_order_crossover(parent1: &Chromosome, parent2: &Chromosome) -> Chromosome {
        let mut rng = thread_rng();
        let length = parent1.operations.len();
    
        // Choose a random subsequence range
        let start = rng.gen_range(0..length);
        let end = rng.gen_range(start..length);
    
        let mut child_operations = vec![None; length];
        let mut parent2_operations = parent2.operations.clone();
    
        // Copy the subsequence from parent1 to the child
        for i in start..end {
            child_operations[i] = Some(parent1.operations[i].clone());
            // Remove the operation from parent2_operations to prevent duplication
            parent2_operations.retain(|op| !(op.job_id == parent1.operations[i].job_id && op.machine_id == parent1.operations[i].machine_id));
        }
    
        // Fill in the remaining positions with operations from parent2
        let mut parent2_index: usize = 0;
        for i in 0..length {
            if child_operations[i].is_none() {
                child_operations[i] = Some(parent2_operations[parent2_index].clone());
                parent2_index += 1;
            }
        }
    
        // Unwrap the Option and calculate the makespan
        let operations = child_operations.into_iter().map(|op| op.unwrap()).collect();
        let makespan = calculate_makespan(&operations);
    
        Chromosome { operations, makespan }
    }
    
    // Order crossover (OX)
    fn order_crossover(parent1: &Chromosome, parent2: &Chromosome) -> Chromosome {
        let mut rng = thread_rng();
        let length = parent1.operations.len();
    
        // Choose a random subsequence range
        let start = rng.gen_range(0..length);
        let end = rng.gen_range(start..length);
    
        let mut child_operations = vec![None; length];
        let mut parent2_operations = parent2.operations.clone();
    
        // Copy the subsequence from parent1 to the child
        for i in start..end {
            child_operations[i] = Some(parent1.operations[i].clone());
            // Remove the operation from parent2_operations to prevent duplication
            parent2_operations.retain(|op| !(op.job_id == parent1.operations[i].job_id && op.machine_id == parent1.operations[i].machine_id));
        }
    
        // Fill in the remaining positions with operations from parent2, starting after the subsequence
        let mut parent2_index = 0;
        for i in end..end+length {
            let idx = i % length;
            if child_operations[idx].is_none() {
                child_operations[idx] = Some(parent2_operations[parent2_index].clone());
                parent2_index += 1;
            }
        }
    
        // Unwrap the Option and calculate the makespan
        let operations = child_operations.into_iter().map(|op| op.unwrap()).collect();
        let makespan = calculate_makespan(&operations);
    
        Chromosome { operations, makespan }
    }

    // Swap mutation
    fn swap_mutation(&mut self) {
        let mut rng = thread_rng();
        let length = self.operations.len();
        let swap_indices = (rng.gen_range(0..length), rng.gen_range(0..length));

        // Swap the operations at the two indices
        let temp = self.operations[swap_indices.0].start_time;
        self.operations[swap_indices.0].start_time = self.operations[swap_indices.1].start_time;
        self.operations[swap_indices.1].start_time = temp;

        // Recalculate the start times after the swap
        for i in swap_indices.0.min(swap_indices.1)+1..swap_indices.0.max(swap_indices.1) {
            if self.operations[i].start_time.is_some() {
                self.operations[i].start_time = Some(self.operations[i-1].start_time.unwrap() + self.operations[i-1].processing_time);
            }
        }
        self.operations.swap(swap_indices.0, swap_indices.1);
        // Recalculate the makespan after mutation
        self.makespan = calculate_makespan(&self.operations);
    }

    fn move_mutation(&mut self) {
        let mut rng = thread_rng();
        let length = self.operations.len();
        
        // Select two different positions at random
        let from_index = rng.gen_range(0..length);
        let to_index = rng.gen_range(0..length);
        
        // Ensure the two indices are not the same
        if from_index != to_index {

            // Perform the move operation
            self.operations[from_index].start_time = self.operations[to_index].start_time;
            let operation = self.operations.remove(from_index);
            self.operations.insert(to_index, operation);

            // Recalculate the start times after the move
            for i in from_index.min(to_index)+1..from_index.max(to_index) {
                if self.operations[i].start_time.is_some() {
                    self.operations[i].start_time = Some(self.operations[i-1].start_time.unwrap() + self.operations[i-1].processing_time);
                }
            }
        }
    
        // Recalculate the makespan after mutation
        self.makespan = calculate_makespan(&self.operations);
    }
    

    // Function to check if a chromosome is redundant
    fn is_redundant(&self, conflict_graph: &ConflictGraph) -> bool {
        // Check for repeated operations
        let mut operation_set = std::collections::HashSet::new();
        for operation in &self.operations {
            if !operation_set.insert((operation.job_id, operation.machine_id)) {
                // If the insertion fails, it means the operation is repeated
                return true;
            }
        }

        // Check for conflicting operations according to the conflict graph
        for &(op1, op2) in &conflict_graph.edges {
            let mut times = self.operations.iter()
                .filter(|&op| op.machine_id == op1 || op.machine_id == op2)
                .map(|op| (op.start_time.unwrap(), op.start_time.unwrap() + op.processing_time))
                .collect::<Vec<_>>();
            // Sort by start time
            times.sort_by_key(|k| k.0);
            println!("times: {:?}", times);
            // Check for overlaps
            for pair in times.windows(2) {
                if let &[(_, end_previous), (start_next, _)] = pair {
                    if start_next < end_previous {
                        // If the start of the next is before the end of the previous, they overlap
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
            // Sort by start time
            times.sort_by_key(|k| k.0);
            println!("times: {:?}", times);
            // Check for overlaps
            for pair in times.windows(2) {
                if let &[(_, end_previous), (start_next, _)] = pair {
                    if start_next < end_previous {
                        // If the start of the next is before the end of the previous, they overlap
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
    // Sort chromosomes by decreasing makespan (fitness)
    fn sort_by_fitness(&mut self) {
        self.chromosomes.sort_by(|a, b| b.makespan.cmp(&a.makespan));
    }

    // Select a parent based on the ranking selection probability
    fn select_parent(&self) -> &Chromosome {
        let mut rng = rand::thread_rng();
        let weights: Vec<_> = (1..=self.chromosomes.len()).map(|rank| 2.0 / (rank as f64)).collect();
        let dist = WeightedIndex::new(&weights).unwrap();
        &self.chromosomes[dist.sample(&mut rng)]
    }

    // Replace a chromosome in the population
    fn replace_chromosome(&mut self, new_chromosome: Chromosome, ga_params: &GAParameters) {
        let mut rng = rand::thread_rng();
        let median_rank = self.chromosomes.len() / 2;
        let replace_index = rng.gen_range(median_rank..self.chromosomes.len());
        self.chromosomes[replace_index] = new_chromosome;
    }
    
    // Selection and replacement procedure
    fn selection_and_replacement(&mut self, conflict_graph: &ConflictGraph, ga_params: &GAParameters) {
        self.sort_by_fitness();

        let parent1 = self.select_parent();
        let parent2 = self.select_parent();

        // Assume we have a crossover function implemented
        let (child1,child2) = Chromosome::one_point_crossover(&parent1, &parent2);
        let (child3,child4) = (Chromosome::linear_order_crossover(&parent1, &parent2),Chromosome::linear_order_crossover(&parent2, &parent1));
        let (child5,child6) = (Chromosome::order_crossover(&parent1, &parent2),Chromosome::order_crossover(&parent2, &parent1));

        let children = vec![child1, child2, child3, child4, child5, child6];

        let mut non_redundant_children = children.into_iter()
            .filter(|child| !child.is_redundant(&conflict_graph))
            .collect::<Vec<_>>();
        // Randomly choose one child for mutation
        let mut rng = rand::thread_rng();

        // Ensure there are enough children to choose from
        if non_redundant_children.len() >= 1 {
            // Get random indices for the elements to mutate
            let idx1 = (0..non_redundant_children.len()).choose(&mut rng).unwrap();
            
            // Mutate the child at idx1 with a certain probability
            if rng.gen::<f64>() < ga_params.mutation_probability {
                let mut child = non_redundant_children[idx1].clone();
                child.swap_mutation();
                if !child.is_redundant(&conflict_graph) {
                    non_redundant_children[idx1] = child;
                }
            }
            
                if non_redundant_children.len() >= 2 {
                    let idx2 = (0..non_redundant_children.len()).choose(&mut rng).unwrap();
                    // Mutate the child at idx2 with a certain probability
                    if rng.gen::<f64>() < ga_params.mutation_probability {
                        let mut child = non_redundant_children[idx2].clone();
                        child.move_mutation();
                        if !child.is_redundant(&conflict_graph) {
                            non_redundant_children[idx2] = child;
                        }
                    }
                }
        }

        // Check if the child is not redundant and replace a chromosome in the population
        for non_redundant_child in non_redundant_children {
            self.replace_chromosome(non_redundant_child, &ga_params);
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

    // The makespan is the maximum end time among all machines
    *machine_end_times.values().max().unwrap_or(&0)
}

fn main() {
    let operations = vec![
        Operation { job_id: 1, machine_id: 1, processing_time: 3, start_time: None },
        Operation { job_id: 1, machine_id: 2, processing_time: 2, start_time: None },
        Operation { job_id: 1, machine_id: 3, processing_time: 2, start_time: None },
        Operation { job_id: 2, machine_id: 1, processing_time: 3, start_time: None },
        Operation { job_id: 2, machine_id: 2, processing_time: 2, start_time: None },
        Operation { job_id: 2, machine_id: 3, processing_time: 1, start_time: None },
        Operation { job_id: 3, machine_id: 1, processing_time: 1, start_time: None },
        Operation { job_id: 3, machine_id: 2, processing_time: 1, start_time: None },
        Operation { job_id: 3, machine_id: 3, processing_time: 2, start_time: None },
    ];

    let conflict_graph = ConflictGraph {
        edges: vec![(1, 2), (1, 3)], // Assuming the edges represent job conflicts
    };

    let mut chromosome1 = Chromosome::initialize(operations.clone(), &conflict_graph);
    let mut chromosome2 = Chromosome::initialize(operations.clone(), &conflict_graph);
    println!("chromosome 1 is redundant: {:?}", chromosome1.is_redundant(&conflict_graph));
    // println!("Chromosome: {:#?}", chromosome1);
    println!("chromosome 2 is redundant: {:?}", chromosome2.is_redundant(&conflict_graph));
    // println!("Chromosome: {:#?}", chromosome2);

    // Calculate the makespan for the chromosome
    // chromosome1.makespan = chromosome1.calculate_makespan();

    let (mut child1,mut child2) = Chromosome::one_point_crossover(&chromosome1, &chromosome2);
    let (mut child3,mut child4) = (Chromosome::linear_order_crossover(&chromosome1, &chromosome2),Chromosome::linear_order_crossover(&chromosome2, &chromosome1));
    let (mut child5,mut child6) = (Chromosome::order_crossover(&chromosome1, &chromosome2),Chromosome::order_crossover(&chromosome2, &chromosome1));
    // println!("Child 1: {:#?}", child1);
    // println!("Child 2: {:#?}", child2);

    println!("Child 1 is redundant: {:?}", child1.is_redundant(&conflict_graph));
    println!("Child 2 is redundant: {:?}", child2.is_redundant(&conflict_graph));
    println!("Child 3 is redundant: {:?}", child3.is_redundant(&conflict_graph));
    println!("Child 4 is redundant: {:?}", child4.is_redundant(&conflict_graph));
    println!("Child 5 is redundant: {:?}", child5.is_redundant(&conflict_graph));
    println!("Child 6 is redundant: {:?}", child6.is_redundant(&conflict_graph));

    let mut child7 = chromosome1.clone();
    child7.swap_mutation();
    println!("Child 7 is redundant: {:?}", child7.is_redundant(&conflict_graph));
    let mut child8 = chromosome1.clone();
    child8.move_mutation();
    println!("Child 8 is redundant: {:?}", child8.is_redundant(&conflict_graph));
}

