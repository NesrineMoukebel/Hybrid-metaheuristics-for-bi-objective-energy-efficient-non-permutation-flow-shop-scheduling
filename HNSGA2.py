import random
import numpy as np
import os
import csv
import time
from multiprocessing import Process

from deap import base, creator, tools


def load_instances(base_dir, num_jobs, num_machines_list, num_instances):
    instances_data = []

    for num_machines in num_machines_list:
        for instance in range(1, num_instances + 1):
            instance_data = {}
            instance_data["jobs"] = None
            instance_data["machines"] = num_machines
            instance_data["processing_times"] = []
            instance_data["energy_prices"] = {"6CW": {}, "CM": {}}
            instance_data["energy_consumption_rates"] = {"PS": [], "PB": []}

            gap_file = f"VFR{num_jobs}_{num_machines}_{instance}_Gap.txt"
            gap_path = os.path.join(base_dir, gap_file)
            if os.path.exists(gap_path):
                with open(gap_path, "r") as file:
                    header = file.readline().strip().split()
                    num_jobs = int(header[0])
                    instance_data["jobs"] = num_jobs
                    processing_times = np.zeros((num_jobs, num_machines), dtype=int)

                    for job_id, line in enumerate(file):
                        values = list(map(int, line.strip().split()))
                        for i in range(0, len(values), 2):
                            machine_id = values[i]
                            processing_time = values[i + 1]
                            processing_times[job_id][machine_id] = processing_time
                    instance_data["processing_times"] = processing_times.tolist()
            else:
                print(f"Missing processing times file: {gap_file}")
                continue

            for tag in ["6CW"]:
                file_name = f"VFR{num_jobs}_{num_machines}_{instance}_Gap__{tag}.txt"
                file_path = os.path.join(base_dir, file_name)
                if os.path.exists(file_path):
                    with open(file_path, "r") as file:
                        lines = file.readlines()
                        time_horizon = int(lines[0].strip())
                        start_vector = list(map(int, lines[1].strip().split()))
                        end_vector = list(map(int, lines[2].strip().split()))
                        price_vector = list(map(float, lines[3].strip().split()))
                        instance_data["energy_prices"][tag] = {
                            "time_horizon": time_horizon,
                            "start": start_vector,
                            "end": end_vector,
                            "prices": price_vector,
                        }
                else:
                    print(f"Missing energy price file: {file_name}")

            for tag in ["PS", "PB"]:
                file_name = f"VFR{num_jobs}_{num_machines}_{instance}_Gap_{tag}.txt"
                file_path = os.path.join(base_dir, file_name)
                if os.path.exists(file_path):
                    with open(file_path, "r") as file:
                        lines = file.readlines()
                        rates = list(map(int, lines[1].strip().split()))
                        instance_data["energy_consumption_rates"][tag] = rates
                else:
                    print(f"Missing energy rates file: {file_name}")

            instances_data.append(instance_data)

    return instances_data


def dominates(fitness_a, fitness_b):
    makespan_a, tec_a = fitness_a
    makespan_b, tec_b = fitness_b
    return (makespan_a <= makespan_b and tec_a <= tec_b) and (makespan_a < makespan_b or tec_a < tec_b)


def calculate_tec(schedule, processing_times, energy_prices, time_periods_start, time_periods_end, energy_rates):
    Tec = 0

    for m, machine_schedule in enumerate(schedule):
        idx = 0
        current_time = 0

        for job, start_time in machine_schedule:
            processing_time = processing_times[job][m]

            if current_time < start_time:
                current_time = start_time

            while processing_time > 0:
                if idx >= len(time_periods_end):
                    energy_used = energy_prices[-1] * processing_time
                    Tec += energy_used
                    current_time += processing_time
                    processing_time = 0
                    break

                if current_time >= time_periods_end[idx]:
                    idx += 1
                    continue

                if current_time < time_periods_start[idx]:
                    current_time = time_periods_start[idx]

                available_time_in_period = time_periods_end[idx] - current_time

                if processing_time <= available_time_in_period:
                    energy_used = energy_prices[idx] * processing_time
                    Tec += energy_used
                    current_time += processing_time
                    processing_time = 0
                else:
                    energy_used = energy_prices[idx] * available_time_in_period
                    Tec += energy_used
                    processing_time -= available_time_in_period
                    current_time = time_periods_end[idx]
                    idx += 1

    return round(Tec, 2)


def calculate_cmax(schedule, machines, jobs, processing_times):
    try:
        job_id = schedule[machines - 1][jobs - 1][0]
        makespan = schedule[machines - 1][jobs - 1][1] + processing_times[job_id][machines - 1]
        return makespan
    except TypeError as e:
        print("Error:", e)
        raise


def evaluate(individual, machines, jobs, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices):
    cmax = calculate_cmax(individual, machines, jobs, processing_times)
    tec = calculate_tec(individual, processing_times, energy_prices, time_periods_start, time_periods_end, energy_consumption_rates)
    return cmax, tec


creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)


def update_start_times_local(schedule, num_machines, jobs, processing_times, machine_idx):
    for machine in range(machine_idx, num_machines):
        for i in range(jobs):
            job, start_time = schedule[machine][i]
            processing_time = processing_times[job][machine]

            if i > 0:
                prev_job, prev_start_time = schedule[machine][i - 1]
                prev_finish_time = prev_start_time + processing_times[prev_job][machine]
                if prev_finish_time > start_time:
                    start_time = prev_finish_time

            if machine > 0:
                prev_machine_finish_time = None
                for j in range(jobs):
                    prev_job, prev_start_time = schedule[machine - 1][j]
                    if prev_job == job:
                        prev_machine_finish_time = prev_start_time + processing_times[prev_job][machine - 1]
                        break

                if prev_machine_finish_time is not None:
                    if prev_machine_finish_time > start_time:
                        start_time = prev_machine_finish_time

            schedule[machine][i] = (job, start_time)


def update_start_times(schedule, num_machines, num_jobs, processing_times):
    for machine in range(num_machines):
        for i in range(num_jobs):
            job, start_time = schedule[machine][i]
            processing_time = processing_times[job][machine]

            if i > 0:
                prev_job, prev_start_time = schedule[machine][i - 1]
                prev_finish_time = prev_start_time + processing_times[prev_job][machine]
                if prev_finish_time > start_time:
                    start_time = prev_finish_time

            if machine > 0:
                prev_machine_finish_time = None
                for j in range(num_jobs):
                    prev_job, prev_start_time = schedule[machine - 1][j]
                    if prev_job == job:
                        prev_machine_finish_time = prev_start_time + processing_times[prev_job][machine - 1]
                        break

                if prev_machine_finish_time is not None:
                    if prev_machine_finish_time > start_time:
                        start_time = prev_machine_finish_time

            schedule[machine][i] = (job, start_time)
            finish_time = start_time + processing_time


def repair_and_update(individual, machines, jobs, processing_times):
    for m in range(machines):
        jobs_seen = set()
        duplicates = []
        missing_jobs = set(range(jobs))

        for idx, (job_id, start_time) in enumerate(individual[m]):
            if job_id in jobs_seen:
                duplicates.append(idx)
            else:
                jobs_seen.add(job_id)
                missing_jobs.discard(job_id)

        repaired_machine_schedule = []
        missing_jobs = list(missing_jobs)

        for idx, (job_id, start_time) in enumerate(individual[m]):
            if idx in duplicates:
                if missing_jobs:
                    missing_job = missing_jobs.pop()
                    repaired_machine_schedule.append((missing_job, 0))
            else:
                repaired_machine_schedule.append((job_id, start_time))

        for missing_job in missing_jobs:
            repaired_machine_schedule.append((missing_job, 0))

        individual[m] = repaired_machine_schedule

    update_start_times(individual, machines, jobs, processing_times)


def is_schedule_feasible(schedule, processing_times):
    job_completion_times = {}

    for machine_idx, machine_schedule in enumerate(schedule):
        previous_end_time = 0
        jobs_seen = set()

        for job, start_time in machine_schedule:
            if job in jobs_seen:
                return False
            jobs_seen.add(job)

            processing_time = processing_times[job][machine_idx]

            if start_time < previous_end_time:
                return False

            if job in job_completion_times and start_time < job_completion_times[job]:
                return False

            end_time = start_time + processing_time
            job_completion_times[job] = end_time
            previous_end_time = end_time

    return True


def create_individual(machines, jobs, processing_times):
    job_sequences = [random.sample(range(0, jobs), jobs) for _ in range(machines)]
    schedule = [[None for _ in range(jobs)] for _ in range(machines)]
    job_completion_times = [0] * jobs

    for machine in range(machines):
        machine_time = 0
        for job_index in job_sequences[machine]:
            if job_completion_times[job_index] > machine_time:
                start_time = job_completion_times[job_index]
            else:
                start_time = machine_time

            schedule[machine][job_index] = (job_index, start_time)

            processing_time = processing_times[job_index][machine]
            machine_time = start_time + processing_time
            job_completion_times[job_index] = machine_time

        schedule[machine] = sorted(schedule[machine], key=lambda x: x[1])

    return schedule


def cxTwoPoint(ind1, ind2, machines, jobs, processing_times):
    cxpoint1, cxpoint2 = sorted(random.sample(range(1, jobs), 2))

    for m in range(machines):
        jobs_ind1 = [job[0] for job in ind1[m]]
        jobs_ind2 = [job[0] for job in ind2[m]]

        jobs_ind1[cxpoint1:cxpoint2], jobs_ind2[cxpoint1:cxpoint2] = (
            jobs_ind2[cxpoint1:cxpoint2],
            jobs_ind1[cxpoint1:cxpoint2],
        )

        ind1[m] = [(job_num, *job[1:]) for job_num, job in zip(jobs_ind1, ind1[m])]
        ind2[m] = [(job_num, *job[1:]) for job_num, job in zip(jobs_ind2, ind2[m])]

    repair_and_update(ind1, machines, jobs, processing_times)
    repair_and_update(ind2, machines, jobs, processing_times)

    return ind1, ind2


def inversion_mutation(schedule, machines, jobs, processing_times):
    new_schedule = [list(machine) for machine in schedule]
    machine = random.randint(0, machines - 1)

    job_indexes = [job[0] for job in new_schedule[machine]]
    start_times = [job[1] for job in new_schedule[machine]]

    p1, p2 = sorted(random.sample(range(jobs), 2))

    inverted_job_indexes = (
        job_indexes[:p1] +
        job_indexes[p1:p2 + 1][::-1] +
        job_indexes[p2 + 1:]
    )

    new_schedule[machine] = [(inverted_job_indexes[i], start_times[i]) for i in range(jobs)]
    update_start_times_local(new_schedule, machines, jobs, processing_times, machine)

    return new_schedule


def total_energy_cost(individual, processing_times, time_periods, energy_prices, energy_consumption_rates, job_index, machine_index, start_time):
    job_number = individual[machine_index][job_index][0]
    processing_time = processing_times[job_number][machine_index]
    end_time = start_time + processing_time

    total_cost = 0
    current_time = start_time

    energy_prices.append(energy_prices[-1])

    for i in range(len(time_periods)):
        period_end = time_periods[i]

        if current_time < period_end:
            period_price = energy_prices[i]
            period_end_time = min(end_time, period_end)
            time_in_period = period_end_time - current_time

            if time_in_period > 0:
                total_cost += time_in_period * period_price

            current_time = period_end_time

        if current_time >= end_time:
            break

    if current_time < end_time:
        remaining_time = end_time - current_time
        total_cost += remaining_time * energy_prices[-1]

    return total_cost


def calculate_tec_mach_vnd(schedule, processing_times, energy_prices, time_periods_start, time_periods_end, energy_rates, machine_index):
    Tec = 0
    idx = 0
    current_time = 0

    for job, start_time in schedule[machine_index]:
        processing_time = processing_times[job][machine_index]

        if current_time < start_time:
            current_time = start_time

        while processing_time > 0:
            if idx >= len(time_periods_end):
                energy_used = energy_prices[-1] * processing_time
                Tec += energy_used
                current_time += processing_time
                processing_time = 0
                break

            if current_time >= time_periods_end[idx]:
                idx += 1
                continue

            if current_time < time_periods_start[idx]:
                current_time = time_periods_start[idx]

            available_time_in_period = time_periods_end[idx] - current_time

            if processing_time <= available_time_in_period:
                energy_used = energy_prices[idx] * processing_time
                Tec += energy_used
                current_time += processing_time
                processing_time = 0
            else:
                energy_used = energy_prices[idx] * available_time_in_period
                Tec += energy_used
                processing_time -= available_time_in_period
                current_time = time_periods_end[idx]
                idx += 1

    return Tec


def machine_sequence_swap_logic(schedule, processing_times, machines, jobs, energy_prices, energy_consumption_rates, time_periods_end, time_periods_start, time_horizon):
    tec_values = []
    solutions = []
    best_solution = None

    original_fitness = evaluate(schedule, machines, jobs, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices)
    original_tec, original_cmax = original_fitness[1], original_fitness[0]

    new_schedule = [list(machine) for machine in schedule]

    for machine in range(machines):
        tec = calculate_tec_mach_vnd(new_schedule, processing_times, energy_prices, time_periods_start, time_periods_end, energy_consumption_rates, machine)
        tec_values.append(tec)

    sorted_tec_indices = sorted(range(machines), key=lambda x: tec_values[x], reverse=True)
    k = 0

    while k < machines:
        machine1 = sorted_tec_indices[k]
        if k < machines - 1:
            machine2 = sorted_tec_indices[k + 1]
        else:
            machine2 = random.randint(0, machines - 1)

        job_indexes1 = [job[0] for job in new_schedule[machine1]]
        job_indexes2 = [job[0] for job in new_schedule[machine2]]

        for i, job in enumerate(new_schedule[machine1]):
            new_schedule[machine1][i] = (job_indexes2[i], job[1])

        for i, job in enumerate(new_schedule[machine2]):
            new_schedule[machine2][i] = (job_indexes1[i], job[1])

        update_start_times(new_schedule, machines, jobs, processing_times)

        new_fitness = evaluate(new_schedule, machines, jobs, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices)
        new_tec, new_cmax = new_fitness[1], new_fitness[0]
        delta_tec = new_tec - original_tec
        delta_cmax = new_cmax - original_cmax

        if delta_tec == 0 and delta_cmax == 0:
            k += 1
            continue

        if new_cmax <= time_horizon:
            if dominates(new_fitness, original_fitness):
                best_solution = [list(machine) for machine in new_schedule]
                return solutions, best_solution
            else:
                if (delta_tec < 0 and delta_cmax > 0) or (delta_cmax < 0 and delta_tec > 0):
                    temp = [list(machine) for machine in new_schedule]
                    solutions.append(temp)

        k += 1

    return solutions, best_solution


def insert_jobs_within_machine2(schedule, processing_times, machines, jobs, energy_prices, energy_consumption_rates, time_periods_end, time_periods_start, time_horizon):
    solutions = []
    best_solution = None

    original_fitness = evaluate(schedule, machines, jobs, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices)
    original_tec, original_cmax = original_fitness[1], original_fitness[0]

    essay = 0
    maxessay = round(2 + (jobs - 10) * (10 - 2) / (800 - 10))

    while essay < maxessay:
        num_jobs_to_insert = random.randint(1, int(jobs / 10))
        essay += 1

        new_schedule = [list(machine) for machine in schedule]
        machine_index = random.randint(0, machines - 1)
        machine = new_schedule[machine_index]

        t = 0
        valid_subsequence_found = False
        while not valid_subsequence_found and t < 10:
            start_index = random.randint(0, jobs - num_jobs_to_insert)
            if start_index + num_jobs_to_insert <= jobs:
                subsequence = machine[start_index:start_index + num_jobs_to_insert]
                valid_subsequence_found = True
            else:
                continue
            t += 1

        if not valid_subsequence_found:
            continue

        new_schedule[machine_index] = machine[:start_index] + machine[start_index + num_jobs_to_insert:]
        insert_position = random.randint(0, len(new_schedule[machine_index]))
        new_schedule[machine_index] = new_schedule[machine_index][:insert_position] + subsequence + new_schedule[machine_index][insert_position:]

        for i in range(num_jobs_to_insert):
            job_number, _ = new_schedule[machine_index][insert_position + i]
            new_schedule[machine_index][insert_position + i] = (job_number, 0)

        update_start_times_local(new_schedule, machines, jobs, processing_times, machine_index)

        new_fitness = evaluate(new_schedule, machines, jobs, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices)
        new_tec, new_cmax = new_fitness[1], new_fitness[0]
        delta_tec = new_tec - original_tec
        delta_cmax = new_cmax - original_cmax

        if delta_tec == 0 and delta_cmax == 0:
            continue

        if new_cmax <= time_horizon:
            if dominates(new_fitness, original_fitness):
                best_solution = [list(machine) for machine in new_schedule]
                return solutions, best_solution
            else:
                if (delta_tec < 0 and delta_cmax > 0) or (delta_cmax < 0 and delta_tec > 0):
                    temp = [list(machine) for machine in new_schedule]
                    solutions.append(temp)

    return solutions, best_solution


def job_swap_on_one_machine(individual, processing_times, machines, jobs, energy_prices, energy_consumption_rates, time_periods_end, time_periods_start, time_horizon):
    solutions = []
    best_solution = None

    original_fitness = evaluate(individual, machines, jobs, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices)
    original_tec, original_cmax = original_fitness[1], original_fitness[0]

    essay = 0
    maxessay = round(2 + (jobs - 10) * (10 - 2) / (800 - 10))

    while essay < maxessay:
        num_jobs_in_subsequence = random.randint(1, int(jobs / 10))
        essay += 1

        machine_index = random.randint(0, machines - 1)
        new_schedule = [list(machine) for machine in individual]
        machine = new_schedule[machine_index]

        valid_subsequences_found = False
        attempts = 0

        while not valid_subsequences_found and attempts < 10:
            start_index1 = random.randint(0, jobs - num_jobs_in_subsequence)

            if start_index1 + num_jobs_in_subsequence <= jobs:
                subsequence1 = machine[start_index1:start_index1 + num_jobs_in_subsequence]
            else:
                attempts += 1
                continue

            start_index2 = random.randint(0, jobs - num_jobs_in_subsequence)

            if abs(start_index1 - start_index2) < num_jobs_in_subsequence:
                attempts += 1
                continue

            if start_index2 + num_jobs_in_subsequence <= jobs:
                subsequence2 = machine[start_index2:start_index2 + num_jobs_in_subsequence]
                valid_subsequences_found = True
            else:
                attempts += 1
                continue

        if not valid_subsequences_found:
            continue

        if start_index1 == start_index2:
            continue

        machine[start_index1:start_index1 + num_jobs_in_subsequence] = subsequence2
        machine[start_index2:start_index2 + num_jobs_in_subsequence] = subsequence1

        for i in range(num_jobs_in_subsequence):
            job_number, _ = machine[start_index1 + i]
            machine[start_index1 + i] = (job_number, 0)
            job_number, _ = machine[start_index2 + i]
            machine[start_index2 + i] = (job_number, 0)

        update_start_times_local(new_schedule, machines, jobs, processing_times, machine_index)

        new_fitness = evaluate(new_schedule, machines, jobs, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices)
        new_tec, new_cmax = new_fitness[1], new_fitness[0]
        delta_tec = new_tec - original_tec
        delta_cmax = new_cmax - original_cmax

        if delta_tec == 0 and delta_cmax == 0:
            continue

        if new_cmax <= time_horizon:
            if dominates(new_fitness, original_fitness):
                best_solution = [list(machine) for machine in new_schedule]
                return solutions, best_solution
            else:
                if (delta_tec < 0 and delta_cmax > 0) or (delta_cmax < 0 and delta_tec > 0):
                    temp = [list(machine) for machine in new_schedule]
                    solutions.append(temp)

    return solutions, best_solution


def job_swap_on_one_machine_logic(individual, processing_times, machines, jobs, energy_prices, energy_consumption_rates, time_periods_end, time_periods_start, time_horizon):
    tec_values = []
    solutions = []
    best_solution = None

    original_fitness = evaluate(individual, machines, jobs, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices)
    original_tec, original_cmax = original_fitness[1], original_fitness[0]

    for machine in range(machines):
        tec = calculate_tec_mach_vnd(individual, processing_times, energy_prices, time_periods_start, time_periods_end, energy_consumption_rates, machine)
        tec_values.append(tec)

    sorted_tec_indices = sorted(range(machines), key=lambda x: tec_values[x], reverse=True)
    num_jobs = round(2 + (jobs - 10) * (10 - 2) / (800 - 10))

    new_individual = [list(machine) for machine in individual]
    machine_index = sorted_tec_indices[0]
    machine = new_individual[machine_index]

    job_costs = [(job_index, total_energy_cost(new_individual, processing_times, time_periods_end, energy_prices, energy_consumption_rates, job_index, machine_index, start_time))
                 for job_index, (job_number, start_time) in enumerate(machine)]

    job_costs.sort(key=lambda x: x[1], reverse=True)
    num_swaps = min(num_jobs, len(job_costs) - 1)

    for i in range(num_swaps):
        job1_index, _ = job_costs[i]
        job2_index, _ = job_costs[i + 1]

        idx1 = next(idx for idx, (job, _) in enumerate(machine) if job == job1_index)
        idx2 = next(idx for idx, (job, _) in enumerate(machine) if job == job2_index)

        machine[idx1], machine[idx2] = machine[idx2], machine[idx1]
        machine[idx1] = (machine[idx1][0], 0)
        machine[idx2] = (machine[idx2][0], 0)

        update_start_times_local(new_individual, machines, jobs, processing_times, machine_index)

        new_fitness = evaluate(new_individual, machines, jobs, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices)
        new_tec, new_cmax = new_fitness[1], new_fitness[0]
        delta_tec, delta_cmax = new_tec - original_tec, new_cmax - original_cmax

        if delta_tec == 0 and delta_cmax == 0:
            continue

        if new_cmax <= time_horizon:
            if dominates(new_fitness, original_fitness):
                best_solution = [list(machine) for machine in new_individual]
                return solutions, best_solution
            else:
                if (delta_tec < 0 and delta_cmax > 0) or (delta_cmax < 0 and delta_tec > 0):
                    temp = [list(machine) for machine in new_individual]
                    solutions.append(temp)

    return solutions, best_solution


def insert_jobs_within_machine_logic(schedule, processing_times, machines, jobs, energy_prices, energy_consumption_rates, time_periods_end, time_periods_start, time_horizon):
    sorted_periods = sorted(range(len(time_periods_start)), key=lambda i: energy_prices[i])

    tec_values = []
    solutions = []
    best_solution = None

    original_fitness = evaluate(schedule, machines, jobs, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices)
    original_tec, original_cmax = original_fitness[1], original_fitness[0]

    for machine in range(machines):
        tec = calculate_tec_mach_vnd(schedule, processing_times, energy_prices, time_periods_start, time_periods_end, energy_consumption_rates, machine)
        tec_values.append(tec)

    sorted_tec_indices = sorted(range(machines), key=lambda x: tec_values[x], reverse=True)
    num_jobs_to_insert = round(2 + (jobs - 10) * (10 - 2) / (800 - 10))

    new_schedule = [list(machine) for machine in schedule]
    machine_index = sorted_tec_indices[0]
    selected_machine_schedule = new_schedule[machine_index][:]

    sorted_jobs = [
        (job[0], job[1], total_energy_cost(
            new_schedule, processing_times, time_periods_end,
            energy_prices, energy_consumption_rates,
            job[0], machine_index, job[1]
        ))
        for job in selected_machine_schedule
    ]
    sorted_jobs.sort(key=lambda x: x[2], reverse=True)
    sorted_jobs = [(job[0], job[1]) for job in sorted_jobs]
    jobs_to_consider = sorted_jobs[:min(num_jobs_to_insert, len(sorted_jobs))]

    for job_number, current_start_time in jobs_to_consider:
        new_schedule = [list(machine) for machine in schedule]
        period_index = sorted_periods[0]
        min_energy_period_start = time_periods_start[period_index]
        new_start_time = min_energy_period_start

        if new_start_time == current_start_time:
            continue

        new_schedule[machine_index] = [job for job in selected_machine_schedule if job[0] != job_number]

        insert_position = next(
            (i for i, (jn, st) in enumerate(new_schedule[machine_index]) if st >= new_start_time),
            jobs
        )

        jobs_to_move = [(job_number, new_start_time)]
        new_schedule[machine_index] = (
            new_schedule[machine_index][:insert_position] + jobs_to_move + new_schedule[machine_index][insert_position:]
        )

        update_start_times_local(new_schedule, machines, jobs, processing_times, machine_index)

        new_fitness = evaluate(new_schedule, machines, jobs, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices)
        new_tec, new_cmax = new_fitness[1], new_fitness[0]
        delta_tec = new_tec - original_tec
        delta_cmax = new_cmax - original_cmax

        if delta_tec == 0 and delta_cmax == 0:
            continue

        if new_cmax <= time_horizon:
            if dominates(new_fitness, original_fitness):
                best_solution = [list(machine) for machine in new_schedule]
                return solutions, best_solution
            else:
                if (delta_tec < 0 and delta_cmax > 0) or (delta_cmax < 0 and delta_tec > 0):
                    temp = [list(machine) for machine in new_schedule]
                    solutions.append(temp)

    return solutions, best_solution


def VND(initial_schedule, processing_times, machines, jobs, energy_consumption_rates, time_periods_end, energy_prices, time_periods_start, time_horizon, operator_counts, chunk_operator_counts, current_chunk):
    local_neighborhoods = [
        insert_jobs_within_machine2,
        insert_jobs_within_machine_logic,
        job_swap_on_one_machine,
        job_swap_on_one_machine_logic,
        machine_sequence_swap_logic
    ]

    best_schedule = [list(machine) for machine in initial_schedule]
    initial_fitness = evaluate(initial_schedule, machines, jobs, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices)
    best_fitness = initial_fitness
    all_solutions = []

    for neighborhood_index, neighborhood in enumerate(local_neighborhoods):
        solutions, dominating_solution = neighborhood(
            best_schedule, processing_times, machines, jobs, energy_prices, energy_consumption_rates, time_periods_end, time_periods_start, time_horizon
        )
        operator_counts[neighborhood.__name__] += 1

        if dominating_solution:
            best_schedule = [list(machine) for machine in dominating_solution]
            best_fitness = evaluate(best_schedule, machines, jobs, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices)
            chunk_operator_counts[current_chunk][neighborhood.__name__] += 1
            return best_schedule, best_fitness, operator_counts, all_solutions
        else:
            if solutions:
                all_solutions.extend(solutions)

    if len(all_solutions) == 0:
        return initial_schedule, initial_fitness, operator_counts, all_solutions

    selected_solution = random.choice(all_solutions)
    selected_solution_fitness = evaluate(selected_solution, machines, jobs, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices)

    best_schedule = [list(machine) for machine in selected_solution]
    best_fitness = selected_solution_fitness

    return best_schedule, best_fitness, operator_counts, None


def left_shift_schedule(schedule, num_machines, num_jobs, processing_times, period_ends, job_info, chosen_periods):
    chosen_periods_sorted = sorted(chosen_periods, key=lambda x: x[0])
    horizon_end = max(period_ends)

    for m in range(num_machines):
        for j in range(num_jobs):
            job_id = schedule[m][j][0]
            pt = processing_times[job_id][m]

            max_shift = 0
            if j > 0:
                if job_info[schedule[m][j - 1][0]][m]['end'] > max_shift:
                    max_shift = job_info[schedule[m][j - 1][0]][m]['end']

            if m > 0:
                if job_info[job_id][m - 1]['end'] > max_shift:
                    max_shift = job_info[job_id][m - 1]['end']

            best_period = None
            for p_idx, p in enumerate(chosen_periods_sorted):
                if p[1] >= max_shift and p[1] - max_shift >= pt:
                    best_period = p
                    break
                if p_idx < len(chosen_periods_sorted) - 1:
                    next_p = chosen_periods_sorted[p_idx + 1]
                    if p[1] + 1 == next_p[0] and p[2] < next_p[2]:
                        if p[1] >= max_shift and next_p[1] - max_shift >= pt:
                            best_period = p
                            break

            if best_period is None:
                break

            if max_shift > best_period[0]:
                new_start = max_shift
            else:
                new_start = best_period[0]

            new_end = new_start + pt

            if new_end > horizon_end:
                new_start = horizon_end - pt
                new_end = horizon_end

            if j > 0:
                if job_info[schedule[m][j - 1][0]][m]['end'] > new_start:
                    new_start = job_info[schedule[m][j - 1][0]][m]['end']
            if m > 0:
                if job_info[job_id][m - 1]['end'] > new_start:
                    new_start = job_info[job_id][m - 1]['end']

            new_end = new_start + pt
            if new_end > horizon_end:
                return job_info

            job_info[job_id][m]['start'] = new_start
            job_info[job_id][m]['end'] = new_end

    return job_info


def right_shift_schedule(schedule, num_machines, num_jobs, processing_times, period_ends, job_info, chosen_periods):
    chosen_periods_sorted = sorted(chosen_periods, key=lambda x: x[0])
    horizon_end = period_ends[-1]

    def get_period_for_time(time):
        return next((p for p in chosen_periods_sorted if p[0] <= time <= p[1]), None)

    def has_more_expensive_period_before(period):
        period_idx = chosen_periods_sorted.index(period)
        return any(p[2] > period[2] for p in chosen_periods_sorted[:period_idx])

    def get_latest_allowed_end(machine, job_idx, job_id):
        if job_idx < num_jobs - 1:
            next_job_id = schedule[machine][job_idx + 1][0]
            allowed_end = job_info[next_job_id][machine]['start']
        else:
            allowed_end = horizon_end

        if machine < num_machines - 1:
            next_machine_start = job_info[job_id][machine + 1]['start']
            if next_machine_start < allowed_end:
                allowed_end = next_machine_start

        return allowed_end

    for m in range(num_machines - 1, -1, -1):
        for j in reversed(range(num_jobs)):
            job_id = schedule[m][j][0]
            pt = processing_times[job_id][m]
            current_start = job_info[job_id][m]['start']
            current_end = job_info[job_id][m]['end']

            current_period = get_period_for_time(current_start)
            if not current_period:
                continue

            allowed_end = get_latest_allowed_end(m, j, job_id)
            if allowed_end <= current_end:
                continue

            best_start = current_start
            best_end = current_end

            current_period_idx = chosen_periods_sorted.index(current_period)
            has_expensive_period_before = has_more_expensive_period_before(current_period)

            for period in chosen_periods_sorted[current_period_idx + 1:]:
                if period[2] >= current_period[2]:
                    continue

                if period[0] > (allowed_end - pt):
                    start_in_period = period[0]
                else:
                    start_in_period = allowed_end - pt

                end_in_period = start_in_period + pt

                if period[0] <= start_in_period and end_in_period <= period[1]:
                    best_start = start_in_period
                    best_end = end_in_period
                    break
                else:
                    period_idx = chosen_periods_sorted.index(period)
                    if period_idx < len(chosen_periods_sorted) - 1:
                        next_period = chosen_periods_sorted[period_idx + 1]
                        if period[1] + 1 == next_period[0]:
                            time_in_first_period = period[1] - start_in_period
                            time_in_second_period = pt - time_in_first_period

                            if time_in_second_period <= next_period[1] - next_period[0]:
                                best_start = start_in_period
                                best_end = start_in_period + pt
                                break

            if has_expensive_period_before and best_start == current_start and best_end == current_end or has_expensive_period_before:
                if (current_period[1] - pt) > (allowed_end - pt):
                    latest_start = current_period[1] - pt
                else:
                    latest_start = allowed_end - pt
                if latest_start >= current_period[0]:
                    best_start = latest_start
                    best_end = latest_start + pt

            elif not has_expensive_period_before and best_start == current_start and best_end == current_end:
                continue

            if best_start >= 0 and best_end <= horizon_end:
                if (j < len(schedule[m]) - 1 and best_end > job_info[schedule[m][j + 1][0]][m]['start']) or \
                   (m < num_machines - 1 and best_end > job_info[job_id][m + 1]['start']):

                    if j < len(schedule[m]) - 1:
                        best_end = job_info[schedule[m][j + 1][0]][m]['start']

                    if m < num_machines - 1:
                        best_end = min(best_end, job_info[job_id][m + 1]['start'])

                best_period = get_period_for_time(best_end)
                if best_period and (best_period[2] > current_period[2]):
                    continue
                job_info[job_id][m]['end'] = best_end
                job_info[job_id][m]['start'] = best_end - pt
                best_start = best_end - pt
                schedule[m][j] = (job_id, best_start)

    return job_info


def tec_reducer(schedule, num_machines, num_jobs, processing_times, period_starts, period_ends, prices):
    job_info = [{} for _ in range(num_jobs)]

    cmax = calculate_cmax(schedule, num_machines, num_jobs, processing_times)

    periods = list(zip(period_starts, period_ends, prices))
    all_periods = sorted(
        periods,
        key=lambda x: (
            x[2],
            not (
                (periods.index(x) > 0 and periods[periods.index(x) - 1][2] < x[2]) or
                (periods.index(x) < len(periods) - 1 and periods[periods.index(x) + 1][2] < x[2])
            ),
            x[0]
        )
    )

    chosen_periods = []
    remaining = cmax
    for (ps, pe, pr) in all_periods:
        if remaining <= 0:
            break
        duration = pe - ps
        chosen_periods.append((ps, pe, pr))
        remaining -= duration

    while remaining > 0:
        for (ps, pe, pr) in all_periods:
            duration = pe - ps
            chosen_periods.append((ps, pe, pr))
            remaining -= duration
            if remaining <= 0:
                break

    chosen_periods.sort(key=lambda x: x[1], reverse=True)

    def are_periods_contiguous(period1, period2):
        return period1[1] + 1 == period2[0]

    last_machine = num_machines - 1
    last_job_idx = num_jobs - 1
    current_period_idx = 0
    current_period = chosen_periods[current_period_idx]

    last_job_id = schedule[last_machine][last_job_idx][0]
    last_job_pt = processing_times[last_job_id][last_machine]

    job_info[last_job_id][last_machine] = {
        'start': current_period[1] - last_job_pt,
        'end': current_period[1]
    }

    for m in reversed(range(num_machines)):
        start_j = num_jobs - 1 if m != last_machine else last_job_idx - 1

        for j in reversed(range(start_j + 1)):
            current_period_idx = 0
            current_period = chosen_periods[current_period_idx]
            current_job_id = schedule[m][j][0]
            pt = processing_times[current_job_id][m]

            latest_possible_end = float('inf')

            if j < num_jobs - 1:
                next_job_id = schedule[m][j + 1][0]
                next_job_start = job_info[next_job_id][m]['start']
                latest_possible_end = next_job_start

            if m < num_machines - 1:
                next_machine_jobs = [job_id for job_id, _ in schedule[m + 1]]
                if current_job_id in next_machine_jobs:
                    next_machine_start = job_info[current_job_id][m + 1]['start']
                    if next_machine_start < latest_possible_end:
                        latest_possible_end = next_machine_start

            if latest_possible_end == float('inf'):
                latest_possible_end = current_period[1]

            job_end = latest_possible_end
            job_start = job_end - pt

            placed = False
            while not placed:
                if job_end <= current_period[1] and job_start >= current_period[0]:
                    placed = True
                elif (current_period_idx < len(chosen_periods) - 1 and
                      are_periods_contiguous(chosen_periods[current_period_idx + 1], current_period) and
                      job_end <= current_period[1] and
                      job_start >= chosen_periods[current_period_idx + 1][0]):
                    if (current_period in chosen_periods and
                            chosen_periods[current_period_idx + 1] in chosen_periods):
                        placed = True
                else:
                    current_period_idx += 1
                    if current_period_idx >= len(chosen_periods):
                        break
                    current_period = chosen_periods[current_period_idx]
                    job_end = min(current_period[1], latest_possible_end)
                    job_start = job_end - pt

            job_info[current_job_id][m] = {
                'start': job_start,
                'end': job_end
            }

    job_info = left_shift_schedule(schedule, num_machines, num_jobs, processing_times, period_ends, job_info, chosen_periods)
    if random.random() > 0.5:
        job_info = right_shift_schedule(schedule, num_machines, num_jobs, processing_times, period_ends, job_info, chosen_periods)

    final_schedule = []
    for m in range(num_machines):
        machine_schedule = []
        for job_id, _ in schedule[m]:
            start_time = job_info[job_id][m]['start']
            machine_schedule.append((job_id, start_time))
        final_schedule.append(machine_schedule)

    return final_schedule


def nfs_heuristic(machines, jobs, processing_times, p):
    def calculate_makespan(schedule, processing_times, machines, global_job_completion):
        max_completion = 0
        for machine in range(machines):
            completion_time = 0
            for idx, job in enumerate(schedule):
                if machine == 0:
                    completion_time += processing_times[job][machine]
                else:
                    prev_machine_time = global_job_completion[job][machine - 1]
                    if completion_time > prev_machine_time:
                        completion_time = completion_time + processing_times[job][machine]
                    else:
                        completion_time = prev_machine_time + processing_times[job][machine]

                global_job_completion[job][machine] = completion_time

            if max_completion < completion_time:
                max_completion = completion_time

        return max_completion

    def insert_at_best_position(schedule, job, processing_times, global_job_completion, machines):
        best_makespan = float('inf')
        best_position = 0

        if jobs > 60:
            num_pos = len(schedule) // 10
        else:
            num_pos = len(schedule)

        for pos in range(num_pos + 1):
            temp_schedule = schedule[:pos] + [job] + schedule[pos:]
            temp_global_completion = {k: v[:] for k, v in global_job_completion.items()}
            makespan = calculate_makespan(temp_schedule, processing_times, machines, temp_global_completion)

            if makespan < best_makespan:
                best_makespan = makespan
                best_position = pos

        schedule.insert(best_position, job)
        calculate_makespan(schedule, processing_times, machines, global_job_completion)
        return schedule

    partial_schedules = [[] for _ in range(machines)]
    global_job_completion = {job: [0] * machines for job in range(jobs)}
    job_completion_times = [0] * jobs

    job_order = list(range(jobs))
    random.shuffle(job_order)

    pn = int(np.floor(p * jobs))
    current_schedule = []

    if pn >= 1:
        selected_jobs = job_order[:pn]
        total_processing_times = [sum(processing_times[job]) for job in selected_jobs]
        first_job = selected_jobs[np.argmax(total_processing_times)]
        current_schedule.append(first_job)

        remaining_jobs = [job for job in selected_jobs if job != first_job]
        for j in remaining_jobs:
            current_schedule = insert_at_best_position(current_schedule, j, processing_times, global_job_completion, machines)

        for i in range(machines):
            partial_schedules[i] = current_schedule

    remaining_jobs = [job for job in range(jobs) if job not in current_schedule]

    for idx, job in enumerate(remaining_jobs):
        best_makespan = float("inf")
        best_schedules = None

        if jobs > 60:
            num_possible_pos = len(current_schedule) // 10
        else:
            num_possible_pos = len(current_schedule)

        for k in range(num_possible_pos + idx + 1):
            temp_schedules = [schedule[:] for schedule in partial_schedules]
            num_machines_to_insert = random.randint(0, machines // 2)

            for i in range(num_machines_to_insert):
                temp_schedules[i].insert(k, job)

            straight_schedules = [schedule[:] for schedule in temp_schedules]
            for m in range(num_machines_to_insert, machines):
                straight_schedules[m].insert(k, job)
            straight_makespan = calculate_makespan(straight_schedules[0], processing_times, machines, global_job_completion)

            if straight_makespan < best_makespan:
                best_makespan = straight_makespan
                best_schedules = straight_schedules

            for pos in range(k - 1, k):
                anticipation_schedules = [schedule[:] for schedule in temp_schedules]
                for m in range(num_machines_to_insert, machines):
                    anticipation_schedules[m].insert(pos, job)
                anticipation_makespan = calculate_makespan(anticipation_schedules[num_machines_to_insert], processing_times, machines, global_job_completion)

                if anticipation_makespan < best_makespan:
                    best_makespan = anticipation_makespan
                    best_schedules = anticipation_schedules

            for pos in range(k + 1, k + 2):
                delay_schedules = [schedule[:] for schedule in temp_schedules]
                for m in range(num_machines_to_insert, machines):
                    delay_schedules[m].insert(pos, job)
                delay_makespan = calculate_makespan(delay_schedules[num_machines_to_insert], processing_times, machines, global_job_completion)

                if delay_makespan < best_makespan:
                    best_makespan = delay_makespan
                    best_schedules = delay_schedules

        if best_schedules is not None:
            partial_schedules = best_schedules[:]

    final_schedule = [[] for _ in range(machines)]

    for i in range(machines):
        current_time = 0
        job_start_times = []

        for job in partial_schedules[i]:
            if current_time > job_completion_times[job]:
                start_time = current_time
            else:
                start_time = job_completion_times[job]

            job_start_times.append((job, start_time))
            job_completion_times[job] = start_time + processing_times[job][i]
            current_time = job_completion_times[job]

        job_start_times.sort(key=lambda x: x[1])
        for job, start_time in job_start_times:
            final_schedule[i].append((job, start_time))

    return final_schedule


def init_population(processing_times, energy_prices, energy_consumption_rates, size_pop, time_periods, time_periods_start, time_periods_end, jobs, machines):
    population = []

    if jobs > 60:
        nfs_size = int(0.03 * size_pop)
    else:
        nfs_size = int(0.2 * size_pop)

    random_size = size_pop - nfs_size

    start_value = 0.1
    increment = round((1 - start_value) / (nfs_size - 1), 2)
    p_values = [round(start_value + i * increment, 2) for i in range(nfs_size)]

    idx = 0
    for p in p_values:
        while True:
            schedule = nfs_heuristic(machines, jobs, processing_times, p)
            if not is_schedule_feasible(schedule, processing_times):
                continue

            schedule_copy = [list(machine) for machine in schedule]
            population.append(creator.Individual(schedule_copy))
            print(f"NFS individual {idx}: Cmax = {calculate_cmax(schedule, machines, jobs, processing_times)}")
            break
        idx += 1

    for i in range(random_size):
        new_individual = create_individual(machines, jobs, processing_times)
        schedule_copy = [list(machine) for machine in new_individual]
        population.append(creator.Individual(schedule_copy))

    print("Population initialized.")
    return population


def filter_duplicates(pareto_front):
    unique_fitness = set()
    filtered_front = []

    for ind in pareto_front:
        fitness = ind.fitness.values
        if fitness not in unique_fitness:
            unique_fitness.add(fitness)
            filtered_front.append(ind)

    return filtered_front


seen_schedules = []
seen_fitness = []


def process_instance(instance, energy_config, consumption_config):
    machines = instance["machines"]
    jobs = instance["jobs"]
    processing_times = instance["processing_times"]
    energy_prices_data = instance["energy_prices"][energy_config]
    time_periods = energy_prices_data["end"]
    time_periods_start = energy_prices_data["start"]
    time_periods_end = energy_prices_data["end"]
    energy_prices = energy_prices_data["prices"]
    energy_consumption_rates = instance["energy_consumption_rates"][consumption_config]

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: create_individual(machines, jobs, processing_times))
    toolbox.register("population", init_population,
                     processing_times=processing_times,
                     energy_prices=energy_prices,
                     energy_consumption_rates=energy_consumption_rates,
                     size_pop=100,
                     time_periods=time_periods,
                     time_periods_start=time_periods_start,
                     time_periods_end=time_periods_end,
                     jobs=jobs,
                     machines=machines)

    toolbox.register("mate", lambda ind1, ind2: cxTwoPoint(ind1, ind2, machines, jobs, processing_times))
    toolbox.register("mutate", lambda ind: inversion_mutation(ind, machines, jobs, processing_times))
    toolbox.register("mutate_tec", lambda ind: tec_reducer(ind, machines, jobs, processing_times, time_periods_start, time_periods_end, energy_prices))
    toolbox.register("evaluate", lambda ind: evaluate(ind, machines, jobs, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices))

    population = toolbox.population()
    for ind in population:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind)

    explored_sol_unfiltered = population[:]

    generations = 100
    Pc = 0.6
    Pm = 0.2
    size_pop = 100
    cpt = 0

    global_pareto_front = tools.sortNondominated(population, size_pop, first_front_only=False)[0]
    cmax_values_init = [ind.fitness.values[0] for ind in global_pareto_front]
    tec_values_init = [ind.fitness.values[1] for ind in global_pareto_front]

    no_improvement_count = 0
    max_no_improvement = 20
    gen = 0
    num_chunks = 10
    operator_counts = {name: 0 for name in ['insert_jobs_within_machine2', 'insert_jobs_within_machine_logic',
                                             'job_swap_on_one_machine', 'job_swap_on_one_machine_logic',
                                             'machine_sequence_swap_logic']}
    chunk_operator_counts = [{op: 0 for op in operator_counts} for _ in range(num_chunks)]

    max_time = jobs * machines / 3.0
    init_time = time.time()

    while (gen < generations) and ((time.time() - init_time) < max_time):
        print(f"Generation: {gen}")

        selected_nsga = tools.selNSGA2(population, size_pop)
        half_selected = selected_nsga[:size_pop // 1]
        offspring = list(map(toolbox.clone, half_selected))
        size_offspring = 100

        for i in range(0, size_offspring - 2, 2):
            parent1 = offspring[i]
            parent2 = offspring[i + 1]
            if random.random() < Pc:
                child1_raw, child2_raw = toolbox.mate(parent1, parent2)
                child1 = creator.Individual([list(machine) for machine in child1_raw])
                child2 = creator.Individual([list(machine) for machine in child2_raw])

                del child1.fitness.values
                del child2.fitness.values

                offspring[i] = child1
                offspring[i + 1] = child2

        for i, mutant in enumerate(offspring):
            if random.random() < Pm:
                mutant_raw = toolbox.mutate(mutant)
                mutant = creator.Individual([list(machine) for machine in mutant_raw])
                cmax = calculate_cmax(mutant, machines, jobs, processing_times)
                if is_schedule_feasible(mutant, processing_times) and cmax <= time_periods_end[-1]:
                    offspring[i] = mutant
                del mutant.fitness.values

        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        if gen > 60 and jobs > 500:
            sorted_by_tec = sorted(offspring, key=lambda ind: ind.fitness.values[1], reverse=True)[:2]
        else:
            sorted_by_tec = sorted(offspring, key=lambda ind: ind.fitness.values[1], reverse=True)[:7]

        for mutant in sorted_by_tec:
            old_cmax, old_tec = toolbox.evaluate(mutant)
            chunk_index = min(int((cpt / 700) * num_chunks), num_chunks - 1)
            current_chunk = chunk_index
            best_schedule, _, operator_counts, _ = VND(
                mutant, processing_times, machines, jobs, energy_consumption_rates,
                time_periods_end, energy_prices, time_periods_start, time_periods_end[-1],
                operator_counts, chunk_operator_counts, current_chunk
            )
            cpt += 1

            if random.random() > 0.4:
                mutated_schedule = toolbox.mutate_tec(best_schedule)
            else:
                mutated_schedule = best_schedule

            cmax = calculate_cmax(mutated_schedule, machines, jobs, processing_times)

            if is_schedule_feasible(mutated_schedule, processing_times) and cmax <= time_periods_end[-1]:
                offspring.append(creator.Individual(mutated_schedule[:]))
            elif not is_schedule_feasible(mutated_schedule, processing_times):
                if is_schedule_feasible(best_schedule, processing_times):
                    offspring.append(creator.Individual(best_schedule[:]))

        combined_population = population[:]
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)
            fitness = ind.fitness.values
            if is_schedule_feasible(ind, processing_times) and fitness[0] <= time_periods_end[-1]:
                combined_population.append(ind)
                seen_schedules.append(tuple(tuple(machine) for machine in ind))
                seen_fitness.append(fitness)

        cutoff = int(1.0 * len(population))
        selected_top = tools.selNSGA2(combined_population, cutoff)
        population[:cutoff] = selected_top
        explored_sol_unfiltered.extend(population)

        print(f"Population size: {len(population)}")
        current_non_dominated = tools.sortNondominated(population, len(population), first_front_only=False)[0]

        global_pareto_front += current_non_dominated
        explored_sol_unfiltered = [ind for ind in explored_sol_unfiltered if ind not in current_non_dominated]

        improvement_found = False
        for new_sol in current_non_dominated:
            for existing_sol in global_pareto_front:
                if dominates(new_sol.fitness.values, existing_sol.fitness.values):
                    improvement_found = True
                    break
            if improvement_found:
                break

        if improvement_found:
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= max_no_improvement:
            print(f"Early stopping at generation {gen}: no improvement for {max_no_improvement} generations.")
            break

        gen += 1

    print(f"Optimization complete. VND calls: {cpt}")
    print(f"Operator counts: {operator_counts}")

    for ind in global_pareto_front:
        fitness = toolbox.evaluate(ind)

    global_pareto_front = tools.sortNondominated(global_pareto_front, len(global_pareto_front), first_front_only=False)[0]

    explored_sol_unfiltered = [ind for ind in explored_sol_unfiltered if ind not in global_pareto_front]

    filtered_front = []
    for ind in global_pareto_front:
        if is_schedule_feasible(ind, processing_times) and ind.fitness.values[0] <= time_periods_end[-1]:
            filtered_front.append(ind)

    explored_sol = []
    for sol in explored_sol_unfiltered:
        ind.fitness.values = toolbox.evaluate(sol)
        if is_schedule_feasible(sol, processing_times):
            explored_sol.append(sol)

    explored_sol = filter_duplicates(explored_sol)
    filtered_front = filter_duplicates(filtered_front)

    cmax_values = [ind.fitness.values[0] for ind in filtered_front]
    tec_values = [ind.fitness.values[1] for ind in filtered_front]

    cmax_values_explored = [sol.fitness.values[0] for sol in explored_sol]
    tec_values_explored = [sol.fitness.values[1] for sol in explored_sol]
    sorted_explored = sorted(zip(cmax_values_explored, tec_values_explored), key=lambda x: x[0])
    sorted_cmax_explored, sorted_tec_explored = zip(*sorted_explored)

    sorted_front = sorted(zip(cmax_values, tec_values), key=lambda x: x[0])
    sorted_cmax, sorted_tec = zip(*sorted_front)

    sorted_front_init = sorted(zip(cmax_values_init, tec_values_init), key=lambda x: x[0])
    sorted_cmax_init, sorted_tec_init = zip(*sorted_front_init)

    return sorted_cmax, sorted_tec, sorted_cmax_init, sorted_tec_init, sorted_cmax_explored, sorted_tec_explored


def save_pareto_front(cmax_values, tec_values, cmax_explored, tec_explored, num_machines, num_jobs, config_type, instance_idx, exec_time, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"M{num_machines}_J{num_jobs}_config_6CW_{instance_idx + 1}.csv")

    explored_solutions = list(zip(cmax_explored, tec_explored))
    pareto_solutions = set(zip(cmax_values, tec_values))

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Makespan", "TEC", "Pareto", "Execution Time"])

        for cmax, tec in zip(cmax_values, tec_values):
            writer.writerow([float(cmax), float(tec), "true", float(exec_time)])

        for cmax, tec in explored_solutions:
            writer.writerow([float(cmax), float(tec), "false", float(exec_time)])

    print(f"Results saved to {filename}.")


def process_instance_parallel(instance, instance_idx, config_type, batch_dir, batch_dir2):
    print(f"Processing Instance {instance_idx} with {instance['machines']} machines and {instance['jobs']} jobs (Config: {config_type}).")
    start_time = time.time()

    cmax_values, tec_values, cmax_init_values, tec_init_values, cmax_explored, tec_explored = process_instance(
        instance, "6CW", config_type
    )
    exec_time = time.time() - start_time

    save_pareto_front(cmax_values, tec_values, cmax_explored, tec_explored,
                      instance['machines'], instance['jobs'], config_type,
                      instance_idx, exec_time, save_dir=batch_dir)


if __name__ == "__main__":
    base_dir = "./data"
    num_jobs = 200
    machines_list = [5, 10, 15, 20, 40, 60]
    num_instances = 10

    instances_data = load_instances(base_dir, num_jobs, machines_list, num_instances)

    instance_types = [
        {"machines": 20, "jobs": num_jobs},
    ]

    for instance_type in instance_types:
        print(f"Processing: {instance_type['machines']} machines, {instance_type['jobs']} jobs.")

        instances_of_type = [
            instance for instance in instances_data
            if instance["machines"] == instance_type["machines"] and instance["jobs"] == instance_type["jobs"]
        ][1:2]

        batch_dir = "./results/NSGA2"
        batch_dir2 = "./results/NSGA2_initial"
        os.makedirs(batch_dir, exist_ok=True)

        processes = []
        for instance_idx, instance in enumerate(instances_of_type):
            for config_type in ["PS"]:
                p = Process(target=process_instance_parallel, args=(instance, instance_idx, config_type, batch_dir, batch_dir2))
                p.start()
                processes.append(p)

            if len(processes) >= os.cpu_count() - 7:
                for p in processes:
                    p.join()
                processes = []

        for p in processes:
            p.join()

    print("Processing complete.")
