import random
import numpy as np
import math
import os
import csv
import time
from multiprocessing import Process


def load_instances(base_dir, num_jobs, num_machines_list, num_instances):
    instances_data = []

    for num_machines in num_machines_list:
        for instance in range(1, num_instances + 1):
            instance_data = {}
            instance_data["jobs"] = None
            instance_data["machines"] = num_machines
            instance_data["instance_number"] = instance
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

            for tag in ["6CW", "CM"]:
                if tag == "6CW":
                    file_name = f"VFR{num_jobs}_{num_machines}_{instance}_Gap__{tag}.txt"
                else:
                    file_name = f"VFR{num_jobs}_{num_machines}_{instance}_Gap_{tag}.txt"
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


def calculate_cmax(schedule, machines, jobs, processing_times):
    job_id = schedule[machines - 1][jobs - 1][0]
    makespan = schedule[machines - 1][jobs - 1][1] + processing_times[job_id][machines - 1]
    return makespan


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


def evaluate(individual, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices, machines, jobs):
    cmax = calculate_cmax(individual, machines, jobs, processing_times)
    tec = calculate_tec(individual, processing_times, energy_prices, time_periods_start, time_periods_end, energy_consumption_rates)
    return int(cmax), tec


def calculate_tec_mach(schedule, processing_times, energy_prices, time_periods_start, time_periods_end, energy_rates, machine_index):
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


def update_start_times(schedule, num_machines, jobs, processing_times):
    for machine in range(num_machines):
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
            finish_time = start_time + processing_time


def update_start_times_local(schedule, num_machines, num_jobs, processing_times, machine_idx):
    for machine in range(machine_idx, num_machines):
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


def dominates(fitness_a, fitness_b):
    makespan_a, tec_a = fitness_a
    makespan_b, tec_b = fitness_b
    return (makespan_a <= makespan_b and tec_a <= tec_b) and (makespan_a < makespan_b or tec_a < tec_b)


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
    if random.random() > 0.2:
        job_info = right_shift_schedule(schedule, num_machines, num_jobs, processing_times, period_ends, job_info, chosen_periods)

    final_schedule = []
    for m in range(num_machines):
        machine_schedule = []
        for job_id, _ in schedule[m]:
            start_time = job_info[job_id][m]['start']
            machine_schedule.append((job_id, start_time))
        final_schedule.append(machine_schedule)

    return final_schedule


def insert_jobs_within_machine(schedule, processing_times, machines, jobs, energy_prices, energy_consumption_rates, time_periods_end, time_periods_start, time_horizon):
    solutions = []
    fitness_Set = []
    best_solution = None

    original_fitness = evaluate(schedule, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices, machines, jobs)
    original_tec, original_cmax = original_fitness[1], original_fitness[0]

    essay = 0
    maxessay = round(2 + (jobs - 10) * (10 - 2) / (800 - 10))

    while essay < maxessay:
        essay += 1
        num_jobs_to_insert = random.randint(1, int(jobs / 10))

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

        new_fitness = evaluate(new_schedule, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices, machines, jobs)
        new_tec, new_cmax = new_fitness[1], new_fitness[0]
        delta_tec = new_tec - original_tec
        delta_cmax = new_cmax - original_cmax

        if delta_tec == 0 and delta_cmax == 0:
            continue

        if new_cmax <= time_horizon:
            if dominates(new_fitness, original_fitness):
                best_solution = [list(machine) for machine in new_schedule]
                fitness_Set.append(new_fitness)
                return solutions, best_solution, fitness_Set
            else:
                temp = [list(machine) for machine in new_schedule]
                solutions.append(temp)
                fitness_Set.append(new_fitness)

    return solutions, best_solution, fitness_Set


def job_swap_on_one_machine(individual, processing_times, machines, jobs, energy_prices, energy_consumption_rates, time_periods_end, time_periods_start, time_horizon):
    solutions = []
    fitness_Set = []
    best_solution = None

    original_fitness = evaluate(individual, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices, machines, jobs)
    original_tec, original_cmax = original_fitness[1], original_fitness[0]

    essay = 0
    maxessay = round(2 + (jobs - 10) * (10 - 2) / (800 - 10))

    while essay < maxessay:
        essay += 1
        num_jobs_in_subsequence = random.randint(1, int(jobs / 10))

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

        new_fitness = evaluate(new_schedule, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices, machines, jobs)
        new_tec, new_cmax = new_fitness[1], new_fitness[0]
        delta_tec = new_tec - original_tec
        delta_cmax = new_cmax - original_cmax

        if delta_tec == 0 and delta_cmax == 0:
            continue

        if new_cmax <= time_horizon:
            if dominates(new_fitness, original_fitness):
                best_solution = [list(machine) for machine in new_schedule]
                fitness_Set.append(new_fitness)
                return solutions, best_solution, fitness_Set
            else:
                temp = [list(machine) for machine in new_schedule]
                solutions.append(temp)
                fitness_Set.append(new_fitness)

    return solutions, best_solution, fitness_Set


def job_swap_on_one_machine_logic(individual, processing_times, machines, jobs, energy_prices, energy_consumption_rates, time_periods_end, time_periods_start, time_horizon):
    tec_values = []
    solutions = []
    fitness_Set = []
    best_solution = None

    original_fitness = evaluate(individual, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices, machines, jobs)
    original_tec, original_cmax = original_fitness[1], original_fitness[0]

    for machine in range(machines):
        tec = calculate_tec_mach(individual, processing_times, energy_prices, time_periods_start, time_periods_end, energy_consumption_rates, machine)
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

        new_fitness = evaluate(new_individual, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices, machines, jobs)
        new_tec, new_cmax = new_fitness[1], new_fitness[0]
        delta_tec, delta_cmax = new_tec - original_tec, new_cmax - original_cmax

        if delta_tec == 0 and delta_cmax == 0:
            continue

        if new_cmax <= time_horizon:
            if dominates(new_fitness, original_fitness):
                best_solution = [list(machine) for machine in new_individual]
                fitness_Set.append(new_fitness)
                return solutions, best_solution, fitness_Set
            else:
                temp = [list(machine) for machine in new_individual]
                solutions.append(temp)
                fitness_Set.append(new_fitness)

    return solutions, best_solution, fitness_Set


def insert_jobs_within_machine_logic(schedule, processing_times, machines, jobs, energy_prices, energy_consumption_rates, time_periods_end, time_periods_start, time_horizon):
    sorted_periods = sorted(range(len(time_periods_start)), key=lambda i: energy_prices[i])

    tec_values = []
    solutions = []
    fitness_Set = []
    best_solution = None

    original_fitness = evaluate(schedule, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices, machines, jobs)
    original_tec, original_cmax = original_fitness[1], original_fitness[0]

    for machine in range(machines):
        tec = calculate_tec_mach(schedule, processing_times, energy_prices, time_periods_start, time_periods_end, energy_consumption_rates, machine)
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

        new_fitness = evaluate(new_schedule, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices, machines, jobs)
        new_tec, new_cmax = new_fitness[1], new_fitness[0]
        delta_tec = new_tec - original_tec
        delta_cmax = new_cmax - original_cmax

        if delta_tec == 0 and delta_cmax == 0:
            continue

        if new_cmax <= time_horizon:
            if dominates(new_fitness, original_fitness):
                best_solution = [list(machine) for machine in new_schedule]
                fitness_Set.append(new_fitness)
                return solutions, best_solution, fitness_Set
            else:
                temp = [list(machine) for machine in new_schedule]
                solutions.append(temp)
                fitness_Set.append(new_fitness)

    return solutions, best_solution, fitness_Set


def machine_sequence_swap_logic(schedule, processing_times, machines, jobs, energy_prices, energy_consumption_rates, time_periods_end, time_periods_start, time_horizon):
    tec_values = []
    solutions = []
    fitness_Set = []
    best_solution = None

    original_fitness = evaluate(schedule, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices, machines, jobs)
    original_tec, original_cmax = original_fitness[1], original_fitness[0]

    new_schedule = [list(machine) for machine in schedule]

    for machine in range(machines):
        tec = calculate_tec_mach(new_schedule, processing_times, energy_prices, time_periods_start, time_periods_end, energy_consumption_rates, machine)
        tec_values.append(tec)

    sorted_tec_indices = sorted(range(machines), key=lambda x: tec_values[x], reverse=True)
    machine1 = sorted_tec_indices[0]
    machine2 = sorted_tec_indices[1]

    job_indexes1 = [job[0] for job in new_schedule[machine1]]
    job_indexes2 = [job[0] for job in new_schedule[machine2]]

    for i, job in enumerate(new_schedule[machine1]):
        new_schedule[machine1][i] = (job_indexes2[i], job[1])

    for i, job in enumerate(new_schedule[machine2]):
        new_schedule[machine2][i] = (job_indexes1[i], job[1])

    update_start_times(new_schedule, machines, jobs, processing_times)

    new_fitness = evaluate(new_schedule, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices, machines, jobs)
    new_tec, new_cmax = new_fitness[1], new_fitness[0]

    if new_cmax <= time_horizon:
        if dominates(new_fitness, original_fitness):
            best_solution = [list(machine) for machine in new_schedule]
            return solutions, best_solution, fitness_Set
        else:
            temp = [list(machine) for machine in new_schedule]
            solutions.append(temp)
            fitness_Set.append(new_fitness)

    return solutions, best_solution, fitness_Set


def normalize_fitness(all_solutions, new_cmax, new_tec, previous_cmax, previous_tec):
    scale_factor = 10

    if len(all_solutions) == 1:
        delta_cmax_normalized = new_cmax - previous_cmax
        delta_tec_normalized = new_tec - previous_tec
        return delta_cmax_normalized, delta_tec_normalized

    cmax_min = min([fitness[0] for fitness in all_solutions])
    cmax_max = max([fitness[0] for fitness in all_solutions])
    tec_min = min([fitness[1] for fitness in all_solutions])
    tec_max = max([fitness[1] for fitness in all_solutions])

    if cmax_max > cmax_min:
        delta_cmax_normalized = math.log1p(abs(new_cmax - previous_cmax)) * (1 if new_cmax >= previous_cmax else -1)
    else:
        delta_cmax_normalized = (new_cmax - previous_cmax)

    if tec_max > tec_min:
        delta_tec_normalized = math.log1p(abs(new_tec - previous_tec)) * (1 if new_tec >= previous_tec else -1)
    else:
        delta_tec_normalized = (new_tec - previous_tec)

    delta_cmax_normalized = round(delta_cmax_normalized, 2) * scale_factor
    delta_tec_normalized = round(delta_tec_normalized, 2) * scale_factor

    return delta_cmax_normalized, delta_tec_normalized


def update_weights(tec_weight, cmax_weight, delta_cmax, delta_tec):
    if delta_tec > 0 and delta_cmax <= 0:
        if tec_weight < 0.9:
            tec_weight += 0.05
        if cmax_weight > 0.1:
            cmax_weight -= 0.05
    elif delta_cmax > 0 and delta_tec <= 0:
        if cmax_weight < 0.9:
            cmax_weight += 0.05
        if tec_weight > 0.1:
            tec_weight -= 0.05
    if delta_tec > 0 and delta_cmax > 0:
        if delta_tec > delta_cmax:
            if tec_weight < 0.9:
                tec_weight += 0.05
            if cmax_weight > 0.1:
                cmax_weight -= 0.05
        else:
            if cmax_weight < 0.9:
                cmax_weight += 0.05
            if tec_weight > 0.1:
                tec_weight -= 0.05

    return round(tec_weight, 3), round(cmax_weight, 3)


def VND(initial_schedule, processing_times, machines, jobs, energy_consumption_rates, time_periods_end, energy_prices, time_periods_start, time_horizon):
    local_neighborhoods = [
        insert_jobs_within_machine,
        insert_jobs_within_machine_logic,
        job_swap_on_one_machine,
        job_swap_on_one_machine_logic,
        machine_sequence_swap_logic
    ]

    best_schedule = [list(machine) for machine in initial_schedule]
    initial_fitness = evaluate(initial_schedule, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices, machines, jobs)
    best_fitness = initial_fitness
    all_solutions = []
    fitness_val = []

    for neighborhood_index, neighborhood in enumerate(local_neighborhoods):
        solutions, dominating_solution, fitness_set = neighborhood(
            best_schedule, processing_times, machines, jobs, energy_prices, energy_consumption_rates, time_periods_end, time_periods_start, time_horizon
        )

        if dominating_solution:
            best_schedule = [list(machine) for machine in dominating_solution]
            best_fitness = evaluate(best_schedule, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices, machines, jobs)
            return best_schedule, best_fitness, fitness_val
        else:
            if solutions:
                all_solutions.extend(solutions)
            if fitness_set:
                fitness_val.extend(fitness_set)

    if len(all_solutions) == 0:
        return initial_schedule, initial_fitness, fitness_val

    selected_solution = random.choice(all_solutions)
    selected_solution_fitness = evaluate(selected_solution, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices, machines, jobs)

    best_schedule = [list(machine) for machine in selected_solution]
    best_fitness = selected_solution_fitness

    return best_schedule, best_fitness, fitness_val


def update_archive(archive, new_solution, new_fitness, processing_times, energy_consumption_rates, time_periods_end, energy_prices, time_periods_start, machines, jobs):
    to_remove = []
    for archived_solution in archive:
        archived_fitness = evaluate(archived_solution, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices, machines, jobs)
        if dominates(archived_fitness, new_fitness):
            return archive
        elif dominates(new_fitness, archived_fitness):
            to_remove.append(archived_solution)

    for solution in to_remove:
        archive.remove(solution)

    if new_solution not in archive:
        archive.append(new_solution)

    return archive


def simulated_annealing(initial_solution, initial_fitness, processing_times, time_periods_end, energy_prices, energy_consumption_rates, machines, jobs, time_periods_start, initial_temperature, alpha, num_iterations, time_horizon):
    best_solution = [list(machine) for machine in initial_solution]
    best_fitness = initial_fitness
    best_tec = best_fitness[1]
    best_cmax = best_fitness[0]

    unique_fitness_values = set([best_fitness])
    all_solutions = [best_fitness]
    archive = [best_solution]

    temperature = initial_temperature
    tec_weight = 0.5
    cmax_weight = 0.5
    no_improvement_counter = 0

    while temperature > 1e-3:
        for iteration in range(num_iterations):
            temp = [list(machine) for machine in best_solution]

            new_schedule, new_fitness, glob = VND(temp, processing_times, machines, jobs, energy_consumption_rates, time_periods_end, energy_prices, time_periods_start, time_horizon)

            for s in glob:
                if s not in unique_fitness_values:
                    unique_fitness_values.add(s)
                    all_solutions.append(s)

            if is_schedule_feasible(new_schedule, processing_times) and new_fitness[0] <= time_horizon:
                if new_fitness not in unique_fitness_values:
                    unique_fitness_values.add(new_fitness)
                    all_solutions.append(new_fitness)
            else:
                continue

            temp3 = [list(machine) for machine in new_schedule]
            new_schedule2 = tec_reducer(temp3, machines, jobs, processing_times, time_periods_start, time_periods_end, energy_prices)
            new_fitness2 = evaluate(new_schedule2, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices, machines, jobs)

            if is_schedule_feasible(new_schedule2, processing_times) and new_fitness2[0] <= time_horizon:
                if dominates(new_fitness2, new_fitness):
                    new_schedule = [list(machine) for machine in new_schedule2]
                    new_fitness = new_fitness2
                if new_fitness2 not in unique_fitness_values:
                    unique_fitness_values.add(new_fitness2)
                    all_solutions.append(new_fitness2)

            new_tec, new_cmax = new_fitness[1], new_fitness[0]

            archive = update_archive(archive, new_schedule, new_fitness, processing_times, energy_consumption_rates, time_periods_end, energy_prices, time_periods_start, machines, jobs)

            if dominates(new_fitness, best_fitness):
                best_solution = [list(machine) for machine in new_schedule]
                best_fitness = new_fitness
                best_tec = new_tec
                best_cmax = new_cmax
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

                delta_tec = new_tec - best_tec
                delta_cmax = new_cmax - best_cmax

                if delta_tec == 0 and delta_cmax == 0:
                    continue

                tec_weight, cmax_weight = update_weights(tec_weight, cmax_weight, delta_cmax, delta_tec)
                delta_cmax_normalized, delta_tec_normalized = normalize_fitness(all_solutions, new_cmax, new_tec, best_cmax, best_tec)
                delta_e = tec_weight * delta_tec_normalized + cmax_weight * delta_cmax_normalized
                acceptance_probability = min(1, math.exp(-abs(delta_e) / temperature))

                if random.random() < acceptance_probability:
                    best_solution = [list(machine) for machine in new_schedule]
                    best_fitness = new_fitness
                    best_tec = new_tec
                    best_cmax = new_cmax

        temperature *= alpha

        if no_improvement_counter >= 10:
            best_solution = random.choice(archive)
            best_fitness = evaluate(best_solution, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices, machines, jobs)
            no_improvement_counter = 0

    return new_schedule, new_fitness, all_solutions


def process_instance(instance, energy_config, consumption_config, initial_temperature, alpha, num_iterations):
    machines = instance["machines"]
    jobs = instance["jobs"]
    processing_times = instance["processing_times"]
    energy_prices_data = instance["energy_prices"][energy_config]
    time_horizon = energy_prices_data["time_horizon"]
    time_periods_end = energy_prices_data["end"]
    time_periods_start = energy_prices_data["start"]
    energy_prices = energy_prices_data["prices"]
    energy_consumption_rates = instance["energy_consumption_rates"][consumption_config]

    p_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    if jobs <= 60:
        best_cmax = float('inf')
        for p in p_values:
            new_schedule = nfs_heuristic(machines, jobs, processing_times, p)
            cmax, tec = evaluate(new_schedule, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices, machines, jobs)
            if cmax < best_cmax:
                best_cmax = cmax
                best_p = p
        p = best_p
    else:
        p = random.choice(p_values)

    new_schedule = nfs_heuristic(machines, jobs, processing_times, p)
    update_start_times(new_schedule, machines, jobs, processing_times)

    fitness = evaluate(new_schedule, processing_times, energy_consumption_rates, time_periods_end, time_periods_start, energy_prices, machines, jobs)
    print(f"Initial solution fitness: {fitness}")

    updated_individual, updated_fitness, all_solutions = simulated_annealing(
        new_schedule, fitness, processing_times, time_periods_end, energy_prices,
        energy_consumption_rates, machines, jobs, time_periods_start,
        initial_temperature, alpha, num_iterations, time_horizon
    )

    return all_solutions, fitness


def is_dominated(solution1, solution2):
    cmax1, tec1 = solution1
    cmax2, tec2 = solution2
    return (cmax1 >= cmax2 and tec1 >= tec2) and (cmax1 > cmax2 or tec1 > tec2)


def get_pareto_front(fitness_values):
    pareto_front = []
    for candidate in fitness_values:
        dominated = False
        for other in fitness_values:
            if is_dominated(candidate, other):
                dominated = True
                break
        if not dominated:
            pareto_front.append(candidate)

    pareto_front.sort(key=lambda x: (x[0], x[1]))
    return pareto_front


CSV_DIR = "./results/SA"
base_dir = "./data"
machines_list = [5, 10, 15, 20, 40, 60]
num_jobs = 400
num_instances = 10


def process_instance_parallel(instance, instance_idx, config1, config2, config_name):
    machines = instance["machines"]
    jobs = instance["jobs"]
    instance_number = instance["instance_number"]

    start_time = time.time()

    if jobs <= 60:
        alpha = 0.99
    elif jobs <= 200:
        alpha = 0.97
    else:
        alpha = 0.9

    print(f"Processing instance {instance_idx % 10}: {jobs} jobs, {machines} machines, config {config_name}")

    all_solutions, initial_solution = process_instance(instance, config1, config2, jobs, alpha, 10)

    pareto_solutions = get_pareto_front([fitness for fitness in all_solutions])
    pareto_solutions.sort(key=lambda x: x[1])

    os.makedirs(CSV_DIR, exist_ok=True)
    end_time = time.time()
    execution_time = end_time - start_time

    instance_csv_path = os.path.join(CSV_DIR, f"M{machines}_J{jobs}_config_{config1}_{instance_number % 10}.csv")
    file_exists = os.path.exists(instance_csv_path)

    with open(instance_csv_path, mode='a', newline='') as csvfile:
        fieldnames = ["Makespan", "TEC", "Pareto", "Execution Time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for fitness in pareto_solutions:
            result_data = {
                "Makespan": fitness[0],
                "TEC": fitness[1],
                "Pareto": "True",
                "Execution Time": execution_time
            }
            writer.writerow(result_data)


def main(instances_data):
    filtered_instances = [instance for instance in instances_data if instance["jobs"] == num_jobs]

    grouped_instances = {60: [], 40: [], 20: []}
    for instance in filtered_instances:
        machines = instance["machines"]
        if machines in grouped_instances:
            grouped_instances[machines].append(instance)

    configurations = [("6CW", "PS", "6CW_PS")]
    selected_instances = []
    for machines, instances in grouped_instances.items():
        selected_instances.extend(instances[:10])

    processes = []
    for instance_idx, instance in enumerate(selected_instances):
        for config1, config2, config_name in configurations:
            p = Process(target=process_instance_parallel, args=(instance, instance_idx, config1, config2, config_name))
            p.start()
            processes.append(p)

        if len(processes) >= os.cpu_count() - 7:
            for p in processes:
                p.join()
            processes = []

    for p in processes:
        p.join()

    print(f"CSV files saved in '{CSV_DIR}'.")


if __name__ == "__main__":
    instances_data = load_instances(base_dir, num_jobs, machines_list, num_instances)
    main(instances_data)
    print("Processing complete.")
