# import system function
from datetime import datetime
from operator import itemgetter

# import from project functions
import initial as init


# write the solution to file
def write_solution(initial_puzzle, puzzle_map, solution, file_path):
    rows = initial_puzzle[0][0]
    cols = initial_puzzle[0][1]

    initial_bulb = []

    for i in range(0, rows):
        for j in range(0, cols):
            if puzzle_map[i][j] == init.CELL_BULB:
                initial_bulb.append([i, j])

    for i in range(0, len(solution)):
        total_bulb = initial_bulb + solution[i].get("bulbs")
        total_bulb.sort()
        first_subfitness = solution[i].get('white_cells') - solution[i].get('empty_cells')
        second_subfitness = solution[i].get('shining_conflict')
        third_subfitness = solution[i].get('black_conflict')

        with open(file_path, 'a+') as wf:
            wf.write(f'\n{first_subfitness}    {second_subfitness}    {third_subfitness}    {i + 1}\n')
            for j in range(0, len(total_bulb)):
                wf.write(f'{total_bulb[j][1] + 1} {total_bulb[j][0] + 1}\n')
        wf.close()


# write logs
def logs_write(log_file_path, runs, run_log):
    with open(log_file_path, 'a+') as wf:
        wf.write(f"\nrun {runs + 1} \n")
        for i in range(0, len(run_log)):
            wf.write(f'{str(run_log[i][0]).ljust(6, " ")}'
                     f'   {str(run_log[i][1]).ljust(4, " ")}\n'
                     )
    wf.close()

