import matplotlib.pyplot as plt
import time
import random
import numpy as np

# get the data in each test case file as a 2d list
def get_data(file_path):
    test_cases = []

    with open(file_path, 'r') as file:
        for line in file:
            nums = list(map(int, line.strip().split()))
            test_cases.append(nums)
    return test_cases

# increment each assignmnet through changing bit values
def increment(assignment):
   
    for i in range(len(assignment)):
        if assignment[i] == 0:
            assignment[i] = 1
            return True
        assignment[i] = 0
    return False

# function to check if the given argument satisfies the formula
def check(Wff, Nvars, Nclauses, Assignment):
    # Run thru all possibilities for assignments to wff
    # At each iteration the assignment is "incremented" to next possible
    while True:
        satisfiable = True
      
        for i in range(Nclauses):
            Clause = Wff[i]
            clause_satisfiable = False
            # loop through each literal in clause
            for Literal in Clause:
                Var = abs(Literal)
                # check if literal makes clause satisfiable
                if (Literal > 0 and Assignment[Var] == True) or (Literal < 0 and Assignment[Var] == False):
                    clause_satisfiable = True
                    break
            # if not satifiable, formula unsatifiable
            if not clause_satisfiable:
                satisfiable = False
                break
        # if satisfiable, return true
        if satisfiable:
            return True
        # increment binary array
        if not increment(Assignment):
            return False



def build_wff(Nvars,Nclauses,LitsPerClause):
    wff=[]
    for i in range(1,Nclauses+1):
        clause=[]
        for j in range(1,LitsPerClause+1):
            var=random.randint(1,Nvars)
            if random.randint(0,1)==0: var=-var
            clause.append(var)
        wff.append(clause)
    return wff

# function to test satisfiability of a given formula and measure execution time
def test_wff(wff,Nvars,Nclauses):
    Assignment=list((0 for x in range(Nvars+2)))
    start = time.time() # Start timer
    SatFlag=check(wff,Nvars,Nclauses,Assignment)
    end = time.time() # End timer
    exec_time=int((end-start)*1e6)
    return [wff,Assignment,SatFlag,exec_time]

def run_cases(TestCases, ProbNum, resultsfile, tracefile, cnffile):
    ShowAnswer = True
    print("S/U will be shown on cnf file" if ShowAnswer else "S/U will NOT be shown on cnf file")

    # open files for writing
    with open(f"{resultsfile}.csv", 'w') as f1, \
         open(f"{tracefile}.csv", 'w') as f2, \
         open(f"{cnffile}.cnf", "w") as f3:

        # make lists
        sizes, times, sflags = [], [], []
        Nwffs = Nsat = Nunsat = 0

        # loop through test cases
        for TestCase in TestCases:
            Nvars, NClauses, LitsPerClause, Ntrials = TestCase
            Scount = Ucount = 0
            AveStime = AveUtime = 0
            MaxStime = MaxUtime = 0

            # run trials for current test case
            for trial_num in range(1, Ntrials + 1):
                Nwffs += 1
                random.seed(ProbNum)
                wff = build_wff(Nvars, NClauses, LitsPerClause)
                results = test_wff(wff, Nvars, NClauses)
                # record results
                sizes.append(Nvars)
                times.append(results[3] / LitsPerClause)
                sflags.append(results[2])

                # update counts and time stats based on satisfiability
                if results[2]:
                    y = 'S'
                    Scount += 1
                    AveStime += results[3]
                    MaxStime = max(MaxStime, results[3])
                    Nsat += 1
                else:
                    y = 'U'
                    Ucount += 1
                    AveUtime += results[3]
                    MaxUtime = max(MaxUtime, results[3])
                    Nunsat += 1

                # Prepare the labeled output format
                x = (f"ProbNum={ProbNum}, NumVars={Nvars}, NumClauses={NClauses}, "
                     f"LitsPerClause={LitsPerClause}, TrialNum={trial_num}, "
                     f"TotalLiterals={NClauses * LitsPerClause}, "
                     f"Satisfiability={y}, Time={results[3]} μs")

                # Include the variable assignments for satisfiable cases
                if results[2]:
                    x += ', Assignment=' + ','.join(map(str, results[1][1:Nvars+1]))

                print(x)
                f1.write(x + '\n')
                f2.write(x + '\n')

                # add wff to cnf file
                if not ShowAnswer:
                    y = '?'
                f3.write(f"c {ProbNum} {LitsPerClause} {y}\n")
                f3.write(f"p cnf {Nvars} {NClauses}\n")
                # write in each clause
                for clause in wff:
                    f3.write(' '.join(map(str, clause)) + ' 0\n')

                # increment problem number
                ProbNum += 1

            # write summary stats for this test case
            counts = f'# Satisfied = {Scount}. # Unsatisfied = {Ucount}'
            maxs = f'Max Sat Time = {MaxStime}. Max Unsat Time = {MaxUtime}'
            aves = f'Ave Sat Time = {AveStime/Ntrials}. Ave UnSat Time = {AveUtime/Ntrials}'
            print(counts)
            print(maxs)
            print(aves)
            f2.write(counts + '\n')
            f2.write(maxs + '\n')
            f2.write(aves + '\n')

        # write final summary
        x = f"{cnffile},TheBoss,{Nwffs},{Nsat},{Nunsat},{Nwffs},{Nwffs}\n"
        f1.write(x)

    # plot the results
    plot_results(sizes, times, sflags)


# function to plot the results of the SAT solver runs
def plot_results(sizes, times, flags):
    plt.figure(figsize=(10, 6))
    added_labels = set()
    unsat_sizes, unsat_times = [], []

    # iterate over sizes, times, and satisfiability flags
    for size, time, is_satisfiable in zip(sizes, times, flags):
        # if satisfiable, add green dot on plot
        if is_satisfiable:
            label = 'Satisfiable'
            if label not in added_labels:
                plt.scatter(size, time, color='blue', marker='o', label=label)
                added_labels.add(label)
            else:
                plt.scatter(size, time, color='blue', marker='o')
        # if unsatisfiable, add red x on plot
        else:
            label = 'Unsatisfiable'
            if label not in added_labels:
                plt.scatter(size, time, color='red', marker='o', label=label)
                added_labels.add(label)
            else:
                plt.scatter(size, time, color='red', marker='o')
            unsat_sizes.append(size)
            unsat_times.append(time)

    # axis labels and titles
    plt.xlabel('Number of Variables')
    plt.ylabel('Execution Time (μs)')
    plt.title('Incremental SAT Solver')

    # if unsatisfiable, produce line of best fit with data points
    if unsat_sizes:
        log_unsat_times = np.log(unsat_times)
        fit = np.polyfit(unsat_sizes, log_unsat_times, 1)
        a, b = np.exp(fit[1]), fit[0]

        # define exponential function
        def exp_fit(x):
            return a * np.exp(b * x)

        # plot line
        sizes_fit = np.linspace(min(unsat_sizes), max(unsat_sizes), 100)
        times_fit = exp_fit(sizes_fit)
        plt.plot(sizes_fit, times_fit, '--k', label=f'Best Fit (Unsatisfied): y = {a:.2f} * e^({b:.2f}x)')

    # plot features
    plt.legend()
    plt.grid(True)
    plt.savefig("plot_mzharima.png")
    plt.show()
    plt.close()

# main execution
if __name__ == "__main__":
    # txt file in format [number of variables, number of clauses, literals per clause, trial numbers]
    file_path = 'test_cases_mzharima.txt'
    TestCases = get_data(file_path)
    ProbNum = 3

    resultsfile = 'output_results_file_mzharima'
    tracefile = 'output_tracefile_mzharima'
    cnffile = 'output_cnffile_mzharima'

    run_cases(TestCases, ProbNum, resultsfile, tracefile, cnffile)