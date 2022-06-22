import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GibsCheckout:
    """
    This class represents the graph of a project through dictionaries of: nodes and their successors, duration of each
    activity and the cash flow of each activity.
    """

    def __init__(self):
        """
        Default settings and libraries.
        """
        # ============================================================================================================ #
        #                                                 Keep in mind!                                                #
        # ============================================================================================================ #
        # The numbering of activities starts at 1, but the indices start at 0.                                         #
        # So, to reference activity 1, we index to 0.                                                                  #
        # ============================================================================================================ #

        # Successors of each activity
        self.activity_successors = np.array([(2, 3, 4), 9, (5, 6, 7), 8, 10, 12, (8, 11), 13, 14, 12, 12, 13, 14, 0])

        # The time corresponds to each activity
        self.activity_duration = np.array([0, 6, 5, 3, 1, 6, 2, 1, 4, 3, 2, 3, 5, 0])

        # Corresponds to the cash flow of each project activity
        self.activity_cash_flow = np.array([0, -140, 318, 312, -329, 153, 193, 361, 24, 33, 387, -386, 171, 0])

        # Activity precedence dictionary
        self.activity_predecessors = np.array([0, 1, 1, 1, 3, 3, 3, (4, 7), 2, 5, 7, (6, 10, 11), (8, 12), (9, 13)])

        # Maximum time for project delivery
        self.deadline = 44

        # Cash value devaluation rate
        self.rate = 0.01

        # Number of samples
        self.sample_number = 500

        # Number of activities listed for the project
        self.project_size = len(self.activity_successors)

        # ============================================================================================================ #
        #                                       Auxiliary arrays and variables:                                        #
        # ============================================================================================================ #

        # 1)    To save the critical paths obtained by the CPM
        self.early_start_time = np.zeros(self.project_size)
        self.early_final_time = np.zeros(self.project_size)
        self.later_start_time = np.zeros(self.project_size)
        self.later_final_time = np.array([self.deadline] * self.project_size)

        # 2)    To save the number of times a certain function is called in the execution of the algorithm and the total
        #       time spent executing each one of them
        self.table = pd.DataFrame({
            'Function': ['Calculate NPV',            # 0
                         'CPM Forward',              # 1
                         'CPM Backward',             # 2
                         'CPM',                      # 3
                         'Validation Path',          # 4
                         'Find Neighbour Schedule',  # 5
                         'Simulated Annealing',      # 6
                         'Total'],                   # 7
            'Number': [0, 0, 0, 0, 0, 0, 0, 0],
            'Initial Schedule': [dt.timedelta(hours=0, minutes=0, seconds=0, milliseconds=0, microseconds=0)]*8,
            'Final Schedule': [dt.timedelta(hours=0, minutes=0, seconds=0, milliseconds=0, microseconds=0)]*8,
            'Total time': [dt.timedelta(hours=0, minutes=0, seconds=0, milliseconds=0, microseconds=0)]*8
        })

        # 3)    Stop condition counter
        self.num_stop_condition = 0

    # ================================================================================================================ #
    #                                            Time manipulation function                                            #
    # ================================================================================================================ #
    @staticmethod
    def time_variation(initial_time, final_time):
        """
        Calculate difference between initial time that a function is called and its final time
        :param initial_time: initial time that a function is called
        :param final_time: final time that a function is called
        :return: the difference between parameters
        """
        i = dt.datetime(year=initial_time.year, month=initial_time.month, day=initial_time.day,
                        hour=initial_time.hour, minute=initial_time.minute, second=initial_time.second,
                        microsecond=initial_time.microsecond)
        f = dt.datetime(year=final_time.year, month=final_time.month, day=final_time.day,
                        hour=final_time.hour, minute=final_time.minute, second=final_time.second,
                        microsecond=final_time.microsecond)
        resp = f - i
        return resp

    def table_complement(self):
        """
        Calculates the total time used in each function and salves it in the auxiliary table
        :return: True any situation
        """
        for i in range(0, self.table.shape[0]):
            self.table.iloc[i, 4] = self.time_variation(self.table.iloc[i, 2], self.table.iloc[i, 3])
        return True

    # ================================================================================================================ #
    #                                               Critical Path Method                                               #
    # ================================================================================================================ #
    # It provides a means of determining which jobs or activities, of the many that comprise a project, are "critical" #
    # in their effect on the total project time and the best way to schedule all activities in the project in order to #
    # meet a deadline with minimal cost.                                                                               #
    #                                                                                                                  #
    # Characteristics of a project:                                                                                    #
    #   1)    Collection of activities that, when completed, mark the end of the project                               #
    #   2)    Activities can be started and stopped independently of each other, within a certain sequence             #
    #   3)    The activities are ordered - that is, they must be performed in technological sequence                   #
    # ================================================================================================================ #
    def cpm_forward(self, est, index_activity=0):
        """
        Calculate early start time list
        :param index_activity: initial activity index
        :param est: early start time list
        :return: early start time calculated
        """
        # 1 - CPM Forward
        if self.table.iloc[1, 1] == 0:                  # Checks if it's the first call of the function
            self.table.iloc[1, 2] = dt.datetime.now()   # Updates the time the function was first called
        self.table.iloc[1, 1] += 1                      # Function call counter

        eft = est + self.activity_duration
        if isinstance(self.activity_successors[index_activity], tuple):  # If there is more than 1 successor
            if self.activity_successors[index_activity][0] != 0:         # If not start or end node
                for node in self.activity_successors[index_activity]:    # Go through the successors
                    # If the most recent activity start schedule is less than the most recent activity end schedule
                    if est[node - 1] < eft[index_activity]:
                        # The most recent schedule to start the activity will be updated
                        est[node - 1] = eft[index_activity]
                    est = self.cpm_forward(est, node - 1)               # Call the function for a new activity
        else:
            if self.activity_successors[index_activity] != 0:
                if est[self.activity_successors[index_activity] - 1] < eft[index_activity]:
                    est[self.activity_successors[index_activity] - 1] = eft[index_activity]
                est = self.cpm_forward(est, self.activity_successors[index_activity] - 1)

        self.table.iloc[1, 3] = dt.datetime.now()       # Updates the last time before the function returns
        return est

    def cpm_backward(self, index_activity, lst, lft):
        """
        Calculate later final time list
        :param index_activity: last activity index
        :param lst: later start time list
        :param lft: later final time list
        :return: later final time list calculated
        """
        # 2 - CPM Backward
        if self.table.iloc[2, 1] == 0:                  # Checks if it's the first call of the function
            self.table.iloc[2, 2] = dt.datetime.now()   # Updates the time the function was first called
        self.table.iloc[2, 1] += 1                      # Function call counter

        lst[index_activity] = lft[index_activity] - self.activity_duration[index_activity]
        if isinstance(self.activity_predecessors[index_activity], tuple):   # If there is more than 1 predecessor
            if self.activity_predecessors[index_activity][0] != 0:          # If not start or end node
                for node in self.activity_predecessors[index_activity]:     # Go through the predecessors
                    # If the most extended final schedule is greater than the most extended initial activity schedule
                    if lft[node - 1] > lst[index_activity]:
                        # The most extended final schedule will be updated
                        lft[node - 1] = lst[index_activity]
                    lft = self.cpm_backward(node - 1, lst, lft)             # Call the function for a new activity
        else:
            if self.activity_predecessors[index_activity] != 0:
                if lft[self.activity_predecessors[index_activity] - 1] > lst[index_activity]:
                    lft[self.activity_predecessors[index_activity] - 1] = lst[index_activity]
                lft = self.cpm_backward(self.activity_predecessors[index_activity] - 1, lst, lft)

        self.table.iloc[2, 3] = dt.datetime.now()       # Updates the last time before the function returns
        return lft

    def cpm(self):
        """
        Calculate the critical path
        :return: Returns an array of lists, where the first element is the earliest start list and the second element is
        the latest end list
        """
        # 3 - CPM
        if self.table.iloc[3, 1] == 0:                  # Checks if it's the first call of the function
            self.table.iloc[3, 2] = dt.datetime.now()   # Updates the time the function was first called
        self.table.iloc[3, 1] += 1                      # Function call counter

        self.early_start_time = self.cpm_forward(self.early_final_time, 0)
        self.early_final_time = self.early_start_time + self.activity_duration
        self.later_final_time = self.cpm_backward(len(self.activity_successors)-1,
                                                  self.later_start_time, self.later_final_time)
        self.later_start_time = self.later_final_time - self.activity_duration

        resp = {'early_start_time': self.early_start_time,
                'early_final_time': self.early_final_time,
                'later_start_time': self.later_start_time,
                'later_final_time': self.later_final_time}

        self.table.iloc[3, 3] = dt.datetime.now()       # Updates the last time before the function returns
        return resp

    # ================================================================================================================ #
    #                                                Net Present Value                                                 #
    # ================================================================================================================ #
    def calculate_npv(self, schedules):
        """
        Calculates net present value for earlier start time scheduling
        :param schedules: early start schedule list in np.array type
        """
        # 0 - Calculate NPV
        if self.table.iloc[0, 1] == 0:                  # Checks if it's the first call of the function
            self.table.iloc[0, 2] = dt.datetime.now()   # Updates the time the function was first called
        self.table.iloc[0, 1] += 1                      # Function call counter

        npv = sum(self.activity_cash_flow * np.exp(-self.rate * (schedules + self.activity_duration)))  # NPV equation

        self.table.iloc[0, 3] = dt.datetime.now()  # Updates the last time before the function returns
        return npv

    # ================================================================================================================ #
    #                                              Validation Function                                                 #
    # ================================================================================================================ #
    def validation_path(self, index_activity, sched):
        """
        Path validation following design constraints
        :param index_activity: activity index
        :param sched: schedule (default: early start time list)
        :return: early start time list
        """
        # 4 - Validation Path
        if self.table.iloc[4, 1] == 0:                  # Checks if it's the first call of the function
            self.table.iloc[4, 2] = dt.datetime.now()   # Updates the time the function was first called
        self.table.iloc[4, 1] += 1                      # Function call counter

        if isinstance(self.activity_successors[index_activity], tuple):  # If there is more than 1 successor
            if self.activity_successors[index_activity][0] != 0:         # If not start or end node
                for node in self.activity_successors[index_activity]:    # Go through the successors
                    if sched[node - 1] < (sched[index_activity] + self.activity_duration[index_activity]):
                        sched[node - 1] = sched[index_activity] + self.activity_duration[index_activity]
                    sched = self.validation_path(node - 1, sched)
        else:
            if self.activity_successors[index_activity] != 0:
                if sched[self.activity_successors[index_activity] - 1] < (sched[index_activity]
                                                                          + self.activity_duration[index_activity]):
                    sched[self.activity_successors[index_activity] - 1] = (sched[index_activity]
                                                                           + self.activity_duration[index_activity])
                sched = self.validation_path(self.activity_successors[index_activity] - 1, sched)

        self.table.iloc[4, 3] = dt.datetime.now()       # Updates the last time before the function returns
        return sched

    # ================================================================================================================ #
    #                                               Auxiliary Function                                                 #
    # ================================================================================================================ #
    def find_neighbour_schedule(self, sched):
        """
        Simulates a delay on a drawn activity and recalculates the schedule from that
        :param sched: the early start time array from a project
        :return:
        """
        # 5 - Find Neighbour Schedule
        if self.table.iloc[5, 1] == 0:                  # Checks if it's the first call of the function
            self.table.iloc[5, 2] = dt.datetime.now()   # Updates the time the function was first called
        self.table.iloc[5, 1] += 1                      # Function call counter

        # 1 - Draw an activity
        index_node = np.random.choice(range(1, 13, 1), 1, replace=True)[0]    # Uniform draw from 1 to 12

        # 2 - Activity start and end limit
        min_start = sched[index_node]
        max_start = self.later_start_time[index_node]

        if min_start < max_start:
            sched[index_node] = min_start + 1   # To indicate a delay in the activity, add 1 to the start date

        # 3 - Evaluates and corrects the new schedule considering the delay
        new_sched = self.validation_path(index_node, sched)

        self.table.iloc[5, 3] = dt.datetime.now()       # Updates the last time before the function returns
        return new_sched

    # ================================================================================================================ #
    #                                               Simulated Annealing                                                #
    # ================================================================================================================ #
    def stop_condition(self, matrix_result):
        """
        Defines the stop condition
        :param matrix_result: result array containing path, npv and probability
        """
        current_npv = matrix_result.iloc[len(matrix_result)-1, 14]
        previous_npv = matrix_result.iloc[len(matrix_result)-2, 14]

        delta = current_npv - previous_npv
        if delta < 0:
            self.num_stop_condition += 1
        else:
            self.num_stop_condition = 0

    def simulated_annealing(self, sched):
        """
        Calculates a sample of NPV, according to a draw if scheduling sequences with probability p
        :param sched: earliest appointment
        :return: NPV, scheduling sequence, probability p for each sample
        """
        # 6 - Simulated Annealing
        if self.table.iloc[6, 1] == 0:                  # Checks if it's the first call of the function
            self.table.iloc[6, 2] = dt.datetime.now()   # Updates the time the function was first called
        self.table.iloc[6, 1] += 1                      # Function call counter

        result_matrix = pd.DataFrame({'act1': [], 'act2': [], 'act3': [], 'act4': [], 'act5': [], 'act6': [],
                                      'act7': [], 'act8': [], 'act9': [], 'act10': [], 'act11': [], 'act12': [],
                                      'act13': [], 'act14': [], 'npv': [], 'probability': []})
        first_row = list(sched)
        first_row.append(self.calculate_npv(sched))
        first_row.append(0)
        result_matrix.loc[len(result_matrix)] = first_row   # original one, from early start time array with its values

        for t in range(self.sample_number-1, 0, -1):
            # 1 - Find neighbour schedule
            new_schedule = self.find_neighbour_schedule(sched)  # new one, from new schedule drawn

            # 2 - Evaluete NPV
            new_npv = self.calculate_npv(new_schedule)
            delta = result_matrix.iloc[len(result_matrix) - 1, 14] - new_npv

            # 3 - Check for acceptance
            if delta > 0:
                new_row = list(new_schedule)
                new_row.append(new_npv)
                new_row.append(1)
                result_matrix.loc[len(result_matrix)] = new_row
                self.stop_condition(result_matrix)
            else:
                if np.log(t) != 0:
                    s = 10 / np.log(t)
                    p = min(1, np.exp(delta * s))
                    u = np.random.binomial(1, p)
                    if u == 1:
                        new_row = list(new_schedule)
                        new_row.append(new_npv)
                        new_row.append(p)
                        result_matrix.loc[len(result_matrix)] = new_row
                        self.stop_condition(result_matrix)

            # Stop condition
            if self.num_stop_condition > 10:
                break

        self.table.iloc[6, 3] = dt.datetime.now()       # Updates the last time before the function returns
        return result_matrix

    # ================================================================================================================ #
    #                                                       MAIN                                                       #
    # ================================================================================================================ #
    def main(self):
        """
        Main function that organizes calls
        :return: None
        """
        # 7 - Main
        if self.table.iloc[7, 1] == 0:                  # Checks if it's the first call of the function
            self.table.iloc[7, 2] = dt.datetime.now()   # Updates the time the function was first called
        self.table.iloc[7, 1] += 1                      # Function call counter

        # 1 - Calculate scheduling data
        schedule = self.cpm()

        # 2 - Call the simulated annealing data
        sim = self.simulated_annealing(schedule['early_start_time'])

        self.table.iloc[7, 3] = dt.datetime.now()  # Updates the last time before the function returns
        # ? - Calculates the total time spent on each function
        self.table_complement()
        return (sim[['npv']][1:]).max()


# ==================================================================================================================== #
#                                                    Generate graph                                                    #
# ==================================================================================================================== #
def generate_graph(max_npv):
    max_npv.sort()
    plt.plot(max_npv)
    plt.xlabel('Indices')
    plt.ylabel('max-NPV')
    plt.savefig('simAnnealingModified_maxNPV_plot.png')
    plt.close()
    plt.scatter(range(1, len(max_npv)+1, 1), max_npv)
    plt.xlabel('Indices')
    plt.ylabel('max-NPV')
    plt.savefig('simAnnealingModified_maxNPV_scatter.png')
    plt.close()


# ==================================================================================================================== #
#                                                    Profile table                                                     #
# ==================================================================================================================== #
main_table = pd.DataFrame({
    'Function': ['Calculate NPV',             # 0
                 'CPM Forward',               # 1
                 'CPM Backward',              # 2
                 'CPM',                       # 3
                 'Validation Path',           # 4
                 'Find Neighbour Schedule',   # 5
                 'Simulated Annealing',       # 6
                 'Total'],                    # 7
    'Number': [0, 0, 0, 0, 0, 0, 0, 0],
    'Total time': [dt.timedelta(hours=0, minutes=0, seconds=0, milliseconds=0, microseconds=0)]*8
})


# ==================================================================================================================== #
#                                                       Result                                                         #
# ==================================================================================================================== #
npv_samples = []
num = 100
while num != 0:
    g = GibsCheckout()
    mNPV = g.main()
    # Save the main values
    npv_samples.append(mNPV['npv'])
    # print('# ---- ', num, ' ---- #')
    # print(g.table[['Function', 'Number', 'Total time']])

    main_table.iloc[0, 1] = main_table.iloc[0, 1] + g.table.iloc[0, 1]   # 0 - 'Calculate NPV'
    main_table.iloc[1, 1] = main_table.iloc[1, 1] + g.table.iloc[1, 1]   # 1 - 'CPM Forward'
    main_table.iloc[2, 1] = main_table.iloc[2, 1] + g.table.iloc[2, 1]   # 2 - 'CPM Backward'
    main_table.iloc[3, 1] = main_table.iloc[3, 1] + g.table.iloc[3, 1]   # 3 - 'CPM'
    main_table.iloc[4, 1] = main_table.iloc[4, 1] + g.table.iloc[4, 1]   # 4 - 'Validation Path'
    main_table.iloc[5, 1] = main_table.iloc[5, 1] + g.table.iloc[5, 1]   # 5 - 'Find Neighbour Schedule'
    main_table.iloc[6, 1] = main_table.iloc[6, 1] + g.table.iloc[6, 1]   # 6 - 'Simulated Annealing'
    main_table.iloc[7, 1] = main_table.iloc[7, 1] + g.table.iloc[7, 1]   # 7 - 'Total'

    main_table.iloc[0, 2] = main_table.iloc[0, 2] + g.table.iloc[0, 4]   # 0 - 'Calculate NPV'
    main_table.iloc[1, 2] = main_table.iloc[1, 2] + g.table.iloc[1, 4]   # 1 - 'CPM Forward'
    main_table.iloc[2, 2] = main_table.iloc[2, 2] + g.table.iloc[2, 4]   # 2 - 'CPM Backward'
    main_table.iloc[3, 2] = main_table.iloc[3, 2] + g.table.iloc[3, 4]   # 3 - 'CPM'
    main_table.iloc[4, 2] = main_table.iloc[4, 2] + g.table.iloc[4, 4]   # 4 - 'Validation Path'
    main_table.iloc[5, 2] = main_table.iloc[5, 2] + g.table.iloc[5, 4]   # 5 - 'Find Neighbour Schedule'
    main_table.iloc[6, 2] = main_table.iloc[6, 2] + g.table.iloc[6, 4]   # 6 - 'Simulated Annealing'
    main_table.iloc[7, 2] = main_table.iloc[7, 2] + g.table.iloc[7, 4]   # 7 - 'Total'

    num -= 1

print('# ---- 000 ---- #')
print(main_table[['Function', 'Number', 'Total time']])
print('# ---- 001 ---- #')
print(npv_samples)
print('# ---- 002 ---- #')
print('Min :    ', np.min(npv_samples))
print('Mean :   ', np.mean(npv_samples))
print('Median : ', np.median(npv_samples))
print('Max :    ', np.max(npv_samples))

# Generate graphs with the results obtained for max-NPV
generate_graph(npv_samples)
