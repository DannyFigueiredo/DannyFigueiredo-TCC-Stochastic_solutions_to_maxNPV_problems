import random

import numpy as np
from random import sample, seed
import seaborn as sns
import matplotlib.pyplot as plt


class Gibs:
    """
    This class represents the graph of a project through dictionaries of: nodes and their successors,
    duration of each activity and the cash flow of each activity.
    """

    def __init__(self):
        """
        Default settings and libraries.
        """
        # np = __import__('numpy')
        # m = __import__('math')

        # Dictionary of successors of each activity
        self.activity_successors = {
            1: [2, 3, 4],
            2: [9],
            3: [5, 6, 7],
            4: [8],
            5: [10],
            6: [12],
            7: [8, 11],
            8: [13],
            9: [14],
            10: [12],
            11: [12],
            12: [13],
            13: [14],
            14: [0]
        }

        # The time corresponds to each activity
        # key = activity and value = duration time
        self.activity_duration = {
            1: [0],
            2: [6],
            3: [5],
            4: [3],
            5: [1],
            6: [6],
            7: [2],
            8: [1],
            9: [4],
            10: [3],
            11: [2],
            12: [3],
            13: [5],
            14: [0]
        }

        # Corresponds to the cash flow of each project activity
        self.activity_cash_flow = {
            1: [0],
            2: [-140],
            3: [318],
            4: [312],
            5: [-329],
            6: [153],
            7: [193],
            8: [361],
            9: [24],
            10: [33],
            11: [387],
            12: [-386],
            13: [171],
            14: [0]
        }

        # Activity precedence dictionary
        self.activity_predecessors = {
            1: [0],
            2: [1],
            3: [1],
            4: [1],
            5: [3],
            6: [3],
            7: [3],
            8: [4, 7],
            9: [2],
            10: [5],
            11: [7],
            12: [6, 10, 11],
            13: [8, 12],
            14: [9, 13]
        }

        # Maximum time for project delivery
        self.deadline = 44

        # cash value devaluation rate
        self.rate = 0.01

        # number of samples
        self.sample_number = 500

        # number of activities listed for the project
        self.project_size = len(self.activity_successors)

        # auxiliary array for eventual calculations
        self.activity_successors_array = np.array(list(self.activity_successors.values()))
        self.activity_duration_array = np.array(list(self.activity_duration.values()))
        self.activity_cash_flow_array = np.array(list(self.activity_cash_flow.values()))
        self.activity_predecessors_array = np.array(list(self.activity_predecessors.values()))

        # Arrays with predefined sample numbers
        self.npv_samples = np.zeros(self.sample_number)
        self.early_start_time = np.zeros(self.project_size)
        self.early_final_time = np.zeros(self.project_size)
        self.later_start_time = np.zeros(self.project_size)
        self.later_final_time = np.array([self.deadline]*self.project_size)

        # Function counters
        self.num_calculate_net_present_value = 0
        self.num_calculate_cpm_foward = 0
        self.num_calculate_cpm_backward = 0
        self.num_calculate_cpm = 0
        self.num_validation_path = 0
        self.num_validation_deadline_activity = 0
        self.num_find_neighbour_schedule = 0
        self.num_simulated_annealing = 0


    # =============================== #
    #   Parameter Setting Functions   #
    # =============================== #
    def set_activity_successors(self, key: int, value: int):
        """
        Saves a new activity successors node in the dictionary
        :param key: activity number
        :param value: list of possible further activities
        """
        self.activity_successors[key] = value

    def set_activity_duration(self, key: int, value: int):
        """
        Saves a new activity duration node in the dictionary
        :param key: activity number
        :param value: activity duration time
        """
        self.activity_duration[key] = value

    def set_activity_cash_flow(self, key: int, value: int):
        """
        Saves a new cash flow node in the dictionary
        :param key: activity number
        :param value: activity cash flow
        """
        self.activity_cash_flow[key] = value

    def set_activity_predecessors(self, key: int, value: int):
        """
        Saves a new activity precedence node in the dictionary
        :param key: activity
        :param value: list of possible predecessor activities
        """
        self.activity_predecessors[key] = value

    def set_sample_number(self, s: int):
        """
        Set a sample number to simulate the calculations
        :param s: sample number
        """
        self.sample_number = s

    def set_deadline(self, d: int):
        """
        Set a deadline to a project
        :param d: Maximum time for project delivery
        """
        self.deadline = d

    def set_rate(self, r: float):
        """
        Set a rate to a project
        :param r: rate
        """
        self.rate = r

    def reset_npv_samples(self):
        """
        Reset NPV sample array
        """
        self.npv_samples = np.zeros(self.sample_number)

    def update_project_size(self):
        """
        Recalculates the project size (in case the project isn't the default)
        """
        self.project_size = len(self.activity_successors)

    # ============================== #
    #   Parameter Return Functions   #
    # ============================== #
    def get_successors_dic(self) -> dict:
        """
        Returns activities dictionary
        """
        return self.activity_successors

    def get_duration_dic(self) -> dict:
        """
        Returns activity duration dictionary by activity
        """
        return self.activity_duration

    def get_cash_flow_dic(self) -> dict:
        """
        Returns cash flow dictionary by activity
        """
        return self.activity_cash_flow

    def get_predecessors_dic(self) -> dict:
        """
        Returns predecessors activities dictionary
        """
        return self.activity_predecessors

    def get_successors_activity(self, act: int) -> list:
        """
        Returns list of successor activities
        :param act: activity number
        """
        return self.activity_successors[act]

    def get_duration_activity(self, act: int) -> list:
        """
        Returns activity duration
        :param act: activity number
        """
        return self.activity_duration[act]

    def get_cash_flow_activity(self, act: int) -> list:
        """
        Returns activity cash flow
        :param act: activity number
        """
        return self.activity_cash_flow[act]

    def get_predecessor_activity(self, act: int) -> list:
        """
        Returns a list of activities that precede immediately the searched activity using direct dictionary
        :param act: activity you are looking for
        """
        return self.activity_predecessors[act]

    def get_deadline(self) -> int:
        """
        Returns maximum time for project delivery
        :return: deadline
        """
        return self.deadline

    def get_rate(self) -> float:
        """
        Discount rate to net present value per unit of time
        :return: Rate
        """
        return self.rate

    def get_sample_number(self) -> int:
        """
        Returns the number of times calculations were simulated
        :return: sample number
        """
        return self.sample_number

    def get_project_size(self) -> int:
        """
        Returns the number of activities that exist in a project
        :return: number of activities
        """
        return self.project_size

    def get_predecessor_activities_path(self, act: int) -> list:
        """
        Returns a list of all activities that precede the searched activity
        :param act: activity you are looking for
        """
        # list of all possible predecessors of the researched activity
        predecessors = []
        # auxiliary list of predecessor activities
        predecessor_auxiliary_activities = []
        # current activity starting with final activity
        current_activity = max(self.activity_successors.keys())

        while True:
            # From the final activity, a task backtracks until it finds the researched activity
            if current_activity > act:
                current_activity -= 1
                continue

            # Adds current activity to predecessor list
            predecessors.append(current_activity)
            # Adds the list of predecessor activities, the predecessor activities of the current node
            predecessor_auxiliary_activities = predecessor_auxiliary_activities \
                                               + self.get_predecessor_activity(current_activity)
            # Remove duplicate activities
            predecessor_auxiliary_activities = list(set(predecessor_auxiliary_activities))
            # Sort activities from smallest to largest
            predecessor_auxiliary_activities.sort()

            try:
                # Defines the current node as the largest predecessor activity included in the auxiliary list
                current_activity = predecessor_auxiliary_activities[-1]
            except:
                # Defines the end of the activities to be covered
                current_activity = 0

            # When the auxiliary list is empty, break the loop
            if current_activity == 0:
                break

            # Remove the last node (current node) from the auxiliary list
            predecessor_auxiliary_activities.pop()

        return predecessors

    # ======================== #
    #   Get Arrays Functions   #
    # ======================== #
    def get_npv_samples(self):
        """
        Returns the net present value for each one of the samples
        """
        return self.npv_samples

    def get_activity_successors_array(self):
        """
        Returns activity successors list for each one of the activities
        :return: activity successors list
        """
        return self.activity_successors_array

    def get_activity_duration_array(self):
        """
        Returns activity duration list for each one of the activities
        :return: activity duration list
        """
        return self.activity_duration_array

    def get_activity_cash_flow_array(self):
        """
        Returns activity cash flow list for each one of the activities
        :return: activity cash flow list
        """
        return self.activity_cash_flow_array

    def get_activity_predecessors_array(self):
        """
        Returns activity predecessors list for each one of the activities
        :return: activity predecessors list
        """
        return self.activity_predecessors_array

    def get_early_start_time(self):
        """
        Returns early final time list for each one of the activities
        :return: early final time list
        """
        return self.early_start_time

    def get_early_final_time(self):
        """
        Returns early final time list for each one of the activities
        :return: early final time list
        """
        return self.early_final_time

    def get_later_start_time(self):
        """
        Returns later start time list for each one of the activities
        :return: later start time list
        """
        return self.later_start_time

    def get_later_final_time(self):
        """
        Returns later final time list for each one of the activities
        :return: later final time list
        """
        return self.later_final_time

    # ======================= #
    #   Setting New Project   #
    # ======================= #
    def reset_project(self):
        """
        Clears the default characteristics of the project so that another one can be started
        """
        self.activity_successors = {}
        self.activity_duration = {}
        self.activity_cash_flow = {}
        self.activity_predecessors = {}

    def reset_arrays(self):
        """
        Resets the length of the early start times, early final start, later start time and later final start lists
        """
        # auxiliary array for eventual calculations
        self.activity_successors_array = np.array(list(self.activity_successors.values()))
        self.activity_duration_array = np.array(list(self.activity_duration.values()))
        self.activity_cash_flow_array = np.array(list(self.activity_cash_flow.values()))
        self.activity_predecessors_array = np.array(list(self.activity_predecessors.values()))

        # Arrays with predefined sample numbers
        self.npv_samples = np.zeros(self.sample_number)
        self.early_start_time = np.zeros(self.project_size)
        self.early_final_time = np.zeros(self.project_size)
        self.later_start_time = np.zeros(self.project_size)
        self.later_final_time = np.array([self.deadline]*self.project_size)

    def set_new_project(self, s_dic: dict, p_dic: dict, d_dic: dict, c_dic: dict, d: int, r: float):
        """
        Set a new project through dictionaries
        :param s_dic: successors activities dictionary
        :param p_dic: predecessor per activity dictionary
        :param d_dic: activity duration dictionary
        :param c_dic: activity cash flow dictionary
        :param d: deadline
        :param r: devaluation rate
        """
        self.reset_project()
        self.activity_successors = s_dic
        self.activity_predecessors = p_dic
        self.activity_duration = d_dic
        self.activity_cash_flow = c_dic
        self.project_size = len(self.activity_successors)
        self.deadline = d
        self.rate = r
        self.reset_arrays()

    # ===================== #
    #   Net Present Value   #
    # ===================== #
    def calculate_net_present_value(self, schedules):
        """
        Calculates net present value for earlier start time scheduling
        :param schedules: early start schedule list in np.array in np.array type
        """
        self.num_calculate_net_present_value += 1

        est_schedule = sum(schedules, self.activity_duration_array)
        return sum(self.activity_cash_flow_array * np.exp(-self.rate * est_schedule))

    # ============================================================================= #
    #                             Critical Path Method                              #
    # ============================================================================= #
    # It provides a means of determining which jobs or activities, of the many      #
    # that comprise a project, are "critical" in their effect on the total project  #
    # time and the best way to schedule all activities in the project in order to   #
    # meet a deadline with minimal cost.                                            #
    #                                                                               #
    # Characteristics of a project:                                                 #
    # 1) collection of activities that, when completed, mark the end of the project #
    # 2) activities can be started and stopped independently of each other, within  #
    #    a certain sequence                                                         #
    # 3) the activities are ordered - that is, they must be performed in            #
    #    technological sequence                                                     #
    # ============================================================================= #

    def calculate_cpm_foward(self, est, eft, initial_activity=0):
        """
        Calculate early start time list
        :param initial_activity: initial activity
        :param est: early start time list
        :param eft: early final time list
        :return: early start time calculated
        """
        self.num_calculate_cpm_foward += 1

        eft[initial_activity] = est[initial_activity] + self.activity_duration_array[initial_activity][0]
        if self.activity_successors_array[initial_activity][0] != 0:
            for i in self.activity_successors_array[initial_activity]:
                if est[i - 1] < eft[initial_activity]:
                    est[i - 1] = eft[initial_activity]
                est = self.calculate_cpm_foward(est, eft, i - 1)
        return est

    def calculate_cpm_backward(self, lft, lst, size=13):
        """
        Calcylate later final time list
        :param size: last activity -1
        :param lst: later start time list
        :param lft: later final time list
        :return: later final time list calculated
        """
        self.num_calculate_cpm_backward += 1

        lst[size] = lft[size] - self.activity_duration_array[size][0]
        if self.activity_predecessors_array[size][0] != 0:
            for i in self.activity_predecessors_array[size]:
                if lft[i - 1] > lst[size]:
                    lft[i - 1] = lst[size]
                lft = self.calculate_cpm_backward(lft, lst, i - 1)
        return lft

    def calculate_cpm(self, est, eft, initial_activity=1):
        """
        Calculate the critical path
        :param est: early start time list
        :param eft: early final time list
        :param initial_activity: initial activity
        :return: Returns an array of lists, where the first element is the earliest start list and the second element is
        the latest end list
        """
        self.num_calculate_cpm += 1

        self.early_start_time = self.calculate_cpm_foward(est, eft, initial_activity)
        for i in range(0, self.project_size):
            self.early_final_time[i] = self.early_start_time[i] + self.activity_duration_array[i][0]
        self.later_final_time = self.calculate_cpm_backward(self.later_final_time, self.later_start_time,
                                                            self.project_size-1)
        for i in range(0, self.project_size):
            self.later_start_time[i] = self.later_final_time[i] - self.activity_duration_array[i][0]
        resp = {'early_start_time': self.early_start_time,
                'early_final_time': self.early_final_time,
                'later_start_time': self.later_start_time,
                'later_final_time': self.later_final_time}
        return resp

    # ======================== #
    #   Validation Functions   #
    # ======================== #
    def validation_deadline_activity(self, schedule):
        """
        Validation of the project deadline constraint
        :param schedule:
        :return: False or True whether or not it violates the deadline, respectively
        """
        self.num_validation_deadline_activity += 1

        value = True
        for i in range(1, self.project_size):
            pr = self.activity_predecessors_array[i]
            for j in pr:
                if schedule[j - 1] < schedule[j] + self.activity_duration_array[j][0]:
                    value = False
                    break
        return value

    def validation_path(self, act, est):
        """

        :param act: activity index
        :param est: early start time list
        :return: early start time list
        """
        self.num_validation_path += 1

        if self.activity_successors_array[act][0] != 0:
            for i in self.activity_successors_array[act]:
                if est[i - 1] < est[act] + self.activity_duration_array[act]:
                    est[i - 1] = est[act] + self.activity_duration_array[act]
                est = self.validation_path(i-1, est)
        return est

    # ======================= #
    #   Auxiliary Functions   #
    # ======================= #
    def find_neighbour_schedule(self, schedule, s):
        """

        :param schedule:
        :return:
        """
        self.num_find_neighbour_schedule += 1

        random.seed(s)
        # draw an activity
        node = sample(range(0, 13, 1), 1)[0]
        # activity start and end limit
        minimum = schedule[node]
        maximum = self.later_start_time[node]
        if schedule[node] < maximum:
            t = schedule[node] + 1
            # t = sample(range(minimum, maximum, 1), 1)[0]
            schedule[node] = t
        # evaluates the new start schedule earlier
        schedule = self.validation_path(node, schedule)
        return schedule

    # ================================ #
    #   Simulated Annealing function   #
    # ================================ #
    def simulated_annealing(self, schedule):
        """

        :param schedule:
        :return:
        """
        self.num_simulated_annealing += 1

        scheduling_sequence = [0]*self.sample_number
        scheduling_sequence[0] = schedule
        sequence_p = np.zeros(self.sample_number) #np.zeros(shape=(self.sample_number, self.project_size))

        seed = 0
        for t in range(self.sample_number-1, 0, -1):
            seed += t
            n_sched = self.find_neighbour_schedule(schedule, seed)
            scheduling_sequence[t-1] = n_sched

            npv = self.calculate_net_present_value(schedule)
            #self.npv_samples[t-1] = n_sched
            #print(n_sched)
            nnpv = self.calculate_net_present_value(n_sched)
            delta = nnpv - npv

            if delta > 0:
                schedule = n_sched
                self.npv_samples[t-1] = nnpv
                est = schedule
            else:
                print(t, np.log10(t+1), 10/np.log10(t+1), np.exp(delta*(10/np.log10(t+1))))
                s = (10 / np.log10(t+1))
                p = min(1, np.exp(delta*s))
                sequence_p[t] = p
                u = np.random.binomial(1, p)
                if u == 1:
                    schedule = n_sched
                    self.npv_samples[t] = nnpv

        self.npv_samples = self.npv_samples[1:]
        sequence_p = sequence_p[1:]
        resp = {'npv': self.npv_samples,
                'seq': scheduling_sequence,
                'p': sequence_p}
        return resp

    # ================= #
    #   Main function   #
    # ================= #
    def main(self):
        r1 = self.calculate_cpm(self.early_start_time, self.early_final_time, 1)
        est = r1['early_start_time']
        eft = r1['early_final_time']
        lst = r1['later_start_time']
        lft = r1['later_final_time']

        r2 = self.simulated_annealing(est)
        sns.scatterplot(data=r2['npv'])
        plt.xlabel("Indice")
        plt.ylabel("Net Present Value")
        plt.savefig('simAnnealing_npv.png')

        prof = {
            'npv' : self.num_calculate_net_present_value,
            'cpm' : self.num_calculate_cpm,
            'cpm_f' : self.num_calculate_cpm_foward,
            'cpm_b' : self.num_calculate_cpm_backward,
            'valid_deadline' : self.num_validation_deadline_activity,
            'valid_path' : self.num_validation_path,
            'neighbor_schedule' : self.num_find_neighbour_schedule,
            'sa' : self.num_simulated_annealing
        }
        print(prof)

#        print(r2['p'])
#        sns.scatterplot(data=r2['p'])
#        plt.savefig('simAnnealing_p.png')


g = Gibs()
g.main()
