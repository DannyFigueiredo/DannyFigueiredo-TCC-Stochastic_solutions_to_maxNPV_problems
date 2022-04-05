from datetime import datetime as dt
import numpy as np
from random import sample, seed
import seaborn as sns
import matplotlib.pyplot as plt


class HillClimbing:
    """
    This class represents the graph of a project through dictionaries of: nodes and their successors,
    duration of each activity and the cash flow of each activity.
    """

    def __init__(self):
        """
        Default settings and libraries.
        """
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
        self.num_find_neighbour_schedule = 0
        self.num_steepest_ascent = 0

        # Function timer
        #self.timer_calculate_net_present_value = datetime(2022, 1, 1, 0, 0, 0, 0)
        #self.timer_calculate_cpm_foward = datetime(2022, 1, 1, 0, 0, 0, 1)
        #self.timer_calculate_cpm_backward = datetime(2022, 1, 1, 0, 0, 0, 0)
        #self.timer_calculate_cpm = datetime(2022, 1, 1, 0, 0, 0, 0)
        #self.timer_validation_path = datetime(2022, 1, 1, 0, 0, 0, 0)
        #self.timer_find_neighbour_schedule = datetime(2022, 1, 1, 0, 0, 0, 0)
        #self.timer_simulated_annealing = datetime(2022, 1, 1, 0, 0, 0, 0)

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
    def calculate_cpm_foward(self, est, eft, initial_activity=1):
        """
        Calculate early start time list
        :param initial_activity: initial activity
        :param est: early start time list
        :param eft: early final time list
        :return: early start time calculated
        """
        # counter to know how many times the function was called
        self.num_calculate_cpm_foward += 1

        eft[initial_activity-1] = est[initial_activity-1] + self.activity_duration_array[initial_activity-1]

        if self.activity_successors_array[initial_activity-1][0] != 0:
            for i in self.activity_successors_array[initial_activity-1]:
                if est[i-1] < eft[initial_activity-1]:
                    est[i-1] = eft[initial_activity-1]
                est = self.calculate_cpm_foward(est, eft, i)

        return est

    def calculate_cpm_backward(self, lft, lst, size=13):
        """
        Calcylate later final time list
        :param size: last activity -1
        :param lst: later start time list
        :param lft: later final time list
        :return: later final time list calculated
        """
        # counter to know how many times the function was called
        self.num_calculate_cpm_backward += 1

        lst[size] = lft[size] - self.activity_duration_array[size][0]
        if self.activity_predecessors_array[size][0] != 0:
            for i in self.activity_predecessors_array[size]:
                if lft[i - 1] > lst[size]:
                    lft[i - 1] = lst[size]
                lft = self.calculate_cpm_backward(lft, lst, i - 1)

        return lft

    def calculate_cpm(self, est, eft, initial_activity = 0):
        """
        Calculate the critical path
        :param est: early start time list
        :param eft: early final time list
        :param initial_activity: initial activity
        :return: Returns an array of lists, where the first element is the earliest start list and the second element is
        the latest end list
        """
        # counter to know how many times the function was called
        self.num_calculate_cpm += 1

        self.early_start_time = self.calculate_cpm_foward(est, eft, initial_activity)
        for i in range(0, self.project_size - 1):
            self.early_final_time[i] = self.early_start_time[i] + self.activity_duration_array[i][0]
        self.later_final_time = self.calculate_cpm_backward(self.later_final_time, self.later_start_time,
                                                            self.project_size - 1)
        for i in range(0, self.project_size):
            self.later_start_time[i] = self.later_final_time[i] - self.activity_duration_array[i][0]
        resp = {'early_start_time': self.early_start_time,
                'early_final_time': self.early_final_time,
                'later_start_time': self.later_start_time,
                'later_final_time': self.later_final_time}

        return resp

    # ===================== #
    #   Net Present Value   #
    # ===================== #
    def calculate_net_present_value(self, schedules):
        """
        Calculates net present value for earlier start time scheduling
        :param schedules: early start schedule list in np.array type
        """
        # counter to know how many times the function was called
        self.num_calculate_net_present_value += 1

        npv_activity = np.zeros(self.project_size)
        for i in range(0, self.project_size):
            npv = self.activity_cash_flow_array[i] * np.exp(-self.rate * (schedules[i] + self.activity_duration_array[i]))
            npv_activity[i] = npv

        a = sum(npv_activity)

        return a

    # ======================== #
    #   Validation Functions   #
    # ======================== #
    def validation_path(self, act_index, est):
        """

        :param act_index: activity index
        :param est: early start time list
        :return: early start time list
        """
        self.num_validation_path += 1

        if self.activity_successors_array[act_index][0] != 0:
            for i in self.activity_successors_array[act_index]:
                if est[i - 1] < est[act_index] + self.activity_duration_array[act_index]:
                    est[i - 1] = est[act_index] + self.activity_duration_array[act_index]
                est = self.validation_path(i-1, est)

        return est

    def validation_schedule(self, schedule):
        val = True
        for i in range(1, self.project_size, 1):
            pr = self.activity_predecessors_array[i]
            for p in pr:
                if schedule[i] < schedule[p-1] + self.activity_duration_array[p-1]:
                    val = False
                    break

        return val

    # ======================= #
    #   Auxiliary Functions   #
    # ======================= #
    def find_max_gradient_neighbour(self, schedule):
        self.num_find_neighbour_schedule += 1

        max_npv = self.calculate_net_present_value(schedule)
        max_sched = schedule

        for t in range(1, self.project_size-1, 1):
            sched_t = schedule
            max = self.later_start_time[t]

            if sched_t[t] < max:
                sched_t[t] = sched_t[t] + 1

            sched_t = self.validation_path(t, sched_t)
            val = self.calculate_net_present_value(sched_t)

            if val > max_npv:
                max_sched = sched_t
                max_npv = val

        return max_sched

    # ============================ #
    #   Steepest Ascent function   #
    # ============================ #
    def steepest_ascent(self, schedule):
        self.num_steepest_ascent += 1
        
        scheduling_sequence_matrix = [0] * self.sample_number
        scheduling_sequence_matrix[0] = schedule

        t = 0
        while True:
            t += 1
            nsched = self.find_max_gradient_neighbour(schedule)

            if self.validation_schedule(nsched):
                npv = self.calculate_net_present_value(schedule)
                self.npv_samples = npv
                nnpv = self.calculate_net_present_value(nsched)
                delta = nnpv - npv

                if delta > 0:
                    sched = nsched
                    scheduling_sequence_matrix[t] = nsched
                    self.npv_samples[t] = nnpv
                else:
                    scheduling_sequence_matrix = scheduling_sequence_matrix[0:t]
                    self.npv_samples = self.npv_samples[0:t]
                    break
            else:
                print('Agendamento invalido : ', nsched)

            resp = {'npv': self.npv_samples,
                    'seq': scheduling_sequence_matrix}

            return resp

    # ================= #
    #   Main function   #
    # ================= #
    def main(self):
        inicio = dt.now()

        r1 = self.calculate_cpm(self.early_start_time, self.early_final_time, 1)
        for i in r1.keys():
            print(i + ' : ', r1[i])

        r2 = self.steepest_ascent(r1['early_start_time'])
        #print('npv : ', r2['npv'])
        #for i in r2.keys():
        #    print(i + ' : ', r2[i])

        fim = dt.now()
        tempo_total = fim - inicio
        print('Tempo total : ', tempo_total)

        #sns.scatterplot(data=r2['p'])
        #plt.xlabel("Indice")
        #plt.ylabel("p")
        #plt.savefig('simAnnealing_condicaoParada_p.png')

        #sns.scatterplot(data=r2['npv'])
        #plt.xlabel("Indice")
        #plt.ylabel("Net Present Value")
        #plt.savefig('simAnnealing_condicaoParada_npv.png')

        #print('---- Quantidade de chamadas por função ----')
        #print('npv : ', self.num_calculate_net_present_value)
        #print('cpm : ', self.num_calculate_cpm)
        #print('cpm_f : ', self.num_calculate_cpm_foward)
        #print('cpm_b : ', self.num_calculate_cpm_backward)
        #print('valid_path : ', self.num_validation_path)
        #print('neighbor_schedule : ', self.num_find_neighbour_schedule)
        #print('sa : ', self.num_steepest_ascent)

        #print('---- Tempo gasto por função ----')
        #print('npv : ', self.timer_calculate_net_present_value)
        #print('cpm : ', self.timer_calculate_cpm)
        #print('cpm_f : ', self.timer_calculate_cpm_foward)
        #print('cpm_b : ', self.timer_calculate_cpm_backward)
        #print('valid_path : ', self.timer_validation_path)
        #print('neighbor_schedule : ', self.timer_find_neighbour_schedule)
        #print('sa : ', self.timer_simulated_annealing)


h = HillClimbing()
h.main()