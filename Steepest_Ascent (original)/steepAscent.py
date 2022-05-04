import datetime as dt
import numpy as np
from random import sample, seed
import seaborn as sns
import matplotlib.pyplot as plt


class HillClimbing:
    """
    This class represents the graph of a project through dictionaries of:
    nodes and their successors, duration of each activity and the cash flow
    of each activity.
    """

    def __init__(self):
        """
        Default settings and libraries.
        """
        # ===================================================================== #
        #                               Keep in mind!                           #
        # ===================================================================== #
        # The numbering of activities starts at 1, but the indices start at 0.  #
        # So, to reference activity 1, we index to 0.                           #
        # ===================================================================== #

        # Successors of each activity
        self.activity_successors = np.array([(2, 3, 4), (9), (5, 6, 7), (8), (10), (12), (8, 11), (13), (14), (12), (12), (13), (14), (0)])

        # The time corresponds to each activity
        self.activity_duration = np.array([(0), (6), (5), (3), (1), (6), (2), (1), (4), (3), (2), (3), (5), (0)])

        # Corresponds to the cash flow of each project activity
        self.activity_cash_flow = np.array([(0), (-140), (318), (312), (-329), (153), (193), (361), (24), (33), (387), (-386), (171), (0)])

        # Activity precedence dictionary
        self.activity_predecessors = np.array([(0), (1), (1), (1), (3), (3), (3), (4, 7), (2), (5), (7), (6, 10, 11), (8, 12), (9, 13)])

        # Maximum time for project delivery
        self.deadline = 44

        # Cash value devaluation rate
        self.rate = 0.01

        # Number of samples
        self.sample_number = 500

        # Number of activities listed for the project
        self.project_size = len(self.activity_successors)

        # ===================================================================== #
        #                       Auxiliary arrays and variables:                 #
        # ===================================================================== #

        # 1)    To save sampled NPVs
        self.npv_samples = np.zeros(self.sample_number)

        # 2)    To save the critical paths obtained by the CPM
        self.early_start_time = np.zeros(self.project_size)
        self.early_final_time = np.zeros(self.project_size)
        self.later_start_time = np.zeros(self.project_size)
        self.later_final_time = np.array([self.deadline] * self.project_size)

        # 3)    To save the number of times a certain function is called in the
        #       execution of the algorithm
        self.num_calculate_net_present_value = 0
        self.num_calculate_cpm_foward = 0
        self.num_calculate_cpm_backward = 0
        self.num_calculate_cpm = 0
        self.num_validation_path = 0
        self.num_validation_schedule = 0
        self.num_find_max_gradient_neighbour = 0
        self.num_steepest_ascent = 0

        # 4)    To save the total time spent executing each function
        self.timer_calculate_net_present_value = dt.timedelta(0)
        self.timer_calculate_cpm_foward = dt.timedelta(0)
        self.timer_calculate_cpm_backward = dt.timedelta(0)
        self.timer_calculate_cpm = dt.timedelta(0)
        self.timer_validation_path = dt.timedelta(0)
        self.timer_validation_schedule = dt.timedelta(0)
        self.timer_find_max_gradient_neighbour = dt.timedelta(0)
        self.timer_steepest_ascent = dt.timedelta(0)

    # ========================================================================= #
    #                           Time manipulation function                      #
    # ========================================================================= #
    def time_variation(self, timer, interval):
        resp = timer + dt.timedelta(seconds=interval.seconds,
                                    microseconds=interval.microseconds)
        return resp

    # ========================================================================= #
    #                             Critical Path Method                          #
    # ========================================================================= #
    # It provides a means of determining which jobs or activities, of the many  #
    # that comprise a project, are "critical" in their effect on the total      #
    # project time and the best way to schedule all activities in the project   #
    # in order to meet a deadline with minimal cost.                            #
    #                                                                           #
    # Characteristics of a project:                                             #
    # 1)    Collection of activities that, when completed, mark the end of      #
    #       the project                                                         #
    # 2)    Activities can be started and stopped independently of each other,  #
    #       within a certain sequence                                           #
    # 3)    The activities are ordered - that is, they must be performed in     #
    #       technological sequence                                              #
    # ========================================================================= #
    def calculate_cpm_foward(self, est, eft, initial_activity=1):
        """
        Calculate early start time list
        :param initial_activity: initial activity
        :param est: early start time list
        :param eft: early final time list
        :return: early start time calculated
        """
        # variables for algorithm profile description
        self.num_calculate_cpm_foward += 1
        start = dt.datetime.now()

        eft[initial_activity - 1] = est[initial_activity - 1] + self.activity_duration[initial_activity - 1]
        if isinstance(self.activity_successors[initial_activity - 1], tuple):
            if self.activity_successors[initial_activity - 1][0] != 0:
                for i in self.activity_successors[initial_activity - 1]:
                    if est[i - 1] < eft[initial_activity - 1]:
                        est[i - 1] = eft[initial_activity - 1]
                    est = self.calculate_cpm_foward(est, eft, i)
        else:
            if self.activity_successors[initial_activity - 1] != 0:
                if est[self.activity_successors[initial_activity - 1] - 1] < eft[initial_activity - 1]:
                    est[self.activity_successors[initial_activity - 1] - 1] = eft[initial_activity - 1]
                est = self.calculate_cpm_foward(est, eft, self.activity_successors[initial_activity - 1])

        # variables for algorithm profile description
        end = dt.datetime.now()
        interval = end - start
        self.timer_calculate_cpm_foward = self.time_variation(self.timer_calculate_cpm_foward, interval)

        return est

    def calculate_cpm_backward(self, lft, lst, size=13):
        """
        Calcylate later final time list
        :param size: last activity -1
        :param lst: later start time list
        :param lft: later final time list
        :return: later final time list calculated
        """
        # variables for algorithm profile description
        self.num_calculate_cpm_backward += 1
        start = dt.datetime.now()

        lst[size] = lft[size] - self.activity_duration[size]
        if isinstance(self.activity_predecessors[size], tuple):
            if self.activity_predecessors[size][0] != 0:
                for i in self.activity_predecessors[size]:
                    if lft[i - 1] > lst[size]:
                        lft[i - 1] = lst[size]
                    # variables for algorithm profile description
                    end = dt.datetime.now()
                    interval = end - start
                    self.timer_calculate_cpm_backward = self.time_variation(self.timer_calculate_cpm_backward, interval)
                    lft = self.calculate_cpm_backward(lft, lst, i - 1)
        else:
            if self.activity_predecessors[size] != 0:
                if lft[self.activity_predecessors[size] - 1] > lst[size]:
                    lft[self.activity_predecessors[size] - 1] = lst[size]
                # variables for algorithm profile description
                end = dt.datetime.now()
                interval = end - start
                self.timer_calculate_cpm_backward = self.time_variation(self.timer_calculate_cpm_backward, interval)
                lft = self.calculate_cpm_backward(lft, lst, self.activity_predecessors[size] - 1)

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
        # variables for algorithm profile description
        self.num_calculate_cpm += 1
        start = dt.datetime.now()

        self.early_start_time = self.calculate_cpm_foward(est, eft, initial_activity)
        self.early_final_time = self.early_start_time + self.activity_duration
        self.later_final_time = self.calculate_cpm_backward(self.later_final_time, self.later_start_time,
                                                            self.project_size - 1)
        self.later_start_time = self.later_final_time - self.activity_duration

        # variables for algorithm profile description
        end = dt.datetime.now()
        interval = end - start
        self.timer_calculate_cpm = self.time_variation(self.timer_calculate_cpm, interval)

        resp = {'early_start_time': self.early_start_time,
                'early_final_time': self.early_final_time,
                'later_start_time': self.later_start_time,
                'later_final_time': self.later_final_time}

        return resp

    # ========================================================================= #
    #                               Net Present Value                           #
    # ========================================================================= #
    def calculate_net_present_value(self, schedules):
        """
        Calculates net present value for earlier start time scheduling
        :param schedules: early start schedule list in np.array type
        """
        # variables for algorithm profile description
        self.num_calculate_net_present_value += 1
        start = dt.datetime.now()

        npv = sum(self.activity_cash_flow * np.exp(-self.rate * (schedules + self.activity_duration)))

        # variables for algorithm profile description
        end = dt.datetime.now()
        interval = end - start
        self.timer_calculate_net_present_value = self.time_variation(self.timer_calculate_net_present_value, interval)

        return npv

    # ========================================================================= #
    #                           Validation Functions                            #
    # ========================================================================= #
    def validation_path(self, act_index, est):
        """
        Path validation following design constraints
        :param act_index: activity index
        :param est: early start time list
        :return: early start time list
        """
        # variables for algorithm profile description
        self.num_validation_path += 1
        start = dt.datetime.now()

        if isinstance(self.activity_successors[act_index], tuple):
            if self.activity_successors[act_index][0] != 0:
                for i in self.activity_successors[act_index]:
                    if est[i - 1] < est[act_index] + self.activity_duration[act_index]:
                        est[i - 1] = est[act_index] + self.activity_duration[act_index]
                    est = self.validation_path(i - 1, est)
        else:
            if self.activity_successors[act_index] != 0:
                if est[self.activity_successors[act_index] - 1] < est[act_index] + self.activity_duration[act_index]:
                    est[self.activity_successors[act_index] - 1] = est[act_index] + self.activity_duration[act_index]
                est = self.validation_path(self.activity_successors[act_index] - 1, est)

        # variables for algorithm profile description
        end = dt.datetime.now()
        interval = end - start
        self.timer_validation_path = self.time_variation(self.timer_validation_path, interval)

        return est

    def validation_schedule(self, schedule):
        """
        Path validation following time constraints
        :param schedule: schedule array
        :return: boolean for path validation
        """
        # variables for algorithm profile description
        self.num_validation_schedule += 1
        start = dt.datetime.now()

        value = True
        for i in range(1, self.project_size):
            pr = self.activity_predecessors[i]
            if isinstance(self.activity_predecessors[i], tuple):
                # print(' - ', self.activity_predecessors[i])
                for p in pr:
                    # print(' - ', p, pr)
                    # print(schedule)
                    if schedule[0][i] < schedule[0][p] + self.activity_duration[p]:
                        value = False
                        break
            else:
                # print(schedule, pr)
                if schedule[0][i] < schedule[0][pr] + self.activity_duration[pr]:
                    value = False
                    break

        # variables for algorithm profile description
        end = dt.datetime.now()
        interval = end - start
        self.timer_validation_schedule = self.time_variation(self.timer_validation_schedule, interval)

        return value

    # ========================================================================= #
    #                           Auxiliary Function                              #
    # ========================================================================= #
    def find_max_gradient_neighbour(self, schedule):
        """
        Analyzes the schedule with the highest possible NPV in the surroundings
        :param schedule: schedule array
        :return: maximized schedule
        """
        # variables for algorithm profile description
        self.num_find_max_gradient_neighbour += 1
        start = dt.datetime.now()

        # print('agendamento : ', schedule, ' - ', id(schedule))
        max_npv = self.calculate_net_present_value(schedule)
        max_sched = [schedule[0:]]
        # print('agendamento maximo : ', max_sched, ' - ', id(max_sched))

        for node in range(1, self.project_size, 1):
            sched_t = [schedule[0:]]
            max = self.later_start_time[node]
            # print('agendamento t : ', sched_t, ' - ', id(sched_t))

            if sched_t[0][node] < max:
                sched_t[0][node] = sched_t[0][node] + 1

            sched_t = self.validation_path(node, sched_t[0])
            val = self.calculate_net_present_value(sched_t[0])
            # print('max-NPV : ', max_npv, 'candidatoo a max-NPV : ', val)

            if val > max_npv:
                max_sched = [sched_t[0:]]
                max_npv = val
                # print('agendamento maximo pós validação: ', max_sched, ' - ', id(max_sched))
                # print('max-NPV : ', max_npv)

        # variables for algorithm profile description
        end = dt.datetime.now()
        interval = end - start
        self.timer_find_max_gradient_neighbour = self.time_variation(self.timer_find_max_gradient_neighbour, interval)

        return max_sched[0]

    # ========================================================================= #
    #                               Steepest Ascent                             #
    # ========================================================================= #
    def steepest_ascent(self, schedule):
        """

        :param schedule:
        :return:
        """
        # variables for algorithm profile description
        self.num_steepest_ascent += 1
        start = dt.datetime.now()

        scheduling_sequence_matrix = [schedule]

        t = 0
        while True:
            t += 1
            #   1)  Find neighbour schedule
            nsched = self.find_max_gradient_neighbour(schedule)
            # print('nsched : ', nsched)

            if self.validation_schedule(nsched):
                #   2)  Evaluates NPVs
                npv = self.calculate_net_present_value(schedule)
                self.npv_samples[t-1] = npv
                nnpv = self.calculate_net_present_value(nsched)
                print('NPV : ', npv, ' - NNPV : ', nnpv)
                delta = nnpv - npv

                #   3)  Check for acceptance
                if delta > 0:
                    schedule = nsched
                    scheduling_sequence_matrix.append(nsched)
                    self.npv_samples[t] = nnpv
                else:
                    scheduling_sequence_matrix = scheduling_sequence_matrix[0:t+1]
                    self.npv_samples = self.npv_samples[0:t+1]
                    break
            else:
                print('Agendamento invalido : ', nsched)

        resp = {'npv': self.npv_samples,
                'seq': scheduling_sequence_matrix}

        # variables for algorithm profile description
        end = dt.datetime.now()
        interval = end - start
        self.timer_steepest_ascent = self.time_variation(self.timer_steepest_ascent, interval)

        return resp

    # ========================================================================= #
    #                                     MAIN                                  #
    # ========================================================================= #
    def main(self):
        start = dt.datetime.now()

        #   1) Calculate scheduling data
        sch = self.calculate_cpm(self.early_start_time, self.early_final_time)
        for i in sch.keys():
            print(i + ' : ', sch[i], '  --> NPV = ', self.calculate_net_present_value(sch[i]))
        print("\n")

        #   2) Call the Simulated Annealing data
        # self.find_max_gradient_neighbour(sch['early_start_time'])

        sim = self.steepest_ascent(sch['early_start_time'])
        # print('NPV \t\t\t', sim['npv'][0:10])
        # print('Seq Schedule \t', sim['seq'][0:10])
        # print('Probability p \t', sim['p'][0:10])
        # print("\n")

        #   3) Generate graphs with the results obtained
        # sns.set()
        # sns.scatterplot(data=sim['npv'][0:])
        # plt.xlabel('Indice')
        # plt.ylabel('NPV')
        # fileName = 'steepAscent_NPV.png'
        # plt.savefig(fileName)
        # plt.close()

        #   4) Generate a profile for the functions (time and number of times they were called)
        print("Function name \t\t Number calls \t Time spent")
        print("------------------ \t ------------ \t ----------")
        print("NPV \t\t\t\t", self.num_calculate_net_present_value - 4, "\t\t\t", self.timer_calculate_net_present_value)
        print("CPM foward \t\t\t", self.num_calculate_cpm_foward, "\t\t\t", self.timer_calculate_cpm_foward)
        print("CPM backword \t\t", self.num_calculate_cpm_backward, "\t\t\t", self.timer_calculate_cpm_backward)
        print("CPM \t\t\t\t", self.num_calculate_cpm, "\t\t\t\t", self.timer_calculate_cpm)
        print("Validation path\t\t", self.num_validation_path, "\t\t\t", self.timer_validation_path)
        print("Validation schedule\t", self.num_validation_schedule, "\t\t\t\t", self.timer_validation_schedule)
        print("Neighbour Schedule \t", self.num_find_max_gradient_neighbour, "\t\t\t\t", self.timer_find_max_gradient_neighbour)
        print("Steepest Ascent \t", self.num_steepest_ascent, "\t\t\t\t", self.timer_steepest_ascent)

        end = dt.datetime.now()
        totalRunTime = end - start
        print("\nTotal Run Time .......... ", totalRunTime)
        print("\nMax-NPV ................. ", max(self.npv_samples))


h = HillClimbing()
h.main()