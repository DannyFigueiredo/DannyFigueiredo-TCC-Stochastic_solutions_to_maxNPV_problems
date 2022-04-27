import project as dtProject
import numpy as np
# g = dtProject.DataProject()

class Cpm:
    """
    CPM - Critical Path Method
    It provides a means of determining which jobs or activities, of the many that comprise a project,
    are "critical" in their effect on the total project time and the best way to schedule all activities
    in the project in order to meet a deadline with minimal cost.
    Characteristics of a project:
    1) collection of activities that, when completed, mark the end of the project
    2) activities can be started and stopped independently of each other, within a certain sequence
    3) the activities are ordered - that is, they must be performed in technological sequence
    """

    def __init__(self):
        self.project = dtProject.Project()
        self.sample_number = 100
        self.project_size = len(self.project.activity_successors)

        # Arrays with predefined sample numbers
        self.activity_duration_array = np.array([])
        # self.NPVsamples = np.zeros(self.sample_number)
        self.early_start_time = np.zeros(self.project_size)
        self.early_final_time = np.zeros(self.project_size)
        self.later_start_time = np.zeros(self.project_size)
        self.later_final_time = np.array([self.project.deadline]*self.project_size)

    # =============================== #
    #   Parameter Setting Functions   #
    # =============================== #
    def setSampleNumber(self, s):
        """
        Set a sample number to simulate the calculations
        :param s: sample number
        """
        self.sample_number = s

    def alterProjetSize(self):
        """
        Recalculates the project size (in case the project isn't the default)
        """
        self.project_size = len(self.project.activity_successors)

    def resetEarlyAndLaterList(self):
        """
        Resets the length of the early start times, early final start,
        later start time and later final start lists
        """
        l = len(self.project.activity_successors)
        self.early_start_time = np.zeros(l)
        self.early_final_time = np.zeros(l)
        self.later_start_time = np.zeros(l)
        self.later_final_time = np.zeros(l)

    def setActivityDurationArray(self):
        """
        Convert dictionary to numpy array
        :return: duration array
        """
        times_list = list(self.project.getDurationDic().values())
        self.activity_duration_array = np.array(times_list)

    # ============================== #
    #   Parameter Return Functions   #
    # ============================== #
    def getSampleNumber(self):
        """
        Returns the number of times calculations were simulated
        :return: sample number
        """
        return self.sample_number

    def getProjectSize(self):
        """
        Returns the number of activities that exist in a project
        :return: number of activities
        """
        return self.project_size

    def getNPVsamples(self):
        """
        Returns the net present value for each one of the samples
        """
        return self.NPVsamples

    def getEarlyStartTime(self):
        """
        Returns early start time list for each one of the activities
        """
        return self.early_start_time

    def getEarlyFinalTime(self):
        """
        Returns early final time list for each one of the activities
        """
        return self.early_final_time

    def getLaterStartTime(self):
        """
        Return later start time list for each one of the activities
        """
        return self.later_start_time

    def getLaterFinalTime(self):
        """
        Return later final time list for each one of the activities
        """
        return self.later_final_time

    # ========================= #
    #   Calculation Functions   #
    # ========================= #
    def calculateCpmEarlyStartTime(self, initial_activity, est, eft):
        """
        Calculate early start time list
        :param initial_activity: initial activity
        :param est: early start time list
        :param eft: early final time list
        :return: early start time calculated
        """
        self.setActivityDurationArray()
        eft[initial_activity] = est[initial_activity] + self.activity_duration_array[initial_activity]
        if self.project.activity_successors[initial_activity][1] != 0:
            for i in range(0, len(self.project.activity_successors[initial_activity])):
                if est[i] < eft[initial_activity]:
                    est[i] = eft[initial_activity]
                est = self.calculateCpmEarlyStartTime(i, est, eft)

        return est

    def calculateCpmLaterFinalTime(self, initial_activity, lst, lft):
        """
        Calcylate later final time list
        :param initial_activity: initial activity
        :param lst: later start time list
        :param lft: later final time list
        :return: later final time list calculated
        """
        self.setActivityDurationArray()
        lst[initial_activity] = lft[initial_activity] - self.activity_duration_array[initial_activity]
        if self.project.activity_predecessors[initial_activity][1] != 0:
            for i in range(0, len(self.project.activity_predecessors[initial_activity])):
                if lft[i] > lst[initial_activity]:
                    lft[i] = lst[initial_activity]
                lft = self.calculateCpmLaterFinalTime(i, lst, lft)

        return lft

    def calculateCpm(self):
        """
        Calculate the critical path
        :return: Returns an array of lists, where the first element is the earliest start list and the second element is the latest end list
        """
        self.early_start_time = self.calculateCpmEarlyStartTime(1, self.early_start_time, self.later_start_time)
        self.early_final_time = self.early_start_time + self.activity_duration_array
        self.later_final_time = self.calculateCpmLaterFinalTime(self.project_size, self.later_final_time, self.later_start_time)
        self.later_start_time = self.later_final_time - self.activity_duration_array
        resp = np.array(self.early_start_time, self.later_final_time)
        return resp

# c = Cpm()
# c.setActivityDurationArray()
# c.early_final_time[1] = c.early_start_time[1] + c.activity_duration_array[1]
# print(c.early_final_time[1])
# if c.project.activity_successors[1][1] != 0:
#     for i in range(0, len(c.project.activity_successors[1])):
#         if c.early_start_time[i] < c.early_final_time[1]:
#             c.early_start_time[i] = c.early_final_time[1]
#         c.early_start_time = c.calculateCpmEarlyStartTime(i, c.early_start_time, c.early_final_time)
# print(a)