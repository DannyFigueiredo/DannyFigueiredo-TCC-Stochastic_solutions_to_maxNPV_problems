# Packges
import math
import numpy as np
# Project definition
import project as dtProject


class Npv:
    """
    This class calculates the net present value of a project.
    """

    def __init__(self):
        """
        Default settings.
        """
        self.project = dtProject.Project()
        self.rate = 0.01
        self.durantions = np.array([])
        self.cashflows = np.array([])
        self.sampleNPV = np.array([])

    # =============================== #
    #   Parameter Setting Functions   #
    # =============================== #
    def setRate(self, r):
        """
        Set a rate to a project
        :param r: rate
        """
        self.rate = r

    # ============================== #
    #   Parameter Return Functions   #
    # ============================== #
    def getRate(self):
        """
        Discount rate to net present value per unit of time
        :return: Rate
        """
        return self.rate

    # ========================= #
    #   Calculation Functions   #
    # ========================= #
    def alterDurationDictionaryToArray(self):
        """
        Convert dictionary to numpy array
        :return: duration array
        """
        times_list = list(self.project.getDurationDic().values())
        self.durantions = np.array(times_list)

    def alterCashflowDictionaryToArray(self):
        """
        Convert dictionary to numpy array
        :return: cash flow array
        """
        cashflow_list = list(self.project.getCashflowDic().values())
        self.cashflows = np.array(cashflow_list)


    def calculaNPV(self, schedules):
        """
        Calculates net present value for earlier start time scheduling
        :param schedules: early start schedule list in np.array in np.array type
        """
        self.alterCashflowDictionaryToArray()
        self.alterDurationDictionaryToArray()
        est_schedule = schedules + self.durantions
        self.sampleNPV = math.fsum(self.cashflows * math.exp(-self.rate * est_schedule))

        return self.sampleNPV