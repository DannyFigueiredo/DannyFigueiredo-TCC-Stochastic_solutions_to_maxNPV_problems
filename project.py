class Project:
    """
    This class represents the graph of a project through dictionaries of: nodes and their successors,
    duration of each activity and the cash flow of each activity.
    """

    def __init__(self):
        """
        Default settings.
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
            14: [None]
        }

        # The time corresponds to each activity
        # key = activity and value = duration time
        self.activity_duration = {
            1: 0,
            2: 6,
            3: 5,
            4: 3,
            5: 1,
            6: 6,
            7: 2,
            8: 1,
            9: 4,
            10: 3,
            11: 2,
            12: 3,
            13: 5,
            14: 0
        }

        # Corresponds to the cash flow of each project activity
        self.activity_cashflow = {
            1: 0,
            2: -140,
            3: 318,
            4: 312,
            5: -329,
            6: 153,
            7: 193,
            8: 361,
            9: 24,
            10: 33,
            11: 387,
            12: -386,
            13: 171,
            14: 0
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

    # =============================== #
    #   Parameter Setting Functions   #
    # =============================== #
    def setNewActivitySuccessors(self, key, value):
        """
        Saves a new activity successors node in the dictionary
        :param key: activity number
        :param value: list of possible further activities
        """
        self.activity_successors[key] = value

    def setNewActivityDuration(self, key, value):
        """
        Saves a new activity duration node in the dictionary
        :param key: activity number
        :param value: activity duration time
        """
        self.activity_duration[key] = value

    def setNewActivityCashflow(self, key, value):
        """
        Saves a new cash flow node in the dictionary
        :param key: activity number
        :param value: activity cash flow
        """
        self.activity_cashflow[key] = value

    def setNewActivityPredecessors(self, key, value):
        """
        Saves a new activity precedence node in the dictionary
        :param key: activity
        :param value: list of possible predecessor activities
        """
        self.activity_predecessors[key] = value

    def setNewProject(self, s_dic, p_dic, d_dic, c_dic):
        """
        Set a new project through dictionaries
        :param s_dic: successors activities dictionary
        :param p_dic: predecessor per activity dictionary
        :param d_dic: activity duration dictionary
        :param c_dic: activity cash flow dictionary
        :return:
        """
        self.activity_successors = s_dic
        self.activity_predecessors = p_dic
        self.activity_duration = d_dic
        self.activity_cashflow = c_dic

    def setDeadline(self, d):
        """
        Set a deadline to a project
        :param d: Maximum time for project delivery
        """
        self.deadline = d

    # ============================== #
    #   Parameter Return Functions   #
    # ============================== #
    def getSuccessorsDic(self):
        """
        Returns activities dictionary
        """
        return self.activity_successors

    def getPredecessorDic(self):
        """
        Returns predecessors activities dictionary
        """
        return self.activity_predecessors

    def getDurationDic(self):
        """
        Returns activity duration dictionary by activity
        """
        return self.activity_duration

    def getCashflowDic(self):
        """
        Returns cash flow dictionary by activity
        """
        return self.activity_cashflow

    def getActivitySucessors(self, key):
        """
        Returns list of successor activities
        """
        return self.activity_successors[key]

    def getActivityDuration(self, key):
        """
        Returns activity duration
        """
        return self.activity_duration[key]

    def getActivityCashflow(self, key):
        """
        Returns activity cash flow
        """
        return self.activity_cashflow[key]

    def getActivityPredecessors(self, key):
        """
        Returns a list of activities that precede immediately the searched activity using direct dictionary
        :param key: activity you are looking for
        """
        return self.activity_predecessors[key]

    def getDeadline(self):
        """
        Returns maximum time for project delivery
        :return: deadline
        """
        return self.deadline

    def getImmediatePredecessors(self, activity_number):
        """
        Returns a list of activities that precede immediately the searched activity
        :param activity_number: activity you are looking for
        """
        pred = []
        for i in self.activity_successors:
            # print( self.activity_successors[i] )
            for j in self.activity_successors[i]:
                if activity_number == j:
                    pred.append(i)

        pred = list(set(pred))
        return pred

    def getAllPredecessors(self, activity_number):
        """
        Returns a list of all activities that precede the searched activity
        :param activity_number: activity you are looking for
        """
        pred = []  # list of all possible predecessors of the researched activity
        predecessor_auxiliary_activities = []  # auxiliary list of predecessor activities
        current_activity = max(self.activity_successors.keys())  # current activity starting with final activity

        while True:
            # From the final activity, a task backtracks until it finds the researched activity
            if current_activity > activity_number:
                current_activity -= 1
                continue

            # Adds current activity to predecessor list
            pred.append(current_activity)
            # Adds the list of predecessor activities, the predecessor activities of the current node
            predecessor_auxiliary_activities = predecessor_auxiliary_activities + self.getActivityPredecessors(current_activity)
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

        return pred