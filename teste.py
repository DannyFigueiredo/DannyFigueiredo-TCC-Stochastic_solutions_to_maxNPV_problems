import numpy as np

activity_successors = {
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

activity_duration = {
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

activity_cash_flow = {
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

activity_predecessors = {
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

deadline = 44
rate = 0.01
sample_number = 500
project_size = len(activity_successors)

activity_successors_array = np.array(list(activity_successors.values()))
activity_duration_array = np.array(list(activity_duration.values()))
activity_cash_flow_array = np.array(list(activity_cash_flow.values()))
activity_predecessors_array = np.array(list(activity_predecessors.values()))

npv_samples = np.zeros(sample_number)
early_start_time = np.zeros(project_size)
early_final_time = np.zeros(project_size)
later_start_time = np.zeros(project_size)
later_final_time = np.zeros(project_size)



def calculate_cpm_foward(est, eft, initial_activity=0):
    """
    Calculate early start time list
    :param initial_activity: initial activity
    :param est: early start time list
    :param eft: early final time list
    :return: early start time calculated
    """
    eft[initial_activity] = est[initial_activity] + activity_duration_array[initial_activity][0]
    if activity_successors_array[initial_activity][0] != 0:
        for i in activity_successors_array[initial_activity]:
            if est[i - 1] < eft[initial_activity]:
                est[i - 1] = eft[initial_activity]
            est = calculate_cpm_foward(est, eft, i - 1)
    return est

def calculate_cpm_backward(lft, lst, size=14):
    """
    Calcylate later final time list
    :param last_activity: last activity
    :param lst: later start time list
    :param lft: later final time list
    :return: later final time list calculated
    """
    lst[size] = lft[size] - activity_duration_array[size][0]
    if activity_predecessors_array[size][0] != 0:
        for i in activity_predecessors_array[size]:
            if lft[i] > lst[size]:
                lft[i] = lst[size]
            lft = calculate_cpm_backward(lft, lst, i - 1)
    return lft


# cpmDriver = function(i=1, est, eft)
# {
#     est = cpmf(1, est, lst)
# eft = est + d
# lft = rep(dMax, times=n)
# lft = cpmb(n, lft, lst)
# r = list(est=est, lft=lft)
# r
# }
def calculate_cpm(est, eft, initial_activity = 1):
    """
    Calculate the critical path
    :param est: early start time list
    :param eft: early final time list
    :param initial_activity: initial activity
    :return: Returns an array of lists, where the first element is the earliest start list and the second element is the latest end list
    """
    self.early_start_time = self.calculate_cpm_foward(est, eft, 1)
    self.early_final_time = sum(self.early_start_time, self.activity_duration_array)
    self.later_final_time = self.calculate_cpm_backward(self.later_final_time, self.later_start_time, self.project_size)



size = 13
print(later_start_time)
print('later_start_time[size] = later_final_time[size] - activity_duration_array[size][0]')
later_start_time[size] = later_final_time[size] - activity_duration_array[size][0]
print('later_start_time[size] = ',later_start_time[size])
print('    if activity_predecessors_array[size][0] != 0:',
      activity_predecessors_array[size][0] != 0)
if activity_predecessors_array[size][0] != 0:
    print('         for i in activity_predecessors_array[size]:',
          activity_predecessors_array[size])
    for i in activity_predecessors_array[size]:
        print('         => ', i)
        print('             if lft[i] > lst[size]:',
              later_final_time[i] > later_start_time[size])
        if later_final_time[i] > later_start_time[size]:
            later_final_time[i] = later_start_time[size]
            print('later_final_time[i] => ', later_final_time[i])

        size = i
        print(later_start_time)
        print('later_start_time[size] = later_final_time[size] - activity_duration_array[size][0]')
        later_start_time[size] = later_final_time[size] - activity_duration_array[size][0]
        print('later_start_time[size] = ',later_start_time[size])
        print('    if activity_predecessors_array[size][0] != 0:',
              activity_predecessors_array[size][0] != 0)
        if activity_predecessors_array[size][0] != 0:
            print('         for i in activity_predecessors_array[size]:',
                  activity_predecessors_array[size])
            for j in activity_predecessors_array[size]:
                print('         => ', i)
                print('             if lft[i] > lst[size]:',
                      later_final_time[j] > later_start_time[size])
                if later_final_time[j] > later_start_time[size]:
                    later_final_time[j] = later_start_time[size]
                    print('later_final_time[i] => ', later_final_time[j])

##    lft = calculate_cpm_backward(lft, lst, i - 1)


                
for size in range(0, 14):
    print(later_final_time[size] - activity_duration_array[size][0])
