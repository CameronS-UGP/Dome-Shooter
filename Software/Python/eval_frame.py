# Class and functions for evaluating detection methods
# Also includes function for calculating distance between two points

class EvalFrame:

    def __init__(self, frame_number, annotations):
        self.frame_number = frame_number
        self.annotations = annotations

    def getFNum(self):
        return self.frame_number

    # At the moment only records against GT
    def evaluateFrame(self, **kwargs):  # yellow_dot=yellow_dot_bboxes, reticle_t=bFound, reticle_coords=(reticle_x, reticle_y)
        # print(f"Yellow Dotty: {kwargs['yellow_dot']}\nReticle: {kwargs['reticle_t']}; {kwargs['reticle_coords']}")

        f_yellow_dots = kwargs['yellow_dot']
        f_reticle_truth = kwargs['reticle_t']
        f_reticle_coords = kwargs['reticle_coords']

        s_yellow_dots = []
        s_reticle_truth = False
        s_reticle_coords = (0,0)

        # Loading object annotations into comparable format of vars
        for annotation in self.annotations:
            assert isinstance(annotation, dict)

            if annotation['label_id'] == 0:  # id 0 is a yellow dot
                s_yellow_dots.append(annotation.get('points'))
            elif annotation['label_id'] == 1: # id 1 is a reticle
                s_reticle_truth = True
                s_reticle_coords = (annotation.get('points')[0], annotation.get('points')[1])
            else:
                raise ValueError("Unexpected label_id found.")

        yellow_gt_score, reticle_gt_score = groundTruth(len(f_yellow_dots), len(s_yellow_dots), f_reticle_truth, s_reticle_truth)

        return yellow_gt_score, reticle_gt_score


def groundTruth(f_yellow_dot_count, s_yellow_dot_count, f_reticle_truth, s_reticle_truth):
    yellow_gt_score = False
    reticle_gt_score = False

    if f_yellow_dot_count == s_yellow_dot_count:
        yellow_gt_score = True

    if f_reticle_truth == s_reticle_truth:
        reticle_gt_score = True

    return yellow_gt_score, reticle_gt_score

# Calculates percentage of True booleans in a list of bools
def percentageFromBools(booList):
    truth_count = booList.count(True)

    frac = truth_count / len(booList)
    perc = frac * 100
    return round(perc,2)

# Calculates distance between two given coords
def calculateDistance(x_A, y_A, x_B, y_B):
    pass
