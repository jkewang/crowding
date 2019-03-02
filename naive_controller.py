

class NaiveCon(object):
    def __init__(self):
        self.NaiveCon = 1

    def gen_action(self,state,rawOcc):
        """
        :param state:
        OccMap,VehicleState,RoadState
        classify:
        BF | BFL | BFR | BHL | BHR
        :return: action
        """
        action = 0

        VehicleState,RoadState = state[280],[state[283],state[284]]
        kinds = self.classify(rawOcc)
        isMaxSpeed = 0
        isLeft = 0
        isRight = 0

        if VehicleState > 0.8:
            isMaxSpeed = True
        else:
            isMaxSpeed = False

        isLeft = RoadState[0]
        isRight = RoadState[1]
        #print("isleft:",isLeft)
        #print("isRight:",isRight)
        if kinds[0] == False:
            if isMaxSpeed:
                action = 4
            else:
                action = 0
        elif kinds[1] == False and kinds[3] == False:
            if not isLeft:
                action = 2
            elif kinds[2] == False and kinds[4] == False:
                if not isRight:
                    action = 3
                else:
                    action = 1
            else:
                action = 1
        elif kinds[2] == False and kinds[4] == False:
            if not isRight:
                action = 3
            else:
                action = 1
        else:
            action = 1

        #print(kinds)
        #print("myaction:",action)
        return action

    def classify(self,rawOcc):
        """
        :param OccMap:
        kind:
        BF | BFL | BFR | BHL | BHR
        :return: what kind of state the ego vehicle is facing
        """
        kinds = [0,0,0,0,0]

        BFL = 0
        BF = 0
        BFR = 0
        BHL = 0
        BHR = 0

        for ii in range(15,30):
            for jj in range(0,2):
                BFL += rawOcc[ii][jj]
            for jj in range(2,5):
                BF += rawOcc[ii][jj]
            for jj in range(5,7):
                BFR += rawOcc[ii][jj]
        for ii in range(30,36):
            for jj in range(0,2):
                BHL += rawOcc[ii][jj]
            for jj in range(5,7):
                BHR += rawOcc[ii][jj]

        kinds = [(BF>0),(BFL>0),(BFR>0),(BHL>0),(BHR>0)]

        return kinds