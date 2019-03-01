import os,sys
import change_trip

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'],'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable SUMO_HOME")

config_path = "/home/jkwang/learn_sumo/straight/straight.sumo.cfg"
sumoBinary = "/usr/bin/sumo"
sumoguiBinary = "/usr/bin/sumo-gui"
sumoCmd = [sumoguiBinary,"-c",config_path,"--collision.action","remove","--start","--no-step-log","--no-warnings","--no-duration-log"]

import traci
import traci.constants as tc
import math
import numpy as np

DEAD_LINE = 300

class TrafficEnv(object):

    def __init__(self):
        traci.start(sumoCmd)

        #Env --lanechange.duration
        self.step_num = 0
        self.AgentId = "agent"
        self.VehicleIds = []
        self.TotalReward = 0
        self.StartTime = 0
        self.end = 0
        #traci.vehicle.add("agent", "agent_route")
        #traci.gui.trackVehicle('View #0', "agent")

        #States
        self.OccMapState = np.zeros((40, 7))
        self.VehicleState = [0,0,0]
        self.RoadState = [0 for i in range(9)]
        self.state = None
        self.LeaderList = set()

        #property to simulate
        self.end_x = 0
        self.end_y = 2000
        self.AgentX = 0
        self.AgentY = 0
        self.AgentSpeed = 10
        self.AgentAccRate = 100
        self.AgentDecRate = 300
        self.minLaneNumber = 0
        self.maxLaneNumber = 1
        self.oldDistance = 0
        self.nowDistance = 0
        self.steptime = 0

    def reset(self):
        try:
            self.end = 0
            self.TotalReward = 0
            self.oldDistance = 0
            self.nowDistance = 0
            self.lastdistance = 0.99
            self.x_v = 0
            self.y_v = 0
            self.is_in = 0
            self.steptime = 0
            self.LeaderList = set()

            change_trip.change()
            traci.load(["-c",config_path,"--collision.action","remove","--no-step-log","--no-warnings","--no-duration-log"])

            print("Resetting...")
            #traci.vehicle.add("agent", "agent_route")
            #traci.gui.trackVehicle('View #0', "agent")

            traci.simulationStep()
            AgentAvailable = False
            while AgentAvailable == False:
                traci.simulationStep()
                self.VehicleIds = traci.vehicle.getIDList()
                if self.AgentId in self.VehicleIds:
                    AgentAvailable = True
                    self.StartTime = traci.simulation.getCurrentTime()
            for vehId in self.VehicleIds:
                traci.vehicle.subscribe(vehId,(tc.VAR_SPEED,tc.VAR_POSITION,tc.VAR_LANE_INDEX,tc.VAR_DISTANCE))
                traci.vehicle.subscribeLeader(self.AgentId,50)
                if vehId == self.AgentId:
                    traci.vehicle.setSpeedMode(self.AgentId,0)
                    traci.vehicle.setLaneChangeMode(self.AgentId,0)
            self.state,breakstop,overtake= self.perception()
        except:
            traci.start(sumoCmd)
            print("retrying")
            self.reset()
        return self.state

    def step(self,action):
        # define action:
        # action  |     meaning
        #    0    |    go straight
        #    1    |    break down
        #    2    |    change left
        #    3    |    change right
        #    4    |    do nothing


        for vehId in self.VehicleIds:
            traci.vehicle.subscribe(vehId,(tc.VAR_SPEED,tc.VAR_POSITION,tc.VAR_LANE_INDEX,tc.VAR_DISTANCE))
            traci.vehicle.subscribeLeader(self.AgentId,50)
            if vehId == self.AgentId:
                traci.vehicle.setSpeedMode(self.AgentId,0)
                traci.vehicle.setLaneChangeMode(self.AgentId,0)
                self.is_in = 1
                #traci.gui.trackVehicle('View #0', "agent")

        self.maxLaneNumber = 2

        self.end = 0
        reward = 0
        DistanceTravelled = 0
        if self.is_in == 1:
            if action == 0:
                maxSpeed = 30
                time = (maxSpeed - (traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_SPEED])) / self.AgentAccRate
                traci.vehicle.slowDown(self.AgentId, maxSpeed, 100*time)
            elif action == 1:
                time = ((traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_SPEED]) - 0)/self.AgentDecRate
                traci.vehicle.slowDown(self.AgentId, 0, 100*time)
            elif action == 2:
                laneindex = traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_LANE_INDEX]
                if laneindex < self.maxLaneNumber:
                    traci.vehicle.changeLane(self.AgentId,laneindex+1,100)
                traci.vehicle.setSpeed(self.AgentId, traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_SPEED])
            elif action == 3:
                laneindex = traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_LANE_INDEX]
                if laneindex > self.minLaneNumber:
                    traci.vehicle.changeLane(self.AgentId,laneindex-1,100)
                traci.vehicle.setSpeed(self.AgentId, traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_SPEED])
            elif action == 4:
                traci.vehicle.setSpeed(self.AgentId,traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_SPEED])

        traci.simulationStep()
        self.steptime += 1
        self.VehicleIds = traci.vehicle.getIDList()

        if self.is_in == 1:
            if self.AgentId in self.VehicleIds and self.steptime < DEAD_LINE:
                for vehId in self.VehicleIds:
                    traci.vehicle.subscribe(vehId,(tc.VAR_SPEED,tc.VAR_POSITION,tc.VAR_LANE_INDEX,tc.VAR_DISTANCE))
                    traci.vehicle.subscribeLeader(self.AgentId,50)

                Vehicle_Params = traci.vehicle.getSubscriptionResults(self.AgentId)
                self.AutocarSpeed = Vehicle_Params[tc.VAR_SPEED]
                posAutox = Vehicle_Params[tc.VAR_POSITION]
                if math.sqrt((self.end_x-posAutox[0])**2+(self.end_y-posAutox[1])**2)<30:
                    self.end = 100

                self.state,breakstop,overtake = self.perception()
                reward = self.cal_reward(self.end,breakstop,overtake)
            elif self.steptime < DEAD_LINE:
                #self.state = self.perception()
                self.end = 1
                reward = self.cal_reward(is_collision=self.end,breakstop=0,overtake=0)
                DistanceTravelled = 0
            else:
                self.end = 1
                reward = self.cal_reward(is_collision=2, breakstop=0,overtake=0)
                DistanceTravelled = 0

        return self.state, reward, self.end, DistanceTravelled

    def cal_reward(self,is_collision,breakstop,overtake):
        if is_collision == 1:
            print("collision!")
            return -30
        elif is_collision == 2:
            print("overtime")
            return -30
        elif is_collision == 100:
            print("arrive!")
            return 50
        else:
            self.nowDistance = traci.vehicle.getDistance(self.AgentId)
            del_distance = self.nowDistance - self.oldDistance
            reward = float(del_distance-8)/8 * self.nowDistance/500 + 5*overtake

            self.oldDistance = self.nowDistance
            if breakstop == 1:
                reward -= 10

            return reward

    def perception(self):

        #the state is defined as:
        # 0   | 1    | 2    | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
        #speed|cos(a)|sin(a)|l? |r? |dis| r | y | g |l? | c? | r? |
        #

        self.VehicleIds = traci.vehicle.getIDList()

        AllVehicleParams = []

        #----------------------------to get the vehicle state------------------------
        for vehId in self.VehicleIds:
            traci.vehicle.subscribe(vehId, (tc.VAR_SPEED, tc.VAR_POSITION, tc.VAR_ANGLE, tc.VAR_LANE_INDEX, tc.VAR_DISTANCE, tc.VAR_LANE_ID))
            VehicleParam = traci.vehicle.getSubscriptionResults(vehId)
            #AllVehicleParams.append(vehId)
            if vehId != self.AgentId:
                AllVehicleParams.append(VehicleParam)
            else:
                self.AgentSpeed = VehicleParam[tc.VAR_SPEED]
                self.AgentAngle = (VehicleParam[tc.VAR_ANGLE]/180)*math.pi
                self.AgentX = VehicleParam[tc.VAR_POSITION][0]
                self.AgentY = VehicleParam[tc.VAR_POSITION][1]
        self.VehicleState = [self.AgentSpeed,math.cos(self.AgentAngle),math.sin(self.AgentAngle)]

        #---------------------to calculate the occupanied state-----------------------
        LOW_X_BOUND = -6
        HIGH_X_BOUND = 6
        LOW_Y_BOUND = -18
        HIGH_Y_BOUND = 60
        self.OccMapState = np.zeros((40, 7))
        for VehicleParam in AllVehicleParams:
            VehiclePos = VehicleParam[tc.VAR_POSITION]
            rol = math.sqrt((VehiclePos[0]-self.AgentX)**2+(VehiclePos[1]-self.AgentY)**2)
            theta = math.atan2(VehiclePos[1]-self.AgentY,VehiclePos[0]-self.AgentX)

            reltheta = theta + self.AgentAngle
            relX = rol*math.cos(reltheta)
            relY = rol*math.sin(reltheta)
            if (relX>LOW_X_BOUND and relX<HIGH_X_BOUND) and (relY>LOW_Y_BOUND and relY<HIGH_Y_BOUND):
                indexX = int((6 + relX)/2 + 0.5)
                indexY = int((60 - relY)/2 + 0.5)

                self.OccMapState[indexY,indexX] = 1.0

            #add for fc dqn
        self.OccMapState = self.OccMapState.reshape(-1)

        #-------------------------------to get the RoadState----------------------------
        #RoadState: [leftcan rightcan distance r y g leftava centerava rightava]
        self.RoadState = [1.0 for i in range(9)]
        now_laneindex = 0

        overtake = 0
        remove_list = []
        for vehId in self.VehicleIds:
            if vehId == self.AgentId:
                now_laneindex = traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_LANE_INDEX]
                leaderinfo = traci.vehicle.getLeader(self.AgentId)
                if leaderinfo != None:
                    leader = leaderinfo[0]
                    self.LeaderList.add(leader)
                    myposition = traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_POSITION][1]
                    for vehicle in self.LeaderList:
                        #print(vehicle)
                        try:
                            position = traci.vehicle.getSubscriptionResults(vehicle)[tc.VAR_POSITION][1]
                            if myposition > position:
                                print("overtaking!")
                                overtake = 1
                                remove_list.append(vehicle)
                        except:
                            pass
                            #print("position get failed")
                    for loser in remove_list:
                        self.LeaderList.remove(loser)

        if now_laneindex == 0:
            self.RoadState = [0,1,1.000,0,0,1,0,1,1]
        elif now_laneindex == 1:
            self.RoadState = [1,1,1.000,0,0,1,1,1,1]
        else:
            self.RoadState = [1,0,1.000,0,0,1,1,1,0]

        if self.AgentSpeed <= 1:
            breakstop = 1
            print("breakstop")
        else:
            breakstop = 0


        return [self.OccMapState,self.VehicleState,self.RoadState],breakstop,overtake

