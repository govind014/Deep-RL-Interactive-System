from Database import *
from Config import *
import numpy as np, math


def availableProducts(tag_dict):
    # categories = ['productname', 'city', 'category', 'pricing']
    print("AVAILABLE PRODUCTS")
    for item in tag_dict:
        print("------>", item['productname'], ': (', item['city'], ')')


def stateFormulation(tmpState):
    stateNames = ['agentIntentRepresentation', 'userIntentRepresentation', 'agentRequestRepresentation',
                  'userInformRepresentation', 'filledSlotsRepresentation', 'turnOneHotRepresentation',
                  'dbResults']
    print("\n>>> State Formulation\n")
    for item, name_ in zip(tmpState[:-1], stateNames[:-1]):
        print(name_, " : ", item, " -> ", len(item))
    print(stateNames[-1], "  :  ", tmpState[-1])



class StateTracker:

    def __init__(self):
        # Dictionaries used for indexing of the one-hot and multi-hot encodings in state representation
        self.userIntentsDictionary = self.ListToIndexDictionary(userIntents)
        self.agentIntentsDictionary = self.ListToIndexDictionary(agentIntents)
        self.slotsDictionary = self.ListToIndexDictionary(allSlots)
        self.filledSlotsDictionary = self.ListToIndexDictionary(fillableSlots)
        self.Reset()

    def Reset(self):
        self.history = []
        self.filledSlots = {}

    # Saves agent action in history
    def UpdateAgentAction(self, agentAction):
        self.history.append(agentAction)

        # print(">>> HISTORY (agent) : ", len(self.history))

    # Saves user action in history and saves the inform slots
    def UpdateUserAction(self, userAction):
        self.history.append(userAction)
        if userAction['intent'] == 'inform':
            for slot, value in userAction['informSlots'].items():
                self.filledSlots[slot] = value
                if slot == 'productname' and value != 'any':
                    self.filledSlots['match'] = value
        # print(">>> HISTORY (user) : ", self.history, len(self.history))

    def GetPossibleEntries(self):
        # start with all database entries and remove misses
        self.possibleEntries = copy.deepcopy(database)
        # e.g. slot = city and value = London
        for slot, value in self.filledSlots.items():
            if value == 'any':
                continue
            possibleEntriesForIteration = copy.deepcopy(self.possibleEntries)
            for entry in possibleEntriesForIteration:
                if slot in entry and value != entry[slot]:
                    self.possibleEntries.remove(entry)
        # print(f'\n>> Possible Entries : \n',possibleEntries)
        #availableProducts(self.possibleEntries)
        return self.possibleEntries

    def GetPossibleEntriesNER(self):
        # keys = ['productname', 'city', 'category', 'pricing']
        self.possibleEntries = copy.deepcopy(database)
        # print(new_dict)
        uniqueKeys = []
        uniqueData = []
        for item in self.filledSlots.keys():
            uniqueKeys.append(item)
            for value in self.filledSlots.values():
                tmp = [item, value]
                uniqueData.append(tmp)
        for items in uniqueData:
            key = items[0]
            value = items[1]
            dictCheck = {key: value}
            # print((dictCheck))
            for data in database:
                if data[key] == dictCheck[key]:
                    print(data)

        # print("\n>>> Possible Entries :  ", possibleEntries)
        return self.possibleEntries

    # Prepares a state representation with useful information for the agent
    def GetStateRepresentation(self):
        lastAgentAction = self.history[-2] if len(self.history) > 1 else None
        lastUserAction = self.history[-1]
        # print('\n>>> LAST AGENT ACTION : ', lastAgentAction)
        # print('>>> LAST USER ACTION : ', lastUserAction)

        #  encoding of the last agent action intent
        agentIntentRepresentation = np.zeros(len(agentIntents))
        if lastAgentAction:
            agentIntentRepresentation[self.agentIntentsDictionary[lastAgentAction['intent']]] = 1.0

        # encoding of the last user action intent
        userIntentRepresentation = np.zeros(len(userIntents))
        userIntentRepresentation[self.userIntentsDictionary[lastUserAction['intent']]] = 1.0

        # encoding of the last agent action request slots
        agentRequestRepresentation = np.zeros(len(allSlots))
        if lastAgentAction and lastAgentAction['requestSlots']:
            agentRequestRepresentation[self.slotsDictionary[lastAgentAction['requestSlots']]] = 1.0

        # encoding of the last user action inform slots
        userInformRepresentation = np.zeros(len(allSlots))
        for key in lastUserAction['informSlots'].keys():
            userInformRepresentation[self.slotsDictionary[key]] = 1.0

        # encoding of the slots that have already been filled
        filledSlotsRepresentation = np.zeros(len(fillableSlots))
        for key in self.filledSlots:
            filledSlotsRepresentation[self.filledSlotsDictionary[key]] = 1.0

        # one-hot encoding of the current turn
        # Get the value of the current turn
        turnRepresentation = math.ceil(len(self.history) / 2)
        turnOneHotRepresentation = np.zeros(TURN_LIMIT)
        turnOneHotRepresentation[turnRepresentation - 1] = 1.0

        # Get the count of matching database entries with current constraints (filledSlots)
        dbResults = 0
        if len(self.GetPossibleEntries()) > 0:
            dbResults = 1

        tmpState = [agentIntentRepresentation, userIntentRepresentation, agentRequestRepresentation,
                    userInformRepresentation, filledSlotsRepresentation, turnOneHotRepresentation,
                    dbResults]

        # Combines all the representations and values in an array of arrays and reduces them into one dimension
        stateRepresentation = np.hstack(
            [agentIntentRepresentation,
             userIntentRepresentation,
             agentRequestRepresentation,
             userInformRepresentation,
             filledSlotsRepresentation,
             turnOneHotRepresentation,
             dbResults]).flatten()

        # print(f'\n>>> State representation : ', stateRepresentation)
        # print(stateRepresentation)
        return stateRepresentation

    def GetStateSize(self):
        # len(agentIntents) + len(userIntents) for one-hot encodings of last agent action intent as well as last user action intent
        # 2 * len(allSlots) for multi-hot encodings of last agent's request slots and the last user's inform slots
        # len(fillableSlots) for multi-hot encodings of filledSlots i.e. the constraints
        # TURN_LIMIT for the one-hot encoding of the current turn
        # 1 for the boolean indicating the presence of any database matches
        return len(agentIntents) + len(userIntents) + 2 * len(allSlots) + len(fillableSlots) + TURN_LIMIT + 1

    # Takes a list and retuns its values as keys and their indices as values in a dictionary
    def ListToIndexDictionary(self, list):
        return {k: v for v, k in enumerate(list)}
