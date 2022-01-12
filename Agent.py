import UserInterface
from DeepQNetwork import *
from StateTracker import *


# Returns a database entry given a product name
def GetEntryFromDb(productname):
    #print(f'Incoming id : {productname}')
    for entry in copy.deepcopy(database):
        if entry['productname'].lower() == productname.lower():
            #print(">>> DATABASE ENTRY : ", entry)
            if entry is not None:
                return entry
            else:
                print("Cannot find product from db")
                return entry


class Agent:

    def __init__(self, stateSize):
        # Replay buffer for storing and sampling experience for learning
        self.memory = ReplayBuffer()
        # Online network for choosing actions
        self.onlineNetwork = InitializeDqn(stateSize)
        # Target network for computing target values in learning
        self.targetNetwork = InitializeDqn(stateSize)
        # The final reservation made to end the dialogue
        self.chosenReservation = None
        # Initialize epsilon with 1 for epsilon-decreasing strategy
        self.epsilon = 1
        self.stateTracker = StateTracker()

    # Takes the current state and its size to choose an action
    def PredictNextAction(self, state, stateSize):
        rand = random.random()
        # Determine if random or policy-based action

        if IN_TRAINING and rand < self.epsilon:
            nextAction = random.choice(agentActions)

        else:
            # Network outputs value for each agent action based on the state
            actionValues = self.onlineNetwork.predict(state.reshape(1, stateSize)).flatten()
            # print(f'\n>>> Action Values : ', actionValues)
            # Gets index of action with highest Q-value
            nextActionIndex = np.argmax(actionValues)
            # Returns corresponding action based on the chosen index
            nextAction = self.IndexToAction(nextActionIndex)

        # Decrease the epsilon
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECREASE

        return nextAction

    # Takes an index and returns the corresponding agent action
    def IndexToAction(self, index):
        for (i, action) in enumerate(agentActions):
            if index == i:
                return copy.deepcopy(action)

    # Takes an agent action and returns the corresponding index
    def ActionToIndex(self, action):
        for (i, a) in enumerate(agentActions):
            if action == a:
                return i

    # Reset the agent for usage in new dialogue
    def Reset(self):
        self.chosenReservation = None

    # Adapts the network weights by learning from memory
    def Learn(self, stateSize):
        # Only start learning if a complete batch can be sampled from the buffer
        if self.memory.indexCounter < BATCH_SIZE:
            return

        batch = self.memory.SampleBatchFromBuffer()
        # print(f'batch (SampleBatchFromBuffer()) : ',batch)

        # For each tuple in the batch
        for state, action, reward, nextState in batch:
            # Compute Q-values of online network
            qNow = self.onlineNetwork.predict(state.reshape(1, stateSize)).flatten()
            # print(f'qNow : ',qNow)
            # Initialize target Q-values with online Q-values
            qTarget = qNow.copy()

            # If the tuple has no next state, the Q-value is the reward
            if isinstance(nextState, list):
                qTarget[self.ActionToIndex(action)] = reward
            else:
                # Compute Q-values of the next state using the target network
                qNext = self.targetNetwork.predict(nextState.reshape(1, stateSize)).flatten()
                # Set target Q-value of the tuple action to the reward plus the discounted maximal Q-value of the next state
                qTarget[self.ActionToIndex(action)] = reward + GAMMA * max(qNext)

            # Adjust the weights according to the difference of online and target values
            self.onlineNetwork.fit(state.reshape(1, stateSize), qTarget.reshape(1, len(agentActions)), epochs=1,
                                   verbose=0)

    # Copies weights of the online network to the target network
    def CopyToTargetNetwork(self):
        self.targetNetwork.set_weights(self.onlineNetwork.get_weights())

    # Saves online network
    def SaveModel(self):
        self.onlineNetwork.save(FILE_NAME)

    # Loads online network
    def LoadModel(self):
        self.onlineNetwork = load_model(FILE_NAME)

    # Choose request utterance based on the slot
    def GenerateRequestResponse(self, nextAction):
        slot = nextAction['requestSlots']

        if slot == 'productname':
            # tmp_possibleEntries = self.stateTracker.GetPossibleEntries()
            # availableProducts(tmp_possibleEntries)
            return '\nDo you have a specific product in mind? \n'
        elif slot == 'numberofproducts':
            return '\nHow many items do you need?\n'
        elif slot == 'city':
            return '\nPlease enter your location for delivery?\n'
        elif slot == 'time':
            return '\nAt what time do you want this product be delivered?\n'
        elif slot == 'category':
            return '\nWhich category are you looking for?\n'
        elif slot == 'pricing':
            return '\nHow high shall the pricing be?\n'

    # Compose response to propose a matching product
    def GenerateMatchFoundResponse(self, nextAction):
        responseString = []
        # Missing inform slots indicate that there is no matching database entry
        if nextAction['informSlots']:
            match = nextAction['informSlots']
            responseString.append(f"How about \"{match['productname']}\"? ")
            responseString.append(f"It is available at {match['city']}  ")
            responseString.append(f" (Category : {match['category']}, Pricing :  {match['pricing'].lower()})")
        else:
            responseString.append('No product matches the current information.')
        return ''.join(responseString)

    # Compose response for finishing the dialogue and informing about the made reservation

    def GenerateDoneResponse(self, filledSlots):
        # try:
        if 'match' in filledSlots.keys():
            self.chosenReservation = GetEntryFromDb(filledSlots['match'])
            #print("Chosen reservation slot : ", self.chosenReservation)
            for done_slot, value in filledSlots.items():
                if done_slot not in self.chosenReservation.keys():
                    self.chosenReservation[done_slot] = value

        else:
            self.chosenReservation = filledSlots

        #print("\n Chosen reservation slot : ", self.chosenReservation)
        if 'productname' in self.chosenReservation.keys() and not self.chosenReservation['productname'] == 'any':
            name = self.chosenReservation['productname']
        else:
            name = 'Unknown'

        if 'city' in self.chosenReservation.keys() and not self.chosenReservation['city'] == 'any':
            city = self.chosenReservation['city']
        else:
            city = 'Unknown'

        if 'numberofproducts' in self.chosenReservation.keys():
            numberofproducts = self.chosenReservation['numberofproducts']
        else:
            numberofproducts = 'Unknown'

        if 'time' in self.chosenReservation.keys():
            time = self.chosenReservation['time']
        else:
            time = 'Unknown'

        return f"The booking has been done for {name} at {city} for {numberofproducts} items and items will be delivered by {time}."
