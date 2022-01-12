from Agent import *
from UserSimulator import *
from StateTracker import *
from NaturalLanguageProcessor import *
from UserInterface import *
import threading
import time

turnCount = 0
resultDictionary = {-1: 'FAIL', 1: 'SUCCESS'}
rewardMatrix = []

#
# def turnCounter():
#     global turnCount
#     turnCount += 1
#     return turnCount
#     # print(f'Turn : {self.turnCount}')


class DialogueManager(threading.Thread):

    def __init__(self):

        if REAL_USER:
            # self.ui = Ui(window)
            self.ui = Ui()
            # Start new thread to run code and ui simultaneously
            threading.Thread.__init__(self)
            self.nlp = NaturalLanguageProcessor()

            # Write initial bot message
            self.ui.messages.insert(END, "\n")
            # self.ui.messages.insert(END, "\tPRODUCT BOOKING CHAT-BOT")
            self.ui.messages.insert(END, "\n")
            self.ui.messages.insert(END, "Bot : Hello, how may I help you ?..")
            self.ui.entryField.focus()
        self.stateMatrix = []
        self.turnCount = 0
        self.user = UserSimulator()
        self.stateTracker = StateTracker()
        self.agent = Agent(self.stateTracker.GetStateSize())
        self.successCounter = 0
        # Done indicates whether a dialogue/ an episode is finished
        self.done = False
        self.Run()

    # Reset user, agent and state tracker for each new dialogue
    def Reset(self):
        global turnCount, rewardMatrix
        turnCount = 0
        rewardMatrix = []
        self.done = False
        self.user.Reset()
        self.agent.Reset()
        self.stateTracker.Reset()

    # Main function
    def Run(self):
        if not IN_TRAINING:
            # Load model if in testing
            self.agent.LoadModel()
            self.GetUserAction(None)
            while not self.done:
                self.Step()
        else:
            self.Train()

    # Main loop for training
    def Train(self):
        # For a number of training episodes
        for x in range(TRAIN_AMOUNT + 1):
            self.Reset()

            if x % PRINT_PROGRESS_INTERVAL == 0:
                self.PrintProgress(x)
            # if PRINTING:
            #     print('New Dialogue:')
            # Performs initial user action, None as parameter because no previous agent action
            self.GetUserAction(None)
            # Execute dialogue turn until agent finishes dialogue
            while not self.done:
                result = self.Step()
            # print(f'Reward Matrix = {rewardMatrix}')
            # print(f'Expected Return = {sum(rewardMatrix)}')
            # print(f'Result : {resultDictionary[result]}')
            # Copy online network weights to target network at specific interval
            if x % TARGET_UPDATE_INTERVAL == 0:
                self.agent.CopyToTargetNetwork()
            # Adjust the agent's network after each dialogue
            self.agent.Learn(self.stateTracker.GetStateSize())

    def Step(self):
        # Prepares a digestible representation of useful information for the network
        stateRepresentation = self.stateTracker.GetStateRepresentation()
        # print("\n>>> STATE REPRESENTATION : ", stateRepresentation)
        self.stateMatrix.append(stateRepresentation)

        # TURN_COUNT = turnCounter()
        # print(TURN_COUNT)
        # Network chooses an action based on the state
        nextAgentAction = self.agent.PredictNextAction(stateRepresentation, self.stateTracker.GetStateSize())

        # Agent generates a template-based response
        if nextAgentAction['intent'] == 'request':
            agentResponse = self.agent.GenerateRequestResponse(nextAgentAction)

        elif nextAgentAction['intent'] == 'matchFound':
            nextAgentAction = self.FillWithMatch(nextAgentAction)
            # self.ui.pushPossibleEntries(self.stateTracker.GetPossibleEntries())
            agentResponse = self.agent.GenerateMatchFoundResponse(nextAgentAction)

        elif nextAgentAction['intent'] == 'done':
            # print(" \n Generate matchFound response \n")
            # print("\n>>> FILLED SLOTS : ", self.stateTracker.filledSlots)
            agentResponse = self.agent.GenerateDoneResponse(self.stateTracker.filledSlots)
            self.done = True

        # if PRINTING:
        #     print(agentResponse)

        # Display the agent response for a user
        if REAL_USER:
            # availableProducts(self.stateTracker.possibleEntries)
            self.ui.messages.insert(END, '\n')
            self.ui.SendAgentMessage(agentResponse)

        # Update state tracker with agent action
        self.stateTracker.UpdateAgentAction(nextAgentAction)

        userAction, reward, result = self.GetUserAction(nextAgentAction)
        # reward = rewardRemap(reward,TURN_COUNT)
        #rewardMatrix.append(reward)
        # print(f'Turn Count {TURN_COUNT} - Boosted Reward : {reward} ')
        if IN_TRAINING and result != NO_RESULT:
            self.done = True
            if result == SUCCESS:
                self.successCounter += 1

        # Prepare a representation of the new state with current information
        nextStateRepresentation = self.stateTracker.GetStateRepresentation()
        self.stateMatrix.append(nextStateRepresentation)

        # Store tuples in replay buffer for later learning
        if IN_TRAINING:
            if self.done:
                nextStateRepresentation = []
            self.agent.memory.StoreTransition(stateRepresentation, nextAgentAction, reward, nextStateRepresentation)

            return result

    # Returns a user action, a reward and a dialogue result based on the last agent action
    def GetUserAction(self, agentAction):
        if not REAL_USER:
            # Use the user simulator to get an action if not a real user
            userAction, reward, result = self.user.GetNextAction(math.ceil(len(self.stateTracker.history) / 2),
                                                                 self.agent.chosenReservation, agentAction)
        else:
            # Get user utterance in UI
            userUtterance = ''
            while not userUtterance:
                userUtterance = self.ui.GetUserInput()
                self.ui.window.update()
            time.sleep(0.1)
            # Extract intent and slots from utterance using the natural language processor
            userAction = self.nlp.GetSemanticFrame(userUtterance, agentAction)
            # Set reward and result = 0 because they only matter in training (when using the user sim)
            reward = 0
            result = 0
            if userAction['intent'] == 'reject':
                self.Reset()

        # Update state tracker with user action
        self.stateTracker.UpdateUserAction(userAction)

        # if PRINTING:
        # print(userAction)

        return userAction, reward, result

    # Chooses a random entry from the database that matches the current constraints and fills agent inform slots
    def FillWithMatch(self, nextAgentAction):
        possibleEntries = self.stateTracker.GetPossibleEntries()
        # print('\n>>> Possible Entries : ', possibleEntries)

        if possibleEntries:
            # Choose random restaurant from the possible entries
            chosenEntry = random.choice(possibleEntries)
            # Set match in the filled slots
            self.stateTracker.filledSlots['match'] = chosenEntry['productname']
            # Fill agent inform slots with information of the chosen restaurant
            for slot, value in chosenEntry.items():
                nextAgentAction['informSlots'][slot] = value
        else:
            # Remove match from filled slots if database has no matches
            if 'match' in self.stateTracker.filledSlots.keys():
                self.stateTracker.filledSlots.pop('match')
            nextAgentAction['informSlots'] = {}

        return nextAgentAction

    # Prints the number of episodes and the average success rate and saves the model each X episodes
    def PrintProgress(self, episodeCount):
        successRate = self.successCounter / PRINT_PROGRESS_INTERVAL
        print(f'\nEpisode: {episodeCount}')
        print(f'Success rate: {successRate}')
        self.successCounter = 0
        self.agent.SaveModel()


# Define window
# window = Tk()
# Start dialogue manager
dm = DialogueManager()
# Start UI if real user
# if REAL_USER:
#     window.mainloop()
