import random, copy

###Constants

# Defines whether the program is in training
IN_TRAINING = True
# Defines whether the program is used by a real user or the user sim
REAL_USER = False
# For printing the dialogues in training
PRINTING = True
# For printing the success rate
PRINT_PROGRESS_INTERVAL = 5
# Minimum epsilon value for epsilon-decreasing exploration
EPSILON_MIN = 0.01
# Value which epsilon is multiplied by to decrease its impact over time
EPSILON_DECREASE = 0.9999
# Learning rate alpha
ALPHA = 0.0002
# Discount factor gamma for value updates
GAMMA = 0.99
# Capacity of the replay buffer
MEMORY_CAPACITY = 10000
# Batch size of sampled tuples from the replay buffer for training
BATCH_SIZE = 16
# File name of the the dqn model saved at the end of training
FILE_NAME = 'dqn_model.h5'
# How many rounds can occur in a conversation at most
TURN_LIMIT = 20
# How many episodes/dialogues to train in total
TRAIN_AMOUNT = 500
# Number of neurons in the deep q networks hidden layer
HIDDEN_SIZE = 80
# Number of turns between target network updates based on the online network
TARGET_UPDATE_INTERVAL = 100
# Used to check for result of dialogue in user simulator
FAIL = -1
NO_RESULT = 0
SUCCESS = 1

# Slots
allSlots = ['productname', 'numberofproducts', 'city', 'time', 'category', 'pricing']
fillableSlots = ['match', 'productname', 'numberofproducts', 'city', 'time', 'category', 'pricing']
necessarySlots = ['productname', 'numberofproducts', 'time']
optionalSlots = ['city', 'category', 'pricing']
requestableSlots = ['city', 'category', 'pricing']

slotDictionary = {'productname': ['Poco M2', 'Motorola Edge 20', 'Apple Iphone', 'Roadster',
                                  'Polo', 'Cashew Nut', 'Headphones', 'Laptop', 'Cricket Bat',
                                  'Suitcase', 'Samsung Note', 'Horlicks', 'Football',
                                  'Gym workout Kit', '4K TV', 'Earphones', 'Redmi Note 9',
                                  'Samsung M51', 'Sunflower Oil', 'Football', 'Yonex racquet',
                                  'Travel bags', 'Tshirt', 'Realme PowerBank', 'DSLR Camera',
                                  'Realme Narzo 5G', 'Ashirvad Atta', 'Peanut Butter', 'Roasted Almond',
                                  'Bata Footwear', 'Carromboard', 'Mobile Case', 'DTH', 'eGPU'],
                  'city': ['Trivandrum', 'Bangalore', 'Kochi', 'Chennai', 'Pune', 'India', 'Kolkata'],
                  'category': ['Sports', 'Grocery', 'Mobile', 'Fashion', 'Electronics'],
                  'pricing': ['Expensive', 'Cheap', 'Average', 'cost', 'cheap', 'price.']}

tagDict = {
    'pricing': ["<B-PRICING>", "<I-PRICING>", "<O-PRICING>"],
    'productname': ["<B-PRODUCTNAME>", "<I-PRODUCTNAME>", "<O-PRODUCTNAME>"],
    'category': ["<B-CATEGORY>", "<I-CATEGORY>", "<O-CATEGORY>"],
    'city': ["<B-CITY>", "<I-CITY>", "<O-CITY>"],
    'numberofproducts': ["<B-NUMBEROFPRODUCTS>", "<I-NUMBEROFPRODUCTS>", "<O-NUMBEROFPRODUCTS>"],
    'time': ["<B-TIME>", "<I-TIME>", "<O-TIME>"]}

# For one-hot encoding in state representation
userIntents = ['inform', 'reject', 'confirm']
agentIntents = ['done', 'matchFound', 'request']

# Set all possible actions of the agent
agentActions = []
agentActions.append({'intent': 'done', 'requestSlots': None})
agentActions.append({'intent': 'matchFound', 'requestSlots': None, 'informSlots': {}})

for slot in allSlots:
    agentActions.append({'intent': 'request', 'requestSlots': slot})
