from flair.data import Sentence
from flair.models import SequenceTagger
from Config import tagDict

model = SequenceTagger.load('H:\\DRL-Interactive-System\\best-model-2.pt')


def entitySelection(input_):
    """ Get user input -> find NER from it -> return dictionary of found entities"""
    sentence = Sentence(input_)
    model.predict(sentence)
    # print(sentence.to_tagged_string())
    out = sentence.to_tagged_string()
    outList = outListFormat(out)
    slot_key, slot_value = combineNER(outList)
    updatedSlots = slotUpdate(slot_key, slot_value)
    #print(updatedSlots)
    return updatedSlots


def outListFormat(out):
    temp_outlist = out.split()
    for items in tagDict.keys():
        # print(items)
        for value in tagDict[items]:
            # print(value)
            if value in temp_outlist:
                # print(temp_outlist.index(value))
                loc = temp_outlist.index(value)
                temp_outlist[loc] = items
    return temp_outlist
    # combineNER(temp_outlist)


def combineNER(temp_outlist):
    slot_key = []
    slot_value = []
    listLength = len(temp_outlist)
    # for i in range(listLength):
    i = 0
    while i < listLength:
        for key in tagDict.keys():
            if key in temp_outlist[i]:
                tmpCombineProduct = []
                if (i < len(temp_outlist) - 2) and (key in (temp_outlist[i] and temp_outlist[i + 2])):

                    tmpCombineProduct = [temp_outlist[i - 1].capitalize(), temp_outlist[i + 1].capitalize()]
                    combineProduct = ' '.join(map(str, tmpCombineProduct))
                    # print(combineProduct)
                    slot_value.append(combineProduct)
                    i += 2
                else:
                    slot_value.append(temp_outlist[i - 1].capitalize())
                slot_key.append(temp_outlist[i])
        i += 1
    # slotUpdate(slot_key, slot_value)
    return slot_key, slot_value


def slotUpdate(slot_key, slot_value):
    slot_dict = {}
    for i in range(len(slot_key)):
        slot_dict.setdefault(slot_key[i], [])
        if slot_key[i] in slot_dict.keys():
            slot_dict[slot_key[i]].append(slot_value[i])
    #print(slot_dict)
    singleSlotDict = {}
    for key in slot_dict.keys():
        if key != 'time':
            singleSlotDict[key] = slot_dict[key][0]
        else:
            tmp = slot_dict[key][0]
            singleSlotDict[key] = tmp.lower()

    # print("\n>>> Entity Selection : ", singleSlotDict)
    return singleSlotDict



#entitySelection('I want Apple iphone at kochi 3 am')








# model = SequenceTagger.load('H:\\Product-Recomendation\\custom-ner-models\\Flair\\custom-ner\\best-model-2.pt')

# # tags = ["<B-PRICING>","<I-PRICING>", "<O-PRICING>",
# #         "<B-PRODUCTNAME>", "<I-PRODUCTNAME>", "<O-PRODUCTNAME>",
# #         "<B-CATEGORY>", "<I-CATEGORY>", "<O-CATEGORY>",
# #         "<B-CITY>", "<I-CITY>", "<O-CITY>",
# #         "<B-NUMBEROFPRODUCTS>", "<I-NUMBEROFPRODUCTS>", "<O-NUMBEROFPRODUCTS>",
# #         "<B-TIME>", "<I-TIME>", "<O-TIME>"]

# def entitySelection(input_):
#     """ Get user input -> find NER from it -> return dictionary of found entities"""

#     sentence = Sentence(input_)
#     model.predict(sentence)
#     #print(sentence.to_tagged_string())
#     out = sentence.to_tagged_string()

#     slot_key = []
#     slot_value = []
#     outList = out.split()
#     for i in range(len(outList)):
#         for tag in tags:
#             #print(tag, outList)
#             if outList[i] == tag:
#                 #print(outList[i-1], tag)
#                 slot_key.append(outList[i][3:-1].lower())
#                 slot_value.append(outList[i-1].capitalize())

#     # slot_dict = {}
#     # for i in range(len(slot_key)):
#     #     slot_dict.setdefault(slot_key[i], [])
#     #     if slot_key[i] in slot_dict.keys():
#     #         slot_dict[slot_key[i]].append(slot_value[i])
#     # #print(slot_dict)
#     # return slot_dict

#     for items in tagDict.keys():
#         # print(items)
#         for value in tagDict[items]:
#             # print(value)
#             if value in tempOutlist:
#                 # print(tempOutlist.index(value))
#                 loc = tempOutlist.index(value)
#                 tempOutlist[loc] = items
#     print(tempOutlist)
#     slot_key = []
#     slot_value = []
#     listLength = len(tempOutlist)
#     # for i in range(listLength):
#     i = 0
#     while i < listLength:

#         for key in tagDict.keys():
#             if key in tempOutlist[i]:
#                 tmpCombineProduct = []
#                 if (i < len(tempOutlist) - 3) and (key in (tempOutlist[i] and tempOutlist[i + 2])):
#                     tmpCombineProduct = [tempOutlist[i - 1], tempOutlist[i + 1]]
#                     combineProduct = ' '.join(map(str, tmpCombineProduct))
#                     # print(combineProduct)
#                     slot_value.append(combineProduct)
#                     i += 2
#                 else:
#                     slot_value.append(tempOutlist[i - 1].capitalize())
#                 slot_key.append(tempOutlist[i])
#         i += 1
#     # print(slot_value, slot_key)
#     slot_dict = {}
#     for i in range(len(slot_key)):
#         slot_dict.setdefault(slot_key[i], [])
#         if slot_key[i] in slot_dict.keys():
#             slot_dict[slot_key[i]].append(slot_value[i])
#     # print(slot_dict)
#     singleSLotDict = {}
#     for key in slot_dict.keys():
#         singleSLotDict[key] = slot_dict[key][0]
#     print(singleSLotDict)

# if __name__ == "__main__":
#     #print("Start")
#     #inp  = str(input("Enter query : "))
#     inp = 'I want poco m2'
#     entitySelection(inp)
