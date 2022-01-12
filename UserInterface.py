from tkinter import *


class Ui:
    def __init__(self, window):
        self.window = window
        self.InitUi()

    def InitUi(self):
        self.window.title("Product Recommendation Chat-bot")
        self.window.resizable(width=TRUE, height=TRUE)

        # Main space where the messages are shown
        self.messagesFrame = Frame(self.window, bd=0, relief="sunken", background="white")
        self.messagesFrame.pack(padx=20, pady=(20, 0), fill=None, expand=False)
        self.messages = Listbox(self.messagesFrame, height=20, width=70, borderwidth=0, highlightthickness=0,
                                font=('Arial', 14), background=self.messagesFrame.cget("background"))
        self.messages.pack(pady=10, padx=10)
        self.messages.insert(END, "\n")
        self.messages.insert(END, "                                                 PRODUCT BOOKING CHAT-BOT\n")
        self.messages.insert(END, "                      Loading language processing model... (Might take up to 30 seconds)\n")

        self.userMessage = StringVar()
        self.userMessageSent = StringVar()
        # Box where the user can type input
        self.entryField = Entry(self.window, width=80, textvariable=self.userMessage, font=('Arial', 12))
        self.entryField.pack(padx=20, pady=(20, 20))
        # Send text when user hits enter key
        self.entryField.bind("<Return>", self.SendUserMessage)

        # Render the elements
        self.window.update()

    # Checks for user input
    def GetUserInput(self):
        # Return user message if one has been sent
        if self.userMessageSent.get():
            utterance = self.userMessageSent.get()
            # utterance = 'I want mobile'
            self.userMessageSent.set('')
            return utterance

    # Sends user message into chat upon hitting enter key
    def SendUserMessage(self, event):
        if self.userMessage.get():
            # Insert message into chat
            self.messages.insert(END, '\n' + "You : " + self.userMessage.get())
            self.userMessageSent.set(self.userMessage.get())
            # Empty input box
            self.userMessage.set('')

    # Display agent response in chat
    def SendAgentMessage(self, agentMessage):
        self.messages.insert(END, '\n' + "Bot : " + agentMessage)

    def pushPossibleEntries(self, possibleEntries):
        outStr = 'Available Products >>>  '
        for item in possibleEntries:
            outStr += str(item['productname']) + ' , '
        self.messages.insert(END, '\n')
        self.messages.insert(END, '\n' + outStr[:-2])