class Memory:

    def __init__(self, max_history=5):
        self.max_history = max_history
        self.history = []

    def add(self, user_input, assistant_output):

        self.history.append((user_input, assistant_output))
        if len(self.history) > self.max_history:
            self.history.pop(0)  