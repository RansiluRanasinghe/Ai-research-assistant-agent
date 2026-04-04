class Memory:

    def __init__(self, max_history=5):
        self.max_history = max_history
        self.history = []