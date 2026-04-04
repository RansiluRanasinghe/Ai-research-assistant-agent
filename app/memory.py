class Memory:

    def __init__(self, max_history=5):
        self.max_history = max_history
        self.history = []

    def add(self, user_input, assistant_output):

        self.history.append((user_input, assistant_output))
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_context(self):

        if not self.history:
            return ""
        
        context = "Previous conversations:\n"
        for user, assistant in self.history:
            context += f"User: {user}\nAssistant: {assistant}\n"

        return context

if __name__ == "__main__":

    memory = Memory(max_history=3)
    memory.add("What is AI?", "AI stands for Artificial Intelligence.")
    memory.add("Tell me more.", "It is the simulation of human intelligence in machines.")
    memory.add("What about machine learning?", "ML is a subset of AI.")
    memory.add("Is that all?", "There are many other aspects.")

    print("Memory context:")
    print(memory.get_context())                 