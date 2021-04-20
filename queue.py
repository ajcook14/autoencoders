class Queue():

    def __init__(self):

        self.queue = []

    def __len__(self):

        return(len(self.queue))

    def is_empty(self):

        return(len(self.queue) == 0)

    def append(self, item):

        self.queue.append(item)

    def serve(self):

        if self.is_empty():

            print("No items to serve.")

            return(None)

        item = self.queue[0]

        self.queue = self.queue[1:]

        return(item)

