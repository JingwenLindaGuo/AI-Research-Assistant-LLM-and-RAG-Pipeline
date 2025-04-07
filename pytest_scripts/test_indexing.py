
# Fake Pinecone client that simulates the behavior of Pinecone.
class FakePinecone:
    def __init__(self):
        self.indexes = []          # List to hold index names.
        self.created_indexes = []  # Log of create_index calls.
    
    def list_indexes(self):
        return [{"name": name} for name in self.indexes]
    
    def create_index(self, name, dimension, metric, spec):
        self.indexes.append(name)
        self.created_indexes.append((name, dimension, metric, spec))
    
    def delete_index(self, index_name):
        if index_name in self.indexes:
            self.indexes.remove(index_name)
        else:
            # Simulate behavior when deletion fails.
            raise Exception("Index does not exist")
    