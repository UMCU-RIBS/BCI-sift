import ray

__all__ = ["DictManager", "ListManager"]


@ray.remote
class DictManager:
    def __init__(self):
        self.data = {}

    def set(self, key, value):
        """Set a key-value pair in the dictionary."""
        self.data[key] = value

    def get(self, key):
        """Retrieve a value by key."""
        return self.data.get(key, None)

    def remove(self, key):
        """Remove a key-value pair by key."""
        if key in self.data:
            del self.data[key]

    def clear(self):
        """Clear the dictionary."""
        self.data.clear()

    def keys(self):
        """Get all keys in the dictionary."""
        return list(self.data.keys())

    def values(self):
        """Get all values in the dictionary."""
        return list(self.data.values())

    def items(self):
        """Get all key-value pairs as a list of tuples."""
        return list(self.data.items())

    def len(self):
        """Get the number of items in the dictionary."""
        return len(self.data)


@ray.remote
class ListManager:
    def __init__(self):
        self.results = []

    def append(self, result):
        """Add a result to the list."""
        self.results.append(result)

    def get(self):
        """Get the list of results."""
        return self.results

    def set(self, new_results):
        """Set the list of results to a new list."""
        self.results = new_results

    def clear(self):
        """Clear the results list."""
        self.results.clear()

    def pop(self, index):
        """Remove a result by index."""
        self.results.pop(index)

    def retrieve(self, index):
        """Get a result by index."""
        return self.results[index]

    def len(self):
        """Get the number of results stored."""
        return len(self.results)
