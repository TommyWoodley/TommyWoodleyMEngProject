import json
import numpy as np

class OptimisedDemo:
    def __init__(self, file_path=None, metadata=None, trajectory=None):
        if file_path:
            self.data = self.load_json(file_path)
            self.metadata = self.data.get('metadata')
            self.trajectory = np.array(self.data.get('trajectory'))
        else:
            self.metadata = metadata
            self.trajectory = trajectory

        self.starting_position = self.metadata.get('starting_position')
        self.duration = self.metadata.get('duration')

    @staticmethod
    def load_json(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    @staticmethod
    def write_json(file_path, data):
        data['trajectory'] = data['trajectory'].tolist()
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

    def save_to_file(self, file_path):
        data = {
            'metadata': self.metadata,
            'trajectory': self.trajectory
        }
        self.write_json(file_path, data)

    def get_starting_position(self):
        return self.starting_position

    def get_duration(self):
        return self.duration
