import pickle
import os

class StateManager:
    def __init__(self, state_file='state.pkl'):
        self.state_file = state_file

    def save_state(self, state_data):
        try:
            with open(self.state_file, 'wb') as f:
                pickle.dump(state_data, f)
        except Exception as e:
            print(f"Error saving state: {e}")

    def load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'rb') as f:
                    return pickle.load(f)
            else:
                return {}
        except Exception as e:
            print(f"Error loading state: {e}")
            return {}
