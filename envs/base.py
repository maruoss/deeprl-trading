from abc import ABC, abstractmethod


class Environment(ABC):
    @property
    @abstractmethod
    def observation_space(self):
        """provide the shape and lower/upper bounds of observations"""
        pass

    @property
    @abstractmethod
    def action_space(self):
        """provide the shape and lower/upper bounds of actions"""
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass
