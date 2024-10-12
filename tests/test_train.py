# tests/test_train.py
import unittest
from train import train_model
import os

class TestTrain(unittest.TestCase):
    def test_model_training(self):
        # Train the model and check if the model file is created
        train_model()
        self.assertTrue(os.path.exists('models/model.pkl'))

if __name__ == '__main__':
    unittest.main()
