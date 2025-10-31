import unittest
from src.utils.io import load_model, save_model
from src.utils.metrics import calculate_accuracy

class TestUtils(unittest.TestCase):

    def test_load_model(self):
        model = load_model('path/to/model')
        self.assertIsNotNone(model)

    def test_save_model(self):
        model = 'dummy_model'
        save_model(model, 'path/to/save/model')
        # Add assertions to check if the model was saved correctly

    def test_calculate_accuracy(self):
        predictions = [1, 0, 1]
        labels = [1, 0, 0]
        accuracy = calculate_accuracy(predictions, labels)
        self.assertEqual(accuracy, 0.6666666666666666)  # Example expected accuracy

if __name__ == '__main__':
    unittest.main()