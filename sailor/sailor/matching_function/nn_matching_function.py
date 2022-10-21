
import torch
from torch.autograd import Variable
from typing import List


class LinearRegressionModel(torch.nn.Module):

    def __init__(self):
        super(LinearRegressionModel, self).__init__()

        self.layers = [
            torch.nn.Linear(5, 4), torch.nn.ReLU(),
            torch.nn.Linear(4, 4), torch.nn.ReLU(),
            torch.nn.Linear(4, 3), torch.nn.ReLU(),
            torch.nn.Linear(3, 3), torch.nn.ReLU(),
            torch.nn.Linear(3, 2), torch.nn.ReLU(),
            torch.nn.Linear(2, 2), torch.nn.ReLU(),
            torch.nn.Linear(2, 1), torch.nn.ReLU()
        ]
        self.last_layer = torch.nn.Linear(1, 1)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return self.last_layer(x)


class NnMatchingFunction:

    def __init__(self) -> None:
        self.model = LinearRegressionModel()

        self.criterion = torch.nn.MSELoss(size_average=False)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def train(self, data: List[float], label: int, epochs: int = 10) -> None:
        x_data = Variable(torch.Tensor([data]))
        y_data = Variable(torch.Tensor([[label]]))

        for _ in range(epochs):
            pred_y = self.model(x_data)
            loss = self.criterion(pred_y, y_data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def match(self,
              class_similarity: float,
              color_histogram_similarity: float,
              position_similarity: float,
              size_similarity: float,
              last_time_seen_similarity: float
              ) -> float:

        new_var = Variable(torch.Tensor(
            [[class_similarity,
              color_histogram_similarity,
              position_similarity,
              size_similarity,
              last_time_seen_similarity]]))
        return self.model(new_var).item()
