import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Class for the neural network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # Define the neural network architecture
        self.fc1 = nn.Linear(2, 50)  # Fully connected layer with 2 input features and 50 hidden units
        self.fc2 = nn.Linear(50, 1)  # Fully connected layer with 50 hidden units and 1 output

    def forward(self, t, x):
        # Forward pass of the neural network
        input_data = torch.cat((t, x), dim=1)  # Concatenate time and position as input
        out = torch.relu(self.fc1(input_data))  # Apply ReLU activation to the first layer
        out = self.fc2(out)  # Output layer
        return out

# Class for the diffusion problem
class DiffusionProblem:
    def __init__(self, num_points=100, lr=0.001):
        # Initialize the diffusion problem
        self.num_points = num_points  # Number of spatial and temporal points
        self.lr = lr  # Learning rate for training the PINN
        self.net = NeuralNet()  # Create an instance of the neural network
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)  # Adam optimizer

    def exact_solution(self, x, t):
        # Define the exact solution of the diffusion problem
        return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)

    """
        The exact_solution function is responsible for defining the exact solution of the 
    diffusion problem. This function calculates the exact temperature at a specific point
    in space (represented by x) and at a particular moment in time (represented by t) based
    on a mathematical equation that describes the physical behavior of the diffusion process.

    The equation used to compute the exact temperature is:
    u(x,t) = e^(-pi²*t)*sin(pi*x)
    where:
    x represents the spatial position where we want to calculate the temperature.
    t represents the time instant at which we want to determine the temperature.
    The exact_solution function applies this equation and returns the exact temperature value
    at the (x, t) point according to the mathematical model of diffusion. This exact solution 
    serves as a reference to evaluate the performance of the neural networks used in the code,
    allowing for a comparison between the network's predictions and the actual temperature values.
    
    """
    def generate_data(self):
        # Generate training data for the diffusion problem
        x = np.linspace(0, 1, self.num_points)  # Spatial positions
        t = np.linspace(0, 0.1, self.num_points)  # Time instances
        x, t = np.meshgrid(x, t)
        x = x.reshape(-1, 1)
        t = t.reshape(-1, 1)
        u_exact = self.exact_solution(x, t)

        # Convert data to PyTorch tensors
        self.x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        self.t = torch.tensor(t, dtype=torch.float32, requires_grad=True)
        self.u_exact = torch.tensor(u_exact, dtype=torch.float32)

    def train(self, num_epochs=10000):
        # Train the Physics-Informed Neural Network (PINN)
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            u_pred = self.net(self.t, self.x)

            du_dt = torch.autograd.grad(u_pred, self.t, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
            du_dx = torch.autograd.grad(u_pred, self.x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
            equation_lhs = du_dt - 0.01 * torch.pow(du_dx, 2)  # Equation of diffusion

            # Compute loss as a combination of the PDE residual and data fidelity term
            loss = torch.mean((equation_lhs)**2) + torch.mean((self.u_exact - u_pred)**2)

            loss.backward()
            self.optimizer.step()

            if epoch % 1000 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}')

    def evaluate(self):
        # Evaluate and visualize the results of the PINN
        with torch.no_grad():
            u_pred = self.net(self.t, self.x)

        x_numpy = self.x.detach().numpy()
        u_exact_numpy = self.u_exact.detach().numpy()
        u_pred_numpy = u_pred.detach().numpy()

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title('Exact Solution')
        plt.xlabel('Position (x)')
        plt.ylabel('Temperature (u)')
        plt.plot(x_numpy, u_exact_numpy, label='Exact Solution', linestyle='--')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.title('Approximated Solution (PINN)')
        plt.xlabel('Position (x)')
        plt.ylabel('Temperature (u)')
        plt.plot(x_numpy, u_pred_numpy, label='PINN Approximated Solution', linestyle='-')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def test_rnn(self):
        # Additional use of the trained neural network for testing purposes
        # For example, you can use it for different predictions or further analysis
        net = NeuralNet()  # Create a new instance of the neural network
        optimizer = optim.Adam(net.parameters(), lr=0.001)  # Reinitialize optimizer

        num_epochs = 10000
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            u_pred = net(self.t, self.x)

            loss = torch.mean((self.u_exact - u_pred)**2)

            loss.backward()
            optimizer.step()

            if epoch % 1000 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}')

        with torch.no_grad():
            u_pred = net(self.t, self.x)

        x_numpy = self.x.detach().numpy()
        u_exact_numpy = self.u_exact.detach().numpy()
        u_pred_numpy = u_pred.detach().numpy()

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.title('Exact Solution')
        plt.xlabel('Position (x)')
        plt.ylabel('Temperature (u)')
        plt.plot(x_numpy, u_exact_numpy, label='Exact Solution', linestyle='--')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.title('Approximated Solution (RNN)')
        plt.xlabel('Position (x)')
        plt.ylabel('Temperature (u)')
        plt.plot(x_numpy, u_pred_numpy, label='Neural Network Approximated Solution', linestyle='-')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

# Class for the inverse problem
class InverseProblem:
    def __init__(self, net, t, x, u_exact):
        self.net = net  # Trained neural network
        self.t = t  # Time instances
        self.x = x  # Spatial positions
        self.u_exact = u_exact  # Exact temperature distribution

    def loss_fn(self, D):
        # Define the loss function for estimating the diffusion coefficient (D)
        u_pred = self.net(self.t, self.x)
        residual = u_pred - self.u_exact
        loss = torch.mean(residual**2)
        return loss

    def optimize(self, num_iterations=1000, lr=0.01):
        # Optimize and estimate the diffusion coefficient (D)
        initial_guess = torch.tensor([0.1], dtype=torch.float32, requires_grad=True)
        optimizer = optim.Adam([initial_guess], lr=lr)

        for i in range(num_iterations):
            optimizer.zero_grad()
            loss = self.loss_fn(initial_guess)
            loss.backward()
            optimizer.step()

        estimated_D = initial_guess.item()
        print(f'Estimated Diffusion Coefficient: {estimated_D}')
        return estimated_D

if __name__ == "__main__":
    # Create an instance of the DiffusionProblem class
    diffusion_problem = DiffusionProblem()
    
    # Generate data and train the PINN to approximate the diffusion problem
    diffusion_problem.generate_data()
    diffusion_problem.train()
    
    # Evaluate and visualize the results of the PINN
    diffusion_problem.evaluate()

    # Create an instance of the InverseProblem class
    inverse_problem = InverseProblem(diffusion_problem.net, diffusion_problem.t, diffusion_problem.x, diffusion_problem.u_exact)
    
    # Optimize and estimate the diffusion coefficient (D) using the trained PINN
    estimated_D = inverse_problem.optimize()
    
    # Test the trained neural network (RNN) for additional purposes
    diffusion_problem.test_rnn()