import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Defina a rede neural para aproximar a solução
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(2, 50)  # Duas entradas: tempo e posição
        self.fc2 = nn.Linear(50, 1)  # Saída única para a temperatura

    def forward(self, t, x):
        input_data = torch.cat((t, x), dim=1)
        out = torch.relu(self.fc1(input_data))
        out = self.fc2(out)
        return out

# Função para calcular a solução exata da equação de difusão
def exact_solution(x, t):
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)

# Dados de treinamento
num_points = 100
x = np.linspace(0, 1, num_points)  # Posições
t = np.linspace(0, 0.1, num_points)  # Tempos
x, t = np.meshgrid(x, t)
x = x.reshape(-1, 1)
t = t.reshape(-1, 1)
u_exact = exact_solution(x, t)

# Converter dados para tensores PyTorch
x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
t = torch.tensor(t, dtype=torch.float32, requires_grad=True)
u_exact = torch.tensor(u_exact, dtype=torch.float32)

# Inicializar a rede neural e otimizador
net = NeuralNet()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Treinamento do PINN
num_epochs = 10000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    u_pred = net(t, x)

    # Equação de difusão (LHS da equação)
    du_dt = torch.autograd.grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    du_dx = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    equation_lhs = du_dt - 0.01 * torch.pow(du_dx, 2)

    # Erro médio quadrático
    loss = torch.mean((equation_lhs)**2) + torch.mean((u_exact - u_pred)**2)

    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}')

# Avaliação do modelo
with torch.no_grad():
    u_pred = net(t, x)

# Converte os tensores em arrays NumPy usando .detach().numpy()
x_numpy = x.detach().numpy()
u_exact_numpy = u_exact.detach().numpy()
u_pred_numpy = u_pred.detach().numpy()

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.title('Solução Exata')
plt.xlabel('Posição (x)')
plt.ylabel('Temperatura (u)')
plt.plot(x_numpy, u_exact_numpy, label='Solução Exata', linestyle='--')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.title('Solução Aproximada')
plt.xlabel('Posição (x)')
plt.ylabel('Temperatura (u)')
plt.plot(x_numpy, u_pred_numpy, label='Solução Aproximada da PINN', linestyle='-')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Inversão para estimar o coeficiente de difusão D
def inverse_problem(x, t, u_exact, u_pred):
    # Defina a função de perda para a inversão
    def loss_fn(D):
        u_pred = net(t, x)
        residual = u_pred - u_exact
        loss = torch.mean(residual**2)
        return loss

    # Otimização para encontrar o coeficiente de difusão D
    initial_guess = torch.tensor([0.1], dtype=torch.float32, requires_grad=True)  # Valor inicial como tensor
    optimizer = optim.Adam([initial_guess], lr=0.01)
    num_iterations = 1000

    for i in range(num_iterations):
        optimizer.zero_grad()
        loss = loss_fn(initial_guess)
        loss.backward()
        optimizer.step()
    print(f'Iteration {i}, Loss: {loss.item()}, D: {initial_guess.item()}')

    estimated_D = initial_guess.item()
    return estimated_D

# Executar a inversão
estimated_D = inverse_problem(x, t, u_exact, u_pred)
print(f'Coeficiente de Difusão Estimado: {estimated_D}')

net = NeuralNet()
optimizer = optim.Adam(net.parameters(), lr=0.001)

num_epochs = 10000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    u_pred = net(t, x)

    # Perda Mean Squared Error (MSE) entre a solução exata e a aproximada
    loss = torch.mean((u_exact - u_pred)**2)

    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}')

# Avaliação do modelo
with torch.no_grad():
    u_pred = net(t, x)

# Converte os tensores em arrays NumPy usando .detach().numpy()
x_numpy = x.detach().numpy()
u_exact_numpy = u_exact.detach().numpy()
u_pred_numpy = u_pred.detach().numpy()

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.title('Solução Exata')
plt.xlabel('Posição (x)')
plt.ylabel('Temperatura (u)')
plt.plot(x_numpy, u_exact_numpy, label='Solução Exata', linestyle='--')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.title('Solução Aproximada')
plt.xlabel('Posição (x)')
plt.ylabel('Temperatura (u)')
plt.plot(x_numpy, u_pred_numpy, label='Solução Aproximada da Rede Neural', linestyle='-')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

