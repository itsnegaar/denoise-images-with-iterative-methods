"""
***Explanatioin***

here we want to use iterative methods for image de-noising.
we will use these 3 methods:
- jacobian iterative method
- gauss seidel iterative method
- SOR  iterative method

 in each part, first i will briefly explain it mathematicaly, them we'll see the code and at the end the result on real world.

**import files and Load the images **
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, data, util
import time

img_path = '/home/negar/Downloads/img.png'
image = cv2.imread(img_path)


# make it square
def make_square(image):
    min_dim = min(image.shape[0], image.shape[1])
    start_row = (image.shape[0] - min_dim) // 2
    start_col = (image.shape[1] - min_dim) // 2
    image = image[start_row:start_row + min_dim, start_col:start_col + min_dim]
    return image


image = make_square(image)

# Display the original image
cv2.imshow('original', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

original_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def add_noise(image, sigma=0.1):
    noisy_image = util.random_noise(image, mode='gaussian', var=sigma ** 2)
    return np.clip(noisy_image, 0, 1)


noisy_image = add_noise(original_image)
print('noise added')

"""
for each method we need to calculate the min iteration required.
we claculate it based on spectral_radius and desired accuracy
k = log epsilon / log spectral radius M"""


def calculate_spectral_radius(matrix):
    eigenvalues, _ = np.linalg.eig(matrix)
    spectral_radius = max(np.abs(eigenvalues))
    return spectral_radius


def calculate_min_iterations_jacobian(image, epsilon):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    n = image.shape[0]  # Assuming A is a square matrix
    D = np.diag(np.diag(image))
    L = np.tril(image, k=-1)
    U = np.triu(image, k=1)

    M_jacobi = -np.linalg.inv(D).dot(L + U)
    spectral_radius_M = calculate_spectral_radius(M_jacobi)

    k = np.ceil(np.log(epsilon) / np.log(spectral_radius_M))
    return int(k)


def calculate_min_iterations_gauss_seidel(image, epsilon):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    n = image.shape[0]  # Assuming A is a square matrix
    D = np.diag(np.diag(image))
    L = np.tril(image, k=-1)
    U = np.triu(image, k=1)

    M_gs = -np.linalg.inv(L + D).dot(U)
    spectral_radius_M = calculate_spectral_radius(M_gs)

    k = np.ceil(np.log(epsilon) / np.log(spectral_radius_M))
    return int(k)


def calculate_min_iterations_sor(image, epsilon, omega):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    n = image.shape[0]
    D = np.diag(np.diag(image))
    L = np.tril(image, k=-1)
    U = np.triu(image, k=1)

    M_sor = -np.linalg.inv(omega * L + D).dot((omega - 1) * D + omega * U)
    spectral_radius_M = calculate_spectral_radius(M_sor)

    k = np.ceil(np.log(epsilon) / np.log(spectral_radius_M))
    return int(k)


epsilon = 1e-5  # desired accuracy

"""
also, to compare the result we need to find out which one has the least error.
so, we need to Calculate the residual error between the original and denoised images."""


def calculate_residual_error(original_image, denoised_image):
    if len(denoised_image.shape) == 3:
        denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_RGB2GRAY)
    absolute_difference = np.abs(original_image - denoised_image)
    mean_squared_error = np.mean(absolute_difference ** 2)
    root_mean_squared_error = np.sqrt(mean_squared_error)

    return root_mean_squared_error


"""***jacobian iterative method***


to solve AX=b in jacobian method, first, we write A like this:

A = D + L + U

D is diameeter of A,

L is anything under diameter.

U is anything upper the diemeter.

we proved that

DX(k+1) = -(L+U).X(k) + b

first we have a guess, x(0)


then, in each step we calculate x(k+1). based o what we said,

x(k+1) = -D-inverse.(L+U).x(k)+D-invrse(b)

here iteration matrix is Mj = -D-inverse.(L+U) and Cj = D-invrse(b)
"""

print('getting started with jacobian method')
start_time = time.time()


def jacobi_iteration(image, iterations):
    height, width = image.shape
    new_image = np.copy(image)

    for _ in range(iterations):
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                new_image[i, j] = 0.25 * (
                        image[i - 1, j] + image[i + 1, j] +
                        image[i, j - 1] + image[i, j + 1]
                )

    return new_image


k_jacobian = calculate_min_iterations_jacobian(np.array(image), epsilon=epsilon)
print("min iteration requiered for Jacobian: ", k_jacobian)

# Perform Jacobi iteration for image denoising
denoised_image_jacobian = jacobi_iteration(noisy_image, iterations=k_jacobian)

jacobian_error = calculate_residual_error(denoised_image_jacobian, image)
print("kacobian error: ", jacobian_error)

cv2.imshow('denoised_image_jacobian', denoised_image_jacobian)
cv2.waitKey(0)
cv2.destroyAllWindows()

end_time = time.time()
print("time requiered for jacobian method: ", end_time - start_time)

"""**gauss-sidel**

this method is similar to jacobian method, but in each step k+1, for updated values, we use the (k+1)th value.


if we write matrix A like this

A = L + D + U

then (L+D).x(k+1) = b - U.x(k) so,

x(k+1) = -(L+D)-inverse .U.x(k) + (L+D)-inverse.b

iteration matrix is Mgs = -(L+D)-inverse .U

Cgs = (L+D)-inverse.b
"""

print('getting started with gauss-sidel method')
start_time = time.time()


def gauss_seidel_iteration(image, iterations):
    height, width = image.shape
    new_image = np.copy(image)

    for _ in range(iterations):
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                new_image[i, j] = 0.25 * (
                        new_image[i - 1, j] + new_image[i + 1, j] +
                        new_image[i, j - 1] + new_image[i, j + 1]
                )

    return new_image


k_gauss_sidel = calculate_min_iterations_gauss_seidel(np.array(image), epsilon=epsilon)
print("min iteration requiered for gauss-sidel: ", k_gauss_sidel)

# Perform Gauss-Seidel iteration for image denoising
denoised_image_gs = gauss_seidel_iteration(noisy_image, iterations=k_gauss_sidel)

gs_error = calculate_residual_error(denoised_image_jacobian, image)
print("gauss-seidel error: ", gs_error)

cv2.imshow('denoised_image - gauss-seidel', denoised_image_gs)
cv2.waitKey(0)
cv2.destroyAllWindows()

end_time = time.time()
print("time requiered for gauss-seidel method: ", end_time - start_time)

"""**SOR**

The (SOR) method is a kind of relaxation methods. This method is a generalized or improved form of the Gauss–Seidel (GS) method and formed by adding a relaxation parameter. When the solution at th iteration is known then we can calculate the solution at kth iteration by applying the SOR method as follows Here,I(l+1) is the GS solution at (k+1)th iteration and w is the relaxation parameter, which is chosen in such a manner that it will accelerate the convergence of the method towards the solution.

x(k+1) = w.I(k+1) + (1-w.x(k))

this method is very similar to gauss-sidel, but we multiple a w to reach the answer faster. like this:

 (wL+D).x(k+1) = wb - ((1-w)D + wU).x(k) so,

x(k+1) = -(wL+D)-inverse .((w-1).D +wU).x(k) + w(wL+D)-inverse.b

Msor = -(wL+D)-inverse .((w-1).D +wU), Csor = w(wL+D)-inverse.b

"""

print('getting started with sor method')
start = time.time()


def sor_iteration(image, iterations, omega):
    height, width = image.shape
    new_image = np.copy(image)

    for _ in range(iterations):
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                new_image[i, j] = (1 - omega) * new_image[i, j] + omega * 0.25 * (
                        new_image[i - 1, j] + new_image[i + 1, j] +
                        new_image[i, j - 1] + new_image[i, j + 1]
                )

    return new_image


omega_values = [0.1, 1, 1.4, 2]  # Experiment with different omega values

for i, omega in enumerate(omega_values):
    k_sor = calculate_min_iterations_sor(np.array(image), epsilon=epsilon, omega=omega)
    print("min iteration requiered for gauss-sidel: ", k_sor)
    denoised_sor = sor_iteration(noisy_image, iterations=k_sor, omega=omega)
    sor_error = calculate_residual_error(denoised_image_jacobian, image)
    print(f"sor error for omega {omega}: ", jacobian_error)

    cv2.imshow(f"Denoised image, omega = {omega}", denoised_sor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_optimal_omega(A, max_iterations=100, epsilon=1e-5, omega_range=(0.5, 2.0, 0.01)):
    n = A.shape[0]  # Assuming A is a square matrix
    D = np.diag(np.diag(A))
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)

    omega_values = np.arange(omega_range[0], omega_range[1], omega_range[2])
    min_spectral_radius = float('inf')
    optimal_omega = 0

    for omega in omega_values:
        M_sor = -np.linalg.inv(omega * L + D).dot((omega - 1) * D + omega * U)
        spectral_radius_M = calculate_spectral_radius(M_sor)

        if spectral_radius_M < min_spectral_radius:
            min_spectral_radius = spectral_radius_M
            optimal_omega = omega

    return optimal_omega


omega_opt = find_optimal_omega(noisy_image)
print("optimal omega = ", omega_opt)
k_sor = calculate_min_iterations_sor(np.array(image), epsilon=epsilon, omega=omega_opt)
print("min iteration requiered for sor - opt omega: ", k_sor)
denoised_sor = sor_iteration(noisy_image, iterations=k_sor, omega=omega_opt)

sor_error = calculate_residual_error(denoised_image_jacobian, image)
print(f"sor error for omega{omega_opt}: ", sor_error)

cv2.imshow(f"Denoised image, omega = {omega_opt}", denoised_sor)
cv2.waitKey(0)
cv2.destroyAllWindows()

end_time = time.time()
print("time required for sor method: ", end_time - start_time)

"""**sources**

https://www.sciencedirect.com/science/article/pii/S0898122115001789
chapter-3
"""
