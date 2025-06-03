import unittest
import numpy as np
from scipy.integrate import odeint
import sys
import os

# 添加项目路径以导入学生代码
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Attempt to import student's functions
try:
    from double_pendulum_simulation_student import (
    #from solution.double_pendulum_simulation_solution import (
        derivatives,
        solve_double_pendulum,
        calculate_energy,
        G_CONST,
        L_CONST,
        M_CONST
    )
    STUDENT_FUNCTIONS_IMPORTED = True
except ImportError:
    STUDENT_FUNCTIONS_IMPORTED = False
    # Define dummy functions if import fails, so tests can be defined
    # Tests will then fail gracefully due to NotImplementedError or incorrect results
    def derivatives(y, t, L1, L2, m1, m2, g):
        raise NotImplementedError("Student's derivatives function not found or import failed.")
    def solve_double_pendulum(initial_conditions, t_span, t_points, L_param, g_param):
        raise NotImplementedError("Student's solve_double_pendulum function not found or import failed.")
    def calculate_energy(sol_arr, L_param, m_param, g_param):
        raise NotImplementedError("Student's calculate_energy function not found or import failed.")
    G_CONST, L_CONST, M_CONST = 9.81, 0.4, 1.0

# Solution functions (copied from solution.py for self-contained testing)
# This is to ensure tests compare against a known correct implementation.

def derivatives_solution(y, t, L1, L2, m1, m2, g):
    theta1, omega1, theta2, omega2 = y
    dtheta1_dt = omega1
    dtheta2_dt = omega2
    num1 = -omega1**2 * np.sin(2*theta1 - 2*theta2) \
           - 2 * omega2**2 * np.sin(theta1 - theta2) \
           - (g/L1) * (np.sin(theta1 - 2*theta2) + 3*np.sin(theta1))
    den1 = 3 - np.cos(2*theta1 - 2*theta2) 
    domega1_dt = num1 / den1
    num2 = 4 * omega1**2 * np.sin(theta1 - theta2) \
           + omega2**2 * np.sin(2*theta1 - 2*theta2) \
           + 2 * (g/L1) * (np.sin(2*theta1 - theta2) - np.sin(theta2))
    den2 = 3 - np.cos(2*theta1 - 2*theta2)
    domega2_dt = num2 / den2
    return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]

def solve_double_pendulum_solution(initial_conditions, t_span, t_points, L_param, g_param, m_param):
    y0 = [initial_conditions['theta1'], initial_conditions['omega1'], 
          initial_conditions['theta2'], initial_conditions['omega2']]
    t_arr = np.linspace(t_span[0], t_span[1], t_points)
    sol_arr = odeint(derivatives_solution, y0, t_arr, args=(L_param, L_param, m_param, m_param, g_param), rtol=1e-8, atol=1e-8)
    return t_arr, sol_arr

def calculate_energy_solution(sol_arr, L_param, m_param, g_param):
    theta1, omega1, theta2, omega2 = sol_arr[:, 0], sol_arr[:, 1], sol_arr[:, 2], sol_arr[:, 3]
    V = -m_param * g_param * L_param * (2 * np.cos(theta1) + np.cos(theta2))
    T = m_param * L_param**2 * (omega1**2 + 0.5 * omega2**2 + omega1 * omega2 * np.cos(theta1 - theta2))
    return T + V

class TestDoublePendulum(unittest.TestCase):
    """Test suite for Project 1: Double Pendulum Simulation"""

    @classmethod
    def setUpClass(cls):
        cls.assertTrue(STUDENT_FUNCTIONS_IMPORTED, "Student functions could not be imported. Check file name and function names.")
        
        cls.L = L_CONST
        cls.g = G_CONST
        cls.m = M_CONST

        cls.initial_conditions_rad = {
            'theta1': np.pi/2, 'omega1': 0.0,
            'theta2': np.pi/2, 'omega2': 0.0
        }
        cls.t_span_short = (0, 1) # Short time for derivative and basic solve tests
        cls.t_points_short = 101
        
        cls.t_span_long = (0, 10) # Longer time for energy conservation test (not 100s to keep tests fast)
        cls.t_points_long = 1001

        # Generate solution data once for comparison
        cls.t_sol_short, cls.sol_short_solution = solve_double_pendulum_solution(
            cls.initial_conditions_rad, cls.t_span_short, cls.t_points_short, cls.L, cls.g, cls.m
        )
        cls.energy_short_solution = calculate_energy_solution(cls.sol_short_solution, cls.L, cls.m, cls.g)

        cls.t_sol_long, cls.sol_long_solution = solve_double_pendulum_solution(
            cls.initial_conditions_rad, cls.t_span_long, cls.t_points_long, cls.L, cls.g, cls.m
        )
        cls.energy_long_solution = calculate_energy_solution(cls.sol_long_solution, cls.L, cls.m, cls.g)

    def test_01_derivatives_function_exists_2pts(self):
        """(2 points) Test if derivatives function is implemented."""
        try:
            # Test with a sample state
            y_sample = [np.pi/4, 0.1, np.pi/3, -0.2]
            t_sample = 0.5
            student_derivs = derivatives(y_sample, t_sample, self.L, self.L, self.m, self.m, self.g)
            self.assertIsInstance(student_derivs, (list, np.ndarray), "Derivatives should return a list or numpy array.")
            self.assertEqual(len(student_derivs), 4, "Derivatives list/array should have 4 elements.")
        except NotImplementedError:
            self.fail("derivatives function is not implemented or raised NotImplementedError.")
        except Exception as e:
            self.fail(f"derivatives function raised an unexpected error: {e}")

    def test_02_derivatives_calculation_13pts(self):
        """(13 points) Test correctness of derivatives calculation at specific points."""
        test_cases = [
            ([np.pi/2, 0, np.pi/2, 0], 0), # Initial state
            ([0, 0, 0, 0], 0),             # Equilibrium (unstable for pi/2, stable for 0)
            ([np.pi/4, 0.5, -np.pi/4, -0.5], 1.0)
        ]
        for y_test, t_test in test_cases:
            with self.subTest(y=y_test, t=t_test):
                try:
                    student_derivs = np.array(derivatives(y_test, t_test, self.L, self.L, self.m, self.m, self.g))
                    solution_derivs = np.array(derivatives_solution(y_test, t_test, self.L, self.L, self.m, self.m, self.g))
                    np.testing.assert_allclose(student_derivs, solution_derivs, rtol=1e-5, atol=1e-7,
                                                err_msg="Student's derivatives calculation is incorrect.")
                except NotImplementedError:
                    self.fail("derivatives function is not implemented.")
                except Exception as e:
                    self.fail(f"derivatives raised an error: {e}") 

    def test_03_solve_double_pendulum_exists_5pts(self):
        """(5 points) Test if solve_double_pendulum function is implemented and returns correct types."""
        try:
            t_student, sol_student = solve_double_pendulum(
                self.initial_conditions_rad, self.t_span_short, self.t_points_short, self.L, self.g
            )
            self.assertIsInstance(t_student, np.ndarray, "Time array should be a numpy array.")
            self.assertIsInstance(sol_student, np.ndarray, "Solution array should be a numpy array.")
            self.assertEqual(t_student.ndim, 1, "Time array should be 1D.")
            self.assertEqual(sol_student.ndim, 2, "Solution array should be 2D.")
            self.assertEqual(len(t_student), self.t_points_short, "Time array length mismatch.")
            self.assertEqual(sol_student.shape, (self.t_points_short, 4), "Solution array shape mismatch.")
        except NotImplementedError:
            self.fail("solve_double_pendulum function is not implemented.")
        except Exception as e:
            self.fail(f"solve_double_pendulum raised an unexpected error: {e}")

    def test_04_solve_double_pendulum_accuracy_20pts(self):
        """(20 points) Test accuracy of solve_double_pendulum solution against reference."""
        try:
            t_student, sol_student = solve_double_pendulum(
                self.initial_conditions_rad, self.t_span_short, self.t_points_short, self.L, self.g
            )
            # Compare a few key points (start, mid, end) and overall trajectory
            np.testing.assert_allclose(sol_student[0], self.sol_short_solution[0], rtol=1e-5, atol=1e-7, err_msg="Mismatch at t=0")
            np.testing.assert_allclose(sol_student[self.t_points_short//2], self.sol_short_solution[self.t_points_short//2], rtol=1e-4, atol=1e-5, err_msg="Mismatch at t_mid") # Looser tolerance for mid-point due to divergence
            np.testing.assert_allclose(sol_student[-1], self.sol_short_solution[-1], rtol=1e-3, atol=1e-4, err_msg="Mismatch at t_end") # Even looser for end due to chaotic nature
            
            # Check overall RMS error for theta1 as a proxy for trajectory similarity
            # This is a tricky test for chaotic systems. We use a short interval.
            rms_error_theta1 = np.sqrt(np.mean((sol_student[:, 0] - self.sol_short_solution[:, 0])**2))
            self.assertLess(rms_error_theta1, 0.01, "RMS error for theta1 trajectory is too high for the short interval.")

        except NotImplementedError:
            self.fail("solve_double_pendulum function is not implemented.")
        except Exception as e:
            self.fail(f"solve_double_pendulum raised an error during accuracy test: {e}")

    def test_05_calculate_energy_exists_5pts(self):
        """(5 points) Test if calculate_energy function is implemented and returns correct type."""
        try:
            # Use solution data as input for this test
            energy_student = calculate_energy(self.sol_short_solution, self.L, self.m, self.g)
            self.assertIsInstance(energy_student, np.ndarray, "Energy array should be a numpy array.")
            self.assertEqual(energy_student.ndim, 1, "Energy array should be 1D.")
            self.assertEqual(len(energy_student), len(self.sol_short_solution), "Energy array length mismatch.")
        except NotImplementedError:
            self.fail("calculate_energy function is not implemented.")
        except Exception as e:
            self.fail(f"calculate_energy raised an unexpected error: {e}")

    def test_06_calculate_energy_accuracy_15pts(self):
        """(15 points) Test accuracy of calculate_energy calculation."""
        try:
            # Use solution data as input to isolate energy calculation errors
            energy_student = calculate_energy(self.sol_short_solution, self.L, self.m, self.g)
            np.testing.assert_allclose(energy_student, self.energy_short_solution, rtol=1e-5, atol=1e-7,
                                        err_msg="Student's energy calculation is incorrect.")
        except NotImplementedError:
            self.fail("calculate_energy function is not implemented.")
        except Exception as e:
            self.fail(f"calculate_energy raised an error during accuracy test: {e}")

    def test_07_energy_conservation_10pts(self):
        """(10 points) Test energy conservation from student's full solution pipeline (long duration)."""
        # This test uses the student's solve_double_pendulum and calculate_energy
        # The goal is to check if their ODE solving parameters (rtol/atol or t_points) are sufficient
        # for the energy conservation requirement stated in the problem (variation < 1e-5 J over 100s).
        # Here we test for 10s, and expect proportionally good conservation.
        # Expected variation for 10s might be ~1e-6 if 100s is 1e-5, but this scaling isn't always linear.
        # We will use a slightly relaxed tolerance for this automated test over 10s.
        target_energy_variation_10s = 5e-6 # Adjusted for 10s test duration

        try:
            t_student_long, sol_student_long = solve_double_pendulum(
                self.initial_conditions_rad, self.t_span_long, self.t_points_long, self.L, self.g
            )
            energy_student_long = calculate_energy(sol_student_long, self.L, self.m, self.g)
            
            energy_variation_student = np.max(energy_student_long) - np.min(energy_student_long)
            self.assertLess(energy_variation_student, target_energy_variation_10s,
                            f"Energy variation ({energy_variation_student:.2e} J) over {self.t_span_long[1]}s exceeds target ({target_energy_variation_10s:.1e} J). "
                            f"Ensure rtol/atol in odeint are small enough (e.g., 1e-7 or 1e-8) or t_points is sufficient.")
        except NotImplementedError:
            self.fail("solve_double_pendulum or calculate_energy is not implemented.")
        except Exception as e:
            self.fail(f"Error during energy conservation test: {e}")

if __name__ == '__main__':
    # You can run the tests from the command line using:
    # python -m unittest test_double_pendulum_simulation.py
# Ensure that double_pendulum_simulation_student.py is in the same directory or accessible via PYTHONPATH.
    
    # For GitHub Classroom autograding, the runner will discover and run tests.
    # This block is mainly for local testing convenience.
    if not STUDENT_FUNCTIONS_IMPORTED:
        print("Failed to import student functions. Tests will likely fail or be skipped.")
        print("Ensure 'double_pendulum_simulation_student.py' is in the same directory as 'test_double_pendulum_simulation.py' or in PYTHONPATH.")
    
    unittest.main(verbosity=2)