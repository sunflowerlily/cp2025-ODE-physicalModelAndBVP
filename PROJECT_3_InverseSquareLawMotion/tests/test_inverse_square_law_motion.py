"""
Unit tests for inverse_square_law_motion.py
File: test_inverse_square_law_motion.py
Author: Trae AI
Date: $(date +%Y-%m-%d)
"""
import unittest
import numpy as np
import sys
import os

# Add the solution directory to the Python path to import the solution module
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'solution')))
# from inverse_square_law_motion_solution import solve_orbit, calculate_energy, calculate_angular_momentum, derivatives, GM

# It's safer to define a fixed path for the solution for consistent testing
SOLUTION_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'solution'))
STUDENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# For autograding, we should test STUDENT code, not solution code
# Only import student code for actual grading

# Placeholder for functions if student module is not found or has errors
def solve_orbit_placeholder(*args, **kwargs):
    raise NotImplementedError("solve_orbit is not implemented or import failed.")

def calculate_energy_placeholder(*args, **kwargs):
    raise NotImplementedError("calculate_energy is not implemented or import failed.")

def calculate_angular_momentum_placeholder(*args, **kwargs):
    raise NotImplementedError("calculate_angular_momentum is not implemented or import failed.")

def derivatives_placeholder(*args, **kwargs):
    raise NotImplementedError("derivatives is not implemented or import failed.")

GM_default = 1.0

# Import student code for grading
try:
    sys.path.insert(0, STUDENT_PATH)
    from inverse_square_law_motion_student import (
        solve_orbit as student_solve_orbit,
        calculate_energy as student_calculate_energy,
        calculate_angular_momentum as student_calculate_angular_momentum,
        derivatives as student_derivatives,
    )
    solve_orbit_to_test = student_solve_orbit
    calculate_energy_to_test = student_calculate_energy
    calculate_angular_momentum_to_test = student_calculate_angular_momentum
    derivatives_to_test = student_derivatives
    GM_tested = GM_default
    print(f"Successfully imported from STUDENT: {STUDENT_PATH}")
except ImportError as e:
    print(f"Could not import from student: {e}")
    print("Using placeholders. Tests will fail due to NotImplementedError.")
    solve_orbit_to_test = solve_orbit_placeholder
    calculate_energy_to_test = calculate_energy_placeholder
    calculate_angular_momentum_to_test = calculate_angular_momentum_placeholder
    derivatives_to_test = derivatives_placeholder
    GM_tested = GM_default

# Optional: Import solution for validation (only for development/debugging)
# This should be commented out or removed for actual student grading
# try:
#     sys.path.insert(0, SOLUTION_PATH)
#     from inverse_square_law_motion_solution import (
#         solve_orbit as solution_solve_orbit,
#         calculate_energy as solution_calculate_energy,
#         calculate_angular_momentum as solution_calculate_angular_momentum,
#         derivatives as solution_derivatives,
#         GM as SOLUTION_GM
#     )
#     print(f"Solution also available for validation: {SOLUTION_PATH}")
# except ImportError:
#     print("Solution not available (this is normal for student grading)")

class TestInverseSquareLawMotion(unittest.TestCase):
    def setUp(self):
        """Set up common parameters for tests."""
        self.gm = GM_tested # Use the GM from the imported module or default
        self.t_start = 0
        self.n_points = 200 # Fewer points for faster tests
        self.rtol_conservation = 1e-5 # Relative tolerance for conservation checks
        self.atol_conservation = 1e-7 # Absolute tolerance for conservation checks
        self.mass_particle = 1.0 # Assume m=1 for specific energy/angular momentum calculations

    def test_01_derivatives_function_5pts(self):
        """Test the derivatives function for a simple case (5 points)."""
        try:
            state = [1.0, 0.0, 0.0, 1.0] # x, y, vx, vy
            # Expected: ax = -GM*x/r^3 = -GM*1/1^3 = -GM; ay = -GM*y/r^3 = 0
            expected_derivatives = [0.0, 1.0, -self.gm, 0.0] # vx, vy, ax, ay
            # Note: derivatives_to_test might be the student's or solution's version
            actual_derivatives = derivatives_to_test(0, state, self.gm)
            np.testing.assert_allclose(actual_derivatives, expected_derivatives, rtol=1e-7, atol=1e-9,
                                     err_msg="Derivatives calculation is incorrect.")
        except NotImplementedError:
            self.fail("derivatives function not implemented by student.")
        except Exception as e:
            self.fail(f"Derivatives function raised an unexpected error: {e}")

    def test_02_circular_orbit_energy_conservation_10pts(self):
        """Test energy conservation for a circular orbit (10 points)."""
        # For circular orbit at r=1, v = sqrt(GM/r) = sqrt(GM)
        r0 = 1.0
        v0 = np.sqrt(self.gm / r0)
        initial_conditions = [r0, 0.0, 0.0, v0]
        t_end = 2 * np.pi * r0 / v0 * 2 # Two periods
        t_eval = np.linspace(self.t_start, t_end, self.n_points)
        
        try:
            sol = solve_orbit_to_test(initial_conditions, (self.t_start, t_end), t_eval, gm_val=self.gm)
            states = sol.y.T # Transpose to get (n_points, n_variables)
            energies = calculate_energy_to_test(states, self.gm, self.mass_particle)
            initial_energy = energies[0]
            final_energy = energies[-1]
            # Check that energy is conserved throughout the orbit
            np.testing.assert_allclose(energies, initial_energy, 
                                     rtol=self.rtol_conservation, atol=self.atol_conservation,
                                     err_msg="Energy not conserved in circular orbit.")
        except NotImplementedError:
            self.fail("solve_orbit or calculate_energy not implemented by student.")
        except Exception as e:
            self.fail(f"Circular orbit test raised an unexpected error: {e}")

    def test_03_circular_orbit_angular_momentum_conservation_10pts(self):
        """Test angular momentum conservation for a circular orbit (10 points)."""
        r0 = 1.0
        v0 = np.sqrt(self.gm / r0)
        initial_conditions = [r0, 0.0, 0.0, v0]
        t_end = 2 * np.pi * r0 / v0 * 2 # Two periods
        t_eval = np.linspace(self.t_start, t_end, self.n_points)

        try:
            sol = solve_orbit_to_test(initial_conditions, (self.t_start, t_end), t_eval, gm_val=self.gm)
            states = sol.y.T
            angular_momenta = calculate_angular_momentum_to_test(states, self.mass_particle)
            initial_Lz = angular_momenta[0]
            # Check that angular momentum is conserved
            np.testing.assert_allclose(angular_momenta, initial_Lz, 
                                     rtol=self.rtol_conservation, atol=self.atol_conservation,
                                     err_msg="Angular momentum not conserved in circular orbit.")
        except NotImplementedError:
            self.fail("solve_orbit or calculate_angular_momentum not implemented by student.")
        except Exception as e:
            self.fail(f"Circular orbit angular momentum test raised an unexpected error: {e}")

    def test_04_elliptical_orbit_properties_10pts(self):
        """Test basic properties of an elliptical orbit (E < 0) (10 points)."""
        # Initial conditions for an elliptical orbit (e.g., x0=1, y0=0, vx0=0, vy0=0.8 for GM=1)
        ic_ellipse = [1.0, 0.0, 0.0, 0.8 * np.sqrt(self.gm)] # Scale vy0 with sqrt(GM) for consistency
        t_end_ellipse = 20 / np.sqrt(self.gm) # Scale time for GM
        t_eval_ellipse = np.linspace(self.t_start, t_end_ellipse, self.n_points * 2) # More points for ellipse

        try:
            sol = solve_orbit_to_test(ic_ellipse, (self.t_start, t_end_ellipse), t_eval_ellipse, gm_val=self.gm)
            states = sol.y.T
            energies = calculate_energy_to_test(states, self.gm, self.mass_particle)
            self.assertTrue(np.all(energies < 0), msg="Energy for an elliptical orbit should be negative.")
            
            # Check if the orbit is bound (particle does not escape to infinity)
            # A simple check: max distance should not grow indefinitely. For a stable ellipse, it's bounded.
            distances = np.sqrt(states[:,0]**2 + states[:,1]**2)
            self.assertTrue(np.max(distances) < 10 * ic_ellipse[0] if ic_ellipse[0] > 0 else 10,
                            msg="Particle seems to escape in what should be an elliptical orbit.")
        except NotImplementedError:
            self.fail("Elliptical orbit test functions not implemented by student.")
        except Exception as e:
            self.fail(f"Elliptical orbit test raised an unexpected error: {e}")

    def test_05_parabolic_orbit_properties_5pts(self):
        """Test basic properties of a parabolic orbit (E approx 0) (5 points)."""
        # For E=0, v_escape = sqrt(2*GM/r). If x0=1, y0=0, then vy0 = sqrt(2*GM/1)
        r0 = 1.0
        vy0_parabolic = np.sqrt(2 * self.gm / r0)
        ic_parabola = [r0, 0.0, 0.0, vy0_parabolic]
        t_end_parabola = 10 / np.sqrt(self.gm) # Scale time
        t_eval_parabola = np.linspace(self.t_start, t_end_parabola, self.n_points)

        try:
            sol = solve_orbit_to_test(ic_parabola, (self.t_start, t_end_parabola), t_eval_parabola, gm_val=self.gm)
            states = sol.y.T
            energies = calculate_energy_to_test(states, self.gm, self.mass_particle)
            # Energy should be very close to zero
            np.testing.assert_allclose(energies, 0, 
                                     rtol=self.rtol_conservation, atol=1e-5, # Slightly larger atol for E=0 check
                                     err_msg="Energy for a parabolic orbit should be approximately zero.")
        except NotImplementedError:
            self.fail("Parabolic orbit test functions not implemented by student.")
        except Exception as e:
            self.fail(f"Parabolic orbit test raised an unexpected error: {e}")

    def test_06_hyperbolic_orbit_properties_5pts(self):
        """Test basic properties of a hyperbolic orbit (E > 0) (5 points)."""
        r0 = 1.0
        vy0_hyperbolic = np.sqrt(2.5 * self.gm / r0) # Speed greater than escape velocity
        ic_hyperbola = [r0, 0.0, 0.0, vy0_hyperbolic]
        t_end_hyperbola = 5 / np.sqrt(self.gm)
        t_eval_hyperbola = np.linspace(self.t_start, t_end_hyperbola, self.n_points)

        try:
            sol = solve_orbit_to_test(ic_hyperbola, (self.t_start, t_end_hyperbola), t_eval_hyperbola, gm_val=self.gm)
            states = sol.y.T
            energies = calculate_energy_to_test(states, self.gm, self.mass_particle)
            self.assertTrue(np.all(energies > 0), msg="Energy for a hyperbolic orbit should be positive.")
        except NotImplementedError:
            self.fail("Hyperbolic orbit test functions not implemented by student.")
        except Exception as e:
            self.fail(f"Hyperbolic orbit test raised an unexpected error: {e}")

    # Reference solution validation test is disabled for student grading
    # This test would only be used during development to validate the test suite itself
    @unittest.skip("Reference solution validation disabled for student grading")
    def test_reference_solution_passes_all_checks(self):
        """Internal check: Ensure reference solution passes its own tests (0 points - validation)."""
        # This test is skipped during student grading
        # It would only be used during development to validate the test suite
        pass

if __name__ == '__main__':
    # This allows running the tests from the command line
    # Ensure the correct path is set for imports when running directly
    # If running this file directly, the initial imports at the top of the file handle path adjustments.
    print(f"Running tests for InverseSquareLawMotion...")
    print(f"GM being used in tests: {GM_tested}")
    print(f"solve_orbit_to_test: {solve_orbit_to_test.__name__ if hasattr(solve_orbit_to_test, '__name__') else 'placeholder'}") 
    unittest.main()