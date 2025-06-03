"""Module: test_solve_bvp
Unit tests for BVP solvers: Finite Difference Method and scipy.solve_bvp wrapper.
File: test_solve_bvp.py

Tests the following functions:
1. solve_bvp_finite_difference - Finite difference method for BVP
2. ode_system_for_solve_bvp - ODE system for scipy.solve_bvp
3. boundary_conditions_for_solve_bvp - Boundary conditions for scipy.solve_bvp
4. solve_bvp_scipy - scipy.solve_bvp wrapper

BVP: y''(x) + sin(x)*y'(x) + exp(x)*y(x) = x^2
Boundary conditions: y(0) = 0, y(5) = 3
"""

import unittest
import numpy as np
import sys
import os
from scipy.integrate import solve_bvp

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLUTION_PATH = os.path.join(PROJECT_ROOT, 'solution')
STUDENT_PATH = PROJECT_ROOT

sys.path.insert(0, SOLUTION_PATH)
sys.path.insert(0, STUDENT_PATH)

# Import functions from reference solution
try:
    from finite_difference_bvp_solution import (
        solve_bvp_finite_difference as sol_solve_bvp_finite_difference,
        ode_system_for_solve_bvp as sol_ode_system_for_solve_bvp,
        boundary_conditions_for_solve_bvp as sol_boundary_conditions_for_solve_bvp,
        solve_bvp_scipy as sol_solve_bvp_scipy
    )
    SOLUTION_AVAILABLE = True
except ImportError as e:
    print(f"Error importing solution module: {e}. Tests against solution will be skipped.")
    SOLUTION_AVAILABLE = False

# Import functions from student's submission
try:
    from finite_difference_bvp_student import (
        solve_bvp_finite_difference as stu_solve_bvp_finite_difference,
        ode_system_for_solve_bvp as stu_ode_system_for_solve_bvp,
        boundary_conditions_for_solve_bvp as stu_boundary_conditions_for_solve_bvp,
        solve_bvp_scipy as stu_solve_bvp_scipy
    )
    STUDENT_CODE_AVAILABLE = True
except ImportError as e:
    print(f"Error importing student module: {e}. Student tests will be skipped or may fail.")
    STUDENT_CODE_AVAILABLE = False
except NotImplementedError:
    print("Student module imported, but some functions raise NotImplementedError.")
    STUDENT_CODE_AVAILABLE = True


@unittest.skipIf(not SOLUTION_AVAILABLE, "Reference solution not available, skipping solution validation tests.")
class TestReferenceSolution(unittest.TestCase):
    """Tests to validate the reference solution itself. (0 points for students)"""
    
    def setUp(self):
        self.n_points = 50
        self.rtol = 1e-3
        self.atol = 1e-4
        
        # Generate benchmark using scipy solve_bvp
        try:
            self.x_benchmark, self.y_benchmark = sol_solve_bvp_scipy()
            if np.any(np.isnan(self.y_benchmark)):
                raise ValueError("Benchmark solution contains NaN values.")
        except Exception as e:
            print(f"Failed to generate benchmark solution: {e}")
            self.x_benchmark, self.y_benchmark = None, None
    
    def test_solution_finite_difference_method(self):
        """Test reference solution finite difference method"""
        if self.y_benchmark is None:
            self.skipTest("Benchmark solution not available.")
        
        x_sol, y_sol = sol_solve_bvp_finite_difference(self.n_points)
        
        # Check boundary conditions
        self.assertAlmostEqual(y_sol[0], 0.0, places=6, msg="Left boundary condition not satisfied")
        self.assertAlmostEqual(y_sol[-1], 3.0, places=6, msg="Right boundary condition not satisfied")
        
        # Check solution shape
        self.assertEqual(len(x_sol), self.n_points + 2, msg="Incorrect number of grid points")
        self.assertEqual(len(y_sol), self.n_points + 2, msg="Incorrect number of solution points")
    
    def test_solution_scipy_bvp_wrapper(self):
        """Test reference solution scipy.solve_bvp wrapper"""
        if self.y_benchmark is None:
            self.skipTest("Benchmark solution not available.")
        
        x_sol, y_sol = sol_solve_bvp_scipy()
        
        # Check boundary conditions
        self.assertAlmostEqual(y_sol[0], 0.0, places=6, msg="Left boundary condition not satisfied")
        self.assertAlmostEqual(y_sol[-1], 3.0, places=6, msg="Right boundary condition not satisfied")
        
        # Check that solution is reasonable
        self.assertTrue(len(x_sol) > 0, msg="Empty solution array")
        self.assertTrue(len(y_sol) > 0, msg="Empty solution array")
        self.assertFalse(np.any(np.isnan(y_sol)), msg="Solution contains NaN values")
    
    def test_solution_ode_system(self):
        """Test reference solution ODE system function"""
        x_test = np.array([0.0, 1.0, 2.0])
        y_test = np.array([[0.1, 0.5, 1.0], [0.2, 0.3, 0.4]])  # [y, y']
        
        result = sol_ode_system_for_solve_bvp(x_test, y_test)
        
        # Check output shape
        self.assertEqual(result.shape, (2, 3), msg="ODE system output shape incorrect")
        
        # Check that first equation is dy/dx = y'
        np.testing.assert_allclose(result[0], y_test[1], rtol=1e-10, 
                                 err_msg="First ODE equation should be dy/dx = y'")
    
    def test_solution_boundary_conditions(self):
        """Test reference solution boundary conditions function"""
        ya_test = np.array([0.1, 0.5])  # [y(0), y'(0)]
        yb_test = np.array([2.9, -0.3])  # [y(5), y'(5)]
        
        result = sol_boundary_conditions_for_solve_bvp(ya_test, yb_test)
        
        # Check output shape
        self.assertEqual(result.shape, (2,), msg="Boundary conditions output shape incorrect")
        
        # Check boundary condition values
        expected = np.array([ya_test[0] - 0, yb_test[0] - 3])
        np.testing.assert_allclose(result, expected, rtol=1e-10,
                                 err_msg="Boundary conditions calculation incorrect")


@unittest.skipIf(not STUDENT_CODE_AVAILABLE, "Student code not available, skipping student tests.")
class TestStudentImplementation(unittest.TestCase):
    """Tests for the student's implementation. Graded tests."""
    
    def setUp(self):
        self.n_points = 30
        self.rtol = 1e-2
        self.atol = 1e-3
        
        if not SOLUTION_AVAILABLE:
            self.skipTest("Reference solution not available, cannot run student tests.")
        
        # Generate benchmark using solution
        try:
            self.x_benchmark, self.y_benchmark = sol_solve_bvp_scipy()
            if np.any(np.isnan(self.y_benchmark)):
                raise ValueError("Benchmark solution contains NaN values.")
        except Exception as e:
            self.fail(f"Failed to generate benchmark solution for student tests: {e}")
    
    def _run_student_function(self, func, *args, **kwargs):
        """Helper to run student function and catch NotImplementedError."""
        try:
            return func(*args, **kwargs)
        except NotImplementedError:
            self.fail(f"Student function {func.__name__} is not implemented.")
        except Exception as e:
            self.fail(f"Student function {func.__name__} raised an unexpected error: {e}")
    
    def test_student_solve_bvp_finite_difference_method_15pts(self):
        """Test student's solve_bvp_finite_difference (15 points)"""
        x_stu, y_stu = self._run_student_function(stu_solve_bvp_finite_difference, self.n_points)
        
        # Check boundary conditions
        self.assertAlmostEqual(y_stu[0], 0.0, places=5, 
                             msg="Student's finite difference method: left boundary condition not satisfied")
        self.assertAlmostEqual(y_stu[-1], 3.0, places=5,
                             msg="Student's finite difference method: right boundary condition not satisfied")
        
        # Check solution shape
        self.assertEqual(len(x_stu), self.n_points + 2, 
                        msg="Student's finite difference method: incorrect number of grid points")
        self.assertEqual(len(y_stu), self.n_points + 2,
                        msg="Student's finite difference method: incorrect number of solution points")
        
        # Check that solution is reasonable (no NaN, not all zeros)
        self.assertFalse(np.any(np.isnan(y_stu)), 
                        msg="Student's finite difference solution contains NaN values")
        self.assertFalse(np.allclose(y_stu[1:-1], 0.0), 
                        msg="Student's finite difference solution is trivially zero")
    
    def test_student_ode_system_for_solve_bvp_5pts(self):
        """Test student's ode_system_for_solve_bvp (5 points)"""
        x_test = np.array([0.0, 1.0, 2.0])
        y_test = np.array([[0.1, 0.5, 1.0], [0.2, 0.3, 0.4]])  # [y, y']
        
        student_result = self._run_student_function(stu_ode_system_for_solve_bvp, x_test, y_test)
        solution_result = sol_ode_system_for_solve_bvp(x_test, y_test)
        
        # Check output shape
        self.assertEqual(student_result.shape, solution_result.shape,
                        msg="Student's ODE system output shape is incorrect")
        
        # Check numerical accuracy
        np.testing.assert_allclose(student_result, solution_result, rtol=1e-5, atol=1e-6,
                                 err_msg="Student's ODE system output is incorrect")
    
    def test_student_boundary_conditions_for_solve_bvp_5pts(self):
        """Test student's boundary_conditions_for_solve_bvp (5 points)"""
        ya_test = np.array([0.1, 0.5])  # [y(0), y'(0)]
        yb_test = np.array([2.9, -0.3])  # [y(5), y'(5)]
        
        student_result = self._run_student_function(stu_boundary_conditions_for_solve_bvp, ya_test, yb_test)
        solution_result = sol_boundary_conditions_for_solve_bvp(ya_test, yb_test)
        
        # Check output shape
        self.assertEqual(student_result.shape, solution_result.shape,
                        msg="Student's boundary conditions output shape is incorrect")
        
        # Check numerical accuracy
        np.testing.assert_allclose(student_result, solution_result, rtol=1e-10, atol=1e-12,
                                 err_msg="Student's boundary conditions output is incorrect")
    
    def test_student_solve_bvp_scipy_wrapper_15pts(self):
        """Test student's solve_bvp_scipy wrapper (15 points)"""
        x_stu, y_stu = self._run_student_function(stu_solve_bvp_scipy)
        
        # Check boundary conditions
        self.assertAlmostEqual(y_stu[0], 0.0, places=5,
                             msg="Student's solve_bvp wrapper: left boundary condition not satisfied")
        self.assertAlmostEqual(y_stu[-1], 3.0, places=5,
                             msg="Student's solve_bvp wrapper: right boundary condition not satisfied")
        
        # Check that solution is reasonable
        self.assertTrue(len(x_stu) > 0, msg="Student's solve_bvp wrapper: empty solution array")
        self.assertTrue(len(y_stu) > 0, msg="Student's solve_bvp wrapper: empty solution array")
        self.assertFalse(np.any(np.isnan(y_stu)), 
                        msg="Student's solve_bvp wrapper: solution contains NaN values")
        
        # Compare with benchmark solution (interpolated to same grid)
        if len(x_stu) != len(self.x_benchmark):
            # Interpolate benchmark to student's grid
            y_benchmark_interp = np.interp(x_stu, self.x_benchmark, self.y_benchmark)
        else:
            y_benchmark_interp = self.y_benchmark
        
        # Check numerical accuracy with relaxed tolerance
        np.testing.assert_allclose(y_stu, y_benchmark_interp, rtol=self.rtol, atol=self.atol,
                                 err_msg="Student's solve_bvp wrapper solution is inaccurate")


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests based on availability
    if SOLUTION_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestReferenceSolution))
    else:
        print("Skipping TestReferenceSolution due to missing solution module.")
    
    if STUDENT_CODE_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestStudentImplementation))
    else:
        print("Skipping TestStudentImplementation due to missing student module.")
    
    # Run tests
    if suite.countTestCases() > 0:
        print(f"Running {suite.countTestCases()} tests...")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Output summary
        total_tests = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped)
        passed = total_tests - failures - errors - skipped
        
        print("\n--- Test Summary ---")
        print(f"Total tests run: {total_tests}")
        print(f"Passed: {passed}")
        print(f"Failures: {failures}")
        print(f"Errors: {errors}")
        print(f"Skipped: {skipped}")
        
        # Calculate approximate score (40 points total)
        if STUDENT_CODE_AVAILABLE and SOLUTION_AVAILABLE:
            if failures == 0 and errors == 0 and passed > 0:
                print("\nAll student tests passed successfully!")
                print("Estimated score: 40/40 points")
            else:
                print("\nSome student tests failed or encountered errors.")
                # Rough estimation based on passed tests
                student_tests_passed = sum(1 for test, _ in result.failures + result.errors 
                                         if 'TestStudentImplementation' not in str(test))
                estimated_score = max(0, (passed - student_tests_passed) * 10)  # Rough estimate
                print(f"Estimated score: {min(estimated_score, 40)}/40 points")
        elif not STUDENT_CODE_AVAILABLE:
            print("\nStudent code was not available. Score: 0/40 points")
        elif not SOLUTION_AVAILABLE:
            print("\nReference solution was not available. Cannot assess student work.")
    else:
        print("No tests were found or loaded. Check test script and file paths.")