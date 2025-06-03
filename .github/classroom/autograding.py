#!/usr/bin/env python3
"""Automated grading script for GitHub Classroom for BVP Project."""
import os
import sys
import json
import unittest
from io import StringIO

# Ensure the project's test directory can be found if script is run from .github/classroom
# This assumes the tests are in a structure like: WEEK_X_TOPIC/PROJECT_Y_NAME/tests/
# And this script is in WEEK_X_TOPIC/.github/classroom/
# We need to go up two levels from .github/classroom to reach WEEK_X_TOPIC,
# then into the specific project and its tests.

# Determine the project name dynamically or set it if fixed
# For this specific BVP project, it's PROJECT_1_SolveBVP
PROJECT_NAME = "PROJECT_1_SolveBVP"
PROJECT_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # WEEK_X_TOPIC
TEST_DIR = os.path.join(PROJECT_BASE_DIR, PROJECT_NAME, 'tests')

if not os.path.isdir(TEST_DIR):
    # Fallback if running from a different context (e.g. project root)
    # This might happen if autograding.json calls python .github/classroom/autograding.py
    # from the root of the student's repository.
    CURRENT_WORKING_DIR = os.getcwd()
    PROBABLE_PROJECT_DIR = os.path.join(CURRENT_WORKING_DIR, PROJECT_NAME)
    PROBABLE_TEST_DIR = os.path.join(PROBABLE_PROJECT_DIR, 'tests')
    if os.path.isdir(PROBABLE_TEST_DIR):
        TEST_DIR = PROBABLE_TEST_DIR
        # Add student's project root and solution path (if needed by tests for imports)
        sys.path.insert(0, PROBABLE_PROJECT_DIR) # For student's code
        sys.path.insert(0, os.path.join(PROBABLE_PROJECT_DIR, 'solution')) # For solution code
    else:
        # If still not found, print an error and exit, as tests cannot be run.
        print(f"Error: Test directory not found. Expected at {TEST_DIR} or {PROBABLE_TEST_DIR}")
        # Output a valid JSON for GitHub Classroom to indicate failure
        print(json.dumps({"tests": [], "feedback": "Autograder setup error: Test directory not found.", "status": "error"}))
        sys.exit(1)
else:
    # Add student's project root and solution path (if needed by tests for imports)
    sys.path.insert(0, os.path.join(PROJECT_BASE_DIR, PROJECT_NAME)) # For student's code
    sys.path.insert(0, os.path.join(PROJECT_BASE_DIR, PROJECT_NAME, 'solution')) # For solution code

# Now try to import the test module
# The test file is assumed to be test_solve_bvp.py
TEST_MODULE_NAME = "test_solve_bvp"

try:
    # Add TEST_DIR to sys.path so 'import test_solve_bvp' works
    sys.path.insert(0, TEST_DIR)
    import test_solve_bvp
except ImportError as e:
    print(f"Error: Could not import test module '{TEST_MODULE_NAME}' from {TEST_DIR}. Exception: {e}")
    # Output a valid JSON for GitHub Classroom
    print(json.dumps({"tests": [], "feedback": f"Autograder setup error: Could not import test module. {e}", "status": "error"}))
    sys.exit(1)

class PointsTestResult(unittest.TextTestResult):
    """A test result class that collects points based on test method names."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_points = {}
        self.total_points_earned = 0
        self.total_points_possible = 0
        self.detailed_results = []

    def startTest(self, test):
        super().startTest(test)
        # Extract points from test method name, e.g., test_something_10pts
        method_name = test._testMethodName
        parts = method_name.split('_')
        points = 0
        if parts[-1].endswith('pts'):
            try:
                points = int(parts[-1][:-3])
            except ValueError:
                points = 0 # Default if parsing fails
        self.current_test_name = method_name
        self.current_test_points_possible = points
        self.total_points_possible += points

    def addSuccess(self, test):
        super().addSuccess(test)
        self.test_points[self.current_test_name] = self.current_test_points_possible
        self.total_points_earned += self.current_test_points_possible
        self.detailed_results.append({
            "name": self.current_test_name,
            "status": "passed",
            "points_earned": self.current_test_points_possible,
            "points_possible": self.current_test_points_possible,
            "output": "Test passed."
        })

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.test_points[self.current_test_name] = 0
        # err is a tuple (type, value, traceback)
        error_message = self._exc_info_to_string(err, test)
        self.detailed_results.append({
            "name": self.current_test_name,
            "status": "failed",
            "points_earned": 0,
            "points_possible": self.current_test_points_possible,
            "output": error_message
        })

    def addError(self, test, err):
        super().addError(test, err)
        self.test_points[self.current_test_name] = 0
        error_message = self._exc_info_to_string(err, test)
        self.detailed_results.append({
            "name": self.current_test_name,
            "status": "error",
            "points_earned": 0,
            "points_possible": self.current_test_points_possible,
            "output": error_message
        })

class AutoGrader:
    def __init__(self, project_name):
        self.project_name = project_name
        self.results_data = {
            "tests": [],
            "feedback": "",
            "status": "success" # Default status
        }
        self.total_score = 0
        self.max_score = 0

    def run_project_tests(self):
        """Run tests for the specified project and collect results."""
        # Discover and run tests from the imported test module
        loader = unittest.TestLoader()
        # Load tests only from TestStudentImplementation class
        try:
            suite = loader.loadTestsFromTestCase(test_solve_bvp.TestStudentImplementation)
        except AttributeError as e:
            # This can happen if TestStudentImplementation is not defined in the test file
            # (e.g., if student code or solution code failed to import in test_solve_bvp.py)
            self.results_data["feedback"] = f"Autograder error: Could not load student tests. Test class 'TestStudentImplementation' might be missing or not loadable due to import errors in the test file itself. Details: {e}"
            self.results_data["status"] = "error"
            return
        except Exception as e:
            self.results_data["feedback"] = f"Autograder error: An unexpected error occurred while loading tests: {e}"
            self.results_data["status"] = "error"
            return

        if suite.countTestCases() == 0:
            # This might happen if student code is not available and tests are skipped by @unittest.skipIf
            # Or if the TestStudentImplementation class has no test methods.
            self.results_data["feedback"] = "No student tests were found or run. This might be due to missing student code, or the test class 'TestStudentImplementation' having no tests, or all tests being skipped."
            # Check if student code was actually available according to the test file's own checks
            if hasattr(test_solve_bvp, 'STUDENT_CODE_AVAILABLE') and not test_solve_bvp.STUDENT_CODE_AVAILABLE:
                 self.results_data["feedback"] += " The test file indicated that the student's code module could not be imported."
            self.results_data["status"] = "success" # Not an error, but 0 points
            return

        # Use a custom TestResult class to capture points
        # Redirect stdout to capture test runner output if needed, though PointsTestResult captures errors.
        # string_io = StringIO()
        # runner = unittest.TextTestRunner(stream=string_io, resultclass=PointsTestResult, verbosity=2)
        
        # Create an instance of our custom result class
        test_result_collector = PointsTestResult(stream=sys.stderr, descriptions=True, verbosity=2)
        
        suite.run(result=test_result_collector)

        self.total_score = test_result_collector.total_points_earned
        self.max_score = test_result_collector.total_points_possible
        self.results_data["tests"] = test_result_collector.detailed_results
        
        feedback_lines = [
            f"Grading for project: {self.project_name}",
            f"Total score: {self.total_score} / {self.max_score} points."
        ]
        if test_result_collector.wasSuccessful() and self.max_score > 0:
            feedback_lines.append("All graded tests passed!")
        elif self.max_score == 0:
             feedback_lines.append("No points were assigned to the tests that ran. Check test naming convention (e.g., test_method_Xpts)." )
        else:
            feedback_lines.append("Some tests failed. See details above or in the 'Checks' tab.")
        
        self.results_data["feedback"] = "\n".join(feedback_lines)
        if test_result_collector.errors or test_result_collector.failures:
            self.results_data["status"] = "failure" # Partial success or failure

    def generate_report(self):
        """Generate JSON report for GitHub Classroom."""
        # The actual output to GitHub Classroom is just a JSON string to stdout
        # Ensure that no other print statements interfere, unless they are for debugging and removed later.
        # The GitHub Classroom runner expects a specific JSON format.
        # A simple format: {"points": achieved_points}
        # A more complex format for autograding.json: {"tests": [...], "feedback": "..."}
        # We will use the more complex format.
        
        # For GitHub Classroom, the key is often 'points' or a structured output via autograding.json.
        # If using autograding.json, the output of this script is read by it.
        # If this script IS the autograding.json (by making it executable and pointing to it),
        # then it needs to output the full JSON structure.
        
        # Let's assume this script's output will be captured and processed, or it's directly used.
        # The structure used in self.results_data is suitable for the `education/autograding@v1` action.
        print(json.dumps(self.results_data, indent=2))

if __name__ == "__main__":
    # This script is intended to be run by GitHub Classroom's autograding environment.
    # It should output a JSON object to stdout that GitHub Classroom can parse.
    
    grader = AutoGrader(project_name=PROJECT_NAME)
    try:
        grader.run_project_tests()
    except Exception as e:
        # Catch-all for unexpected errors during the grading process itself
        grader.results_data["feedback"] = f"A critical error occurred during autograding: {e}. Please contact the instructor."
        grader.results_data["status"] = "error"
        # Ensure tests array is empty or reflects the error state
        grader.results_data["tests"] = [{
            "name": "Autograder Execution",
            "status": "error",
            "points_earned": 0,
            "points_possible": grader.max_score if grader.max_score > 0 else 100, # Assign some possible points if none were set
            "output": f"Critical error: {e}"
        }]
    finally:
        grader.generate_report()