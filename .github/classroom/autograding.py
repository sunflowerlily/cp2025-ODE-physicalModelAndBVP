#!/usr/bin/env python3
"""Automated grading script for GitHub Classroom for ODE Physical Model and BVP Projects."""
import os
import sys
import json
import unittest
import importlib.util
from io import StringIO

# Project configuration
PROJECTS = [
    {
        "name": "PROJECT_1_FiniteDifferenceBVP",
        "test_module": "test_finite_difference_bvp",
        "test_class": "TestStudentImplementation"
    },
    {
        "name": "PROJECT_2_ShootingMethod", 
        "test_module": "test_shooting_method",
        "test_class": "TestStudentImplementation"
    },
    {
        "name": "PROJECT_3_InverseSquareLawMotion",
        "test_module": "test_inverse_square_law_motion", 
        "test_class": "TestStudentImplementation"
    },
    {
        "name": "PROJECT_4_DoublePendulumSimulation",
        "test_module": "test_double_pendulum_simulation",
        "test_class": "TestStudentImplementation"
    }
]

PROJECT_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # Root directory

def load_test_module(project_config):
    """Load test module for a specific project."""
    project_name = project_config["name"]
    test_module_name = project_config["test_module"]
    
    # Determine project directory
    project_dir = os.path.join(PROJECT_BASE_DIR, project_name)
    test_dir = os.path.join(project_dir, 'tests')
    
    # Check if project exists
    if not os.path.isdir(project_dir):
        return None, f"Project directory not found: {project_dir}"
    
    if not os.path.isdir(test_dir):
        return None, f"Test directory not found: {test_dir}"
    
    # Add paths for imports
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    if os.path.join(project_dir, 'solution') not in sys.path:
        sys.path.insert(0, os.path.join(project_dir, 'solution'))
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)
    
    # Try to import the test module
    try:
        test_module_path = os.path.join(test_dir, f"{test_module_name}.py")
        if not os.path.exists(test_module_path):
            return None, f"Test file not found: {test_module_path}"
        
        spec = importlib.util.spec_from_file_location(test_module_name, test_module_path)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        
        return test_module, None
    except Exception as e:
        return None, f"Could not import test module '{test_module_name}': {e}"

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

    def run_project_tests(self, test_module, test_class_name):
        """Run tests for the specified project and collect results."""
        # Discover and run tests from the imported test module
        loader = unittest.TestLoader()
        
        try:
            # Get the test class from the module
            test_class = getattr(test_module, test_class_name)
            suite = loader.loadTestsFromTestCase(test_class)
        except AttributeError as e:
            self.results_data["feedback"] = f"Autograder error: Could not load test class '{test_class_name}' from test module. Details: {e}"
            self.results_data["status"] = "error"
            return
        except Exception as e:
            self.results_data["feedback"] = f"Autograder error: An unexpected error occurred while loading tests: {e}"
            self.results_data["status"] = "error"
            return

        if suite.countTestCases() == 0:
            self.results_data["feedback"] = f"No tests were found in class '{test_class_name}'. This might be due to missing student code or all tests being skipped."
            # Check if student code was actually available according to the test file's own checks
            if hasattr(test_module, 'STUDENT_CODE_AVAILABLE') and not test_module.STUDENT_CODE_AVAILABLE:
                 self.results_data["feedback"] += " The test file indicated that the student's code module could not be imported."
            self.results_data["status"] = "success" # Not an error, but 0 points
            return

        # Create an instance of our custom result class
        test_result_collector = PointsTestResult(stream=sys.stderr, descriptions=True, verbosity=2)
        
        suite.run(result=test_result_collector)

        self.total_score += test_result_collector.total_points_earned
        self.max_score += test_result_collector.total_points_possible
        self.results_data["tests"].extend(test_result_collector.detailed_results)
        
        # Update status if there are failures or errors
        if test_result_collector.errors or test_result_collector.failures:
            self.results_data["status"] = "failure"
    
    def run_all_projects(self):
        """Run tests for all available projects."""
        project_results = []
        
        for project_config in PROJECTS:
            project_name = project_config["name"]
            test_module, error = load_test_module(project_config)
            
            if test_module is None:
                # Project not available or has errors
                project_results.append({
                    "project": project_name,
                    "status": "skipped",
                    "reason": error
                })
                continue
            
            # Run tests for this project
            try:
                self.run_project_tests(test_module, project_config["test_class"])
                project_results.append({
                    "project": project_name,
                    "status": "completed"
                })
            except Exception as e:
                project_results.append({
                    "project": project_name,
                    "status": "error",
                    "reason": str(e)
                })
        
        # Generate final feedback
        feedback_lines = [
            "=== ODE Physical Model and BVP Projects Grading Report ===",
            f"Total score: {self.total_score} / {self.max_score} points.",
            ""
        ]
        
        # Add project-by-project summary
        for result in project_results:
            if result["status"] == "completed":
                feedback_lines.append(f"✓ {result['project']}: Tests completed")
            elif result["status"] == "skipped":
                feedback_lines.append(f"- {result['project']}: Skipped ({result['reason']})")
            else:
                feedback_lines.append(f"✗ {result['project']}: Error ({result['reason']})")
        
        feedback_lines.append("")
        
        if self.total_score == self.max_score and self.max_score > 0:
            feedback_lines.append("🎉 All available tests passed!")
        elif self.max_score == 0:
            feedback_lines.append("⚠️  No tests were run or no points were assigned.")
        else:
            feedback_lines.append("📊 Some tests failed. See details above or in the 'Checks' tab.")
        
        self.results_data["feedback"] = "\n".join(feedback_lines)

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
    
    grader = AutoGrader(project_name="ODE_Physical_Model_and_BVP_Projects")
    try:
        grader.run_all_projects()
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