#!/usr/bin/env python3
"""
Comprehensive test runner for Discord monitoring assistant.
"""
import argparse
import subprocess
import sys
import time
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.utils.quality_metrics import get_quality_collector, QualityMonitor


class TestRunner:
    """Comprehensive test runner with quality metrics tracking."""
    
    def __init__(self):
        self.start_time = time.time()
        self.results = {}
        self.collector = get_quality_collector()
        self.monitor = QualityMonitor(self.collector)
        
    def run_command(self, cmd: List[str], description: str) -> Dict[str, Any]:
        """Run a command and track results."""
        print(f"\n{'='*60}")
        print(f"🔄 {description}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            if success:
                print(f"✅ {description} - PASSED ({duration:.1f}s)")
            else:
                print(f"❌ {description} - FAILED ({duration:.1f}s)")
                print(f"Error output:\n{result.stderr}")
            
            # Record test result
            self.collector.record_test_result(
                test_name=description.lower().replace(' ', '_'),
                status='pass' if success else 'fail',
                duration=duration,
                error_message=result.stderr if not success else None
            )
            
            return {
                'success': success,
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"⏰ {description} - TIMEOUT ({duration:.1f}s)")
            
            self.collector.record_test_result(
                test_name=description.lower().replace(' ', '_'),
                status='fail',
                duration=duration,
                error_message='Test timed out'
            )
            
            return {
                'success': False,
                'duration': duration,
                'stdout': '',
                'stderr': 'Test timed out',
                'returncode': -1
            }
    
    def run_unit_tests(self) -> bool:
        """Run unit tests."""
        cmd = [
            'pytest', 'tests/unit/',
            '--verbose',
            '--tb=short',
            '--cov=lib',
            '--cov-report=term-missing',
            '--cov-report=html:htmlcov-unit',
            '--cov-report=xml:coverage-unit.xml',
            '--junitxml=junit-unit.xml',
            '-m', 'unit and not slow'
        ]
        
        result = self.run_command(cmd, "Unit Tests")
        self.results['unit_tests'] = result
        
        # Extract coverage from output
        if 'TOTAL' in result['stdout']:
            lines = result['stdout'].split('\n')
            for line in lines:
                if 'TOTAL' in line and '%' in line:
                    try:
                        coverage_str = line.split()[-1].rstrip('%')
                        coverage = float(coverage_str) / 100
                        self.collector.record_metric('test_coverage', coverage)
                        break
                    except (ValueError, IndexError):
                        pass
        
        return result['success']
    
    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        cmd = [
            'pytest', 'tests/integration/',
            '--verbose',
            '--tb=short',
            '--junitxml=junit-integration.xml',
            '-m', 'integration and not slow'
        ]
        
        result = self.run_command(cmd, "Integration Tests")
        self.results['integration_tests'] = result
        return result['success']
    
    def run_ml_tests(self) -> bool:
        """Run ML model tests."""
        cmd = [
            'pytest', 'tests/ml/',
            '--verbose',
            '--tb=short',
            '--junitxml=junit-ml.xml',
            '-m', 'ml and not slow'
        ]
        
        result = self.run_command(cmd, "ML Model Tests")
        self.results['ml_tests'] = result
        
        # Run accuracy assessment
        self._assess_ml_accuracy()
        
        return result['success']
    
    def _assess_ml_accuracy(self):
        """Assess ML model accuracy and record metrics."""
        try:
            from lib.importance_detector import MessageImportanceDetector
            
            detector = MessageImportanceDetector()
            
            # Test cases with expected results
            test_cases = [
                ("URGENT: Server is completely down!", True),
                ("Emergency maintenance needed ASAP", True),
                ("Group buy deadline tomorrow!", True),
                ("Meeting reminder for next week", True),
                ("Hello everyone, good morning", False),
                ("Thanks for the help", False),
                ("How's everyone doing today?", False),
                ("Nice weather outside", False),
            ]
            
            correct_predictions = 0
            false_positives = 0
            false_negatives = 0
            
            for message, expected_important in test_cases:
                result = detector.detect_importance(message)
                predicted_important = result.score >= 0.7
                
                if predicted_important == expected_important:
                    correct_predictions += 1
                elif predicted_important and not expected_important:
                    false_positives += 1
                elif not predicted_important and expected_important:
                    false_negatives += 1
            
            # Calculate metrics
            accuracy = correct_predictions / len(test_cases)
            false_positive_rate = false_positives / len([tc for tc in test_cases if not tc[1]])
            false_negative_rate = false_negatives / len([tc for tc in test_cases if tc[1]])
            
            # Record metrics
            self.collector.record_metric('importance_accuracy', accuracy)
            self.collector.record_metric('false_positive_rate', false_positive_rate)
            self.collector.record_metric('false_negative_rate', false_negative_rate)
            
            print(f"📊 ML Accuracy Assessment:")
            print(f"   Accuracy: {accuracy:.2f}")
            print(f"   False Positive Rate: {false_positive_rate:.2f}")
            print(f"   False Negative Rate: {false_negative_rate:.2f}")
            
        except Exception as e:
            print(f"⚠️ ML accuracy assessment failed: {e}")
    
    def run_performance_tests(self) -> bool:
        """Run performance tests."""
        cmd = [
            'pytest', 'tests/performance/',
            '--verbose',
            '--tb=short',
            '--junitxml=junit-performance.xml',
            '--benchmark-only',
            '--benchmark-json=benchmark-results.json',
            '-m', 'performance and not stress'
        ]
        
        result = self.run_command(cmd, "Performance Tests")
        self.results['performance_tests'] = result
        
        # Extract performance metrics from benchmark results
        self._extract_performance_metrics()
        
        return result['success']
    
    def _extract_performance_metrics(self):
        """Extract performance metrics from benchmark results."""
        try:
            if os.path.exists('benchmark-results.json'):
                with open('benchmark-results.json', 'r') as f:
                    benchmark_data = json.load(f)
                
                for benchmark in benchmark_data.get('benchmarks', []):
                    stats = benchmark.get('stats', {})
                    mean_time = stats.get('mean', 0)
                    
                    # Record performance metrics based on test name
                    test_name = benchmark.get('name', '')
                    if 'database' in test_name.lower():
                        self.collector.record_metric('database_query_time', mean_time)
                    elif 'api' in test_name.lower():
                        self.collector.record_metric('api_response_time', mean_time)
                    elif 'importance' in test_name.lower():
                        self.collector.record_metric('importance_scoring_time', mean_time)
                        
        except Exception as e:
            print(f"⚠️ Performance metrics extraction failed: {e}")
    
    def run_quality_tests(self) -> bool:
        """Run quality metric tests."""
        cmd = [
            'pytest', 'tests/quality/',
            '--verbose',
            '--tb=short',
            '--junitxml=junit-quality.xml',
            '-m', 'quality'
        ]
        
        result = self.run_command(cmd, "Quality Metrics Tests")
        self.results['quality_tests'] = result
        return result['success']
    
    def run_stress_tests(self) -> bool:
        """Run stress tests."""
        cmd = [
            'pytest', 'tests/performance/',
            '--verbose',
            '--tb=short',
            '--junitxml=junit-stress.xml',
            '-m', 'stress',
            '--maxfail=1'
        ]
        
        result = self.run_command(cmd, "Stress Tests")
        self.results['stress_tests'] = result
        return result['success']
    
    def run_code_quality_checks(self) -> bool:
        """Run code quality checks."""
        checks = [
            (['black', '--check', '--diff', '.'], "Code Formatting (Black)"),
            (['flake8', '.', '--count', '--select=E9,F63,F7,F82', '--show-source'], "Linting (Flake8)"),
            (['mypy', 'lib/', '--ignore-missing-imports'], "Type Checking (MyPy)"),
            (['bandit', '-r', 'lib/', '-f', 'json', '-o', 'bandit-report.json'], "Security Scan (Bandit)"),
        ]
        
        all_passed = True
        
        for cmd, description in checks:
            result = self.run_command(cmd, description)
            if not result['success']:
                all_passed = False
        
        return all_passed
    
    def generate_final_report(self):
        """Generate final test report with quality metrics."""
        total_time = time.time() - self.start_time
        
        print(f"\n{'='*80}")
        print(f"📋 FINAL TEST REPORT")
        print(f"{'='*80}")
        
        # Test results summary
        print(f"\n🧪 Test Results:")
        for test_type, result in self.results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"   {test_type.replace('_', ' ').title():<20} {status:<10} ({result['duration']:.1f}s)")
        
        # Overall success rate
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['success'])
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        print(f"\n📊 Summary:")
        print(f"   Total Test Suites: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   Success Rate: {success_rate:.1%}")
        print(f"   Total Time: {total_time:.1f}s")
        
        # Quality metrics
        report = self.collector.generate_quality_report()
        print(f"\n🎯 Quality Metrics:")
        print(f"   Overall Quality Score: {report.overall_score:.2f}")
        
        for metric in report.metrics:
            status_emoji = "✅" if metric.status == "pass" else "❌"
            print(f"   {metric.name:<25} {status_emoji} {metric.value:.3f} (threshold: {metric.threshold:.3f})")
        
        # Recommendations
        if report.recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Save detailed report
        detailed_report = {
            'timestamp': datetime.now().isoformat(),
            'total_time': total_time,
            'test_results': self.results,
            'success_rate': success_rate,
            'quality_score': report.overall_score,
            'quality_metrics': [
                {
                    'name': m.name,
                    'value': m.value,
                    'threshold': m.threshold,
                    'status': m.status
                } for m in report.metrics
            ],
            'recommendations': report.recommendations
        }
        
        with open('test-report.json', 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: test-report.json")
        
        # Return overall success
        quality_passed = report.overall_score >= 0.75
        tests_passed = success_rate >= 0.8
        
        overall_success = quality_passed and tests_passed
        
        if overall_success:
            print(f"\n🎉 ALL TESTS PASSED! System is ready for deployment.")
        else:
            print(f"\n⚠️ SOME TESTS FAILED. Review the results before deployment.")
            if not tests_passed:
                print(f"   - Test success rate ({success_rate:.1%}) below threshold (80%)")
            if not quality_passed:
                print(f"   - Quality score ({report.overall_score:.2f}) below threshold (0.75)")
        
        return overall_success


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description="Discord LLM Test Runner")
    parser.add_argument('--suite', choices=['all', 'unit', 'integration', 'ml', 'performance', 'quality', 'stress'], 
                       default='all', help='Test suite to run')
    parser.add_argument('--skip-quality-checks', action='store_true', help='Skip code quality checks')
    parser.add_argument('--include-stress', action='store_true', help='Include stress tests')
    parser.add_argument('--fast', action='store_true', help='Run only fast tests')
    
    args = parser.parse_args()
    
    print(f"🚀 Discord LLM Test Runner")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Suite: {args.suite}")
    
    runner = TestRunner()
    
    # Start quality monitoring
    runner.monitor.start_monitoring()
    
    try:
        overall_success = True
        
        # Code quality checks
        if not args.skip_quality_checks and args.suite in ['all']:
            if not runner.run_code_quality_checks():
                overall_success = False
        
        # Run selected test suites
        if args.suite in ['all', 'unit']:
            if not runner.run_unit_tests():
                overall_success = False
        
        if args.suite in ['all', 'integration']:
            if not runner.run_integration_tests():
                overall_success = False
        
        if args.suite in ['all', 'ml']:
            if not runner.run_ml_tests():
                overall_success = False
        
        if args.suite in ['all', 'performance']:
            if not runner.run_performance_tests():
                overall_success = False
        
        if args.suite in ['all', 'quality']:
            if not runner.run_quality_tests():
                overall_success = False
        
        if args.include_stress or args.suite == 'stress':
            if not runner.run_stress_tests():
                overall_success = False
        
        # Generate final report
        final_success = runner.generate_final_report()
        
        # Exit with appropriate code
        sys.exit(0 if final_success else 1)
        
    finally:
        runner.monitor.stop_monitoring()


if __name__ == '__main__':
    main()